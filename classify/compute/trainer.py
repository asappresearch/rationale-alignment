from copy import deepcopy
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from classify.data import Sampler
from classify.metric import Metric
from utils.utils import prod, save_model, NoamLR
from classify.metric.dev.rationacc import compute_raionale_metrics
from classify.metric.loss.regularizor import ReguCost
from classify.metric.loss.rationaleloss import RationaleBCELoss


class AlignmentTrainer:
    def __init__(
        self,
        args,
        train_sampler: Sampler,
        dev_sampler: Sampler,
        test_sampler: Sampler,
        model: nn.Module,
        loss_fn: Metric,
        metric_fn: Metric,
        optimizer: Adam,
        scheduler: Optional[_LRScheduler],
        epochs: int,
        lower_is_better: bool = False,
        dev_loss_fn: Metric = None,
        extra_training_metrics: Optional[List[Metric]] = None,
        extra_validation_metrics: Optional[List[Metric]] = None,
        log_dir: Optional[str] = None,
        log_frequency: int = 20,
        gradient_accumulation_steps: int = 1,
        sparsity_thresholds: Optional[List[float]] = None,
        saved_step: Optional[int] = 0,
    ):
        self.args = args
        self.train_sampler = train_sampler
        self.dev_sampler = dev_sampler
        self.test_sampler = test_sampler
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lower_is_better = lower_is_better
        self.dev_loss_fn = dev_loss_fn or self.loss_fn
        self.extra_training_metrics = extra_training_metrics or []
        self.extra_validation_metrics = extra_validation_metrics or []
        self.validation_metrics = [self.metric_fn] + self.extra_validation_metrics
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        self.sparsity_thresholds = sparsity_thresholds or []
        self.gradient_accumulation_steps = gradient_accumulation_steps

        device = (
            torch.device(args.gpu) if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = device

        self.epochs = epochs
        self.save_every = args.save_every if args.save_every else math.ceil(epochs / 10)
        self._best_metric = None
        self._best_model = None
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        self._log: Dict[str, Dict[int, float]] = {}

        # Easier tensorboard plotting when resuming experiments from checkpoint
        self._step = 0 if not saved_step else saved_step
        # self._examples = 0 if not saved_step else saved_step*len(train_sampler)*3
        self._examples = 0 if not saved_step else saved_step * len(train_sampler.data)

        # Experimenting with dynamic epsilon for sinkhorn to acheive best balance between speed and performance.
        self.epsilon_min = 1e-4
        self.epsilonpower = math.log(self.args.epsilon / self.epsilon_min) / math.log(
            self.args.step_aneal
        )
        self.epsilon_aneal_start = self.args.epochs - 5 - self.args.step_aneal
        self.epsilon = self.args.epsilon

        # Adding any regularization if needed
        self.regu_fn = ReguCost(args.cost_regulate, args.regu_type, device)

        # Adding extra loss if decided to train model with rationale
        if self.args.train_rationale:
            self.rationale_loss = RationaleBCELoss(domain=self.model.domain)

    def log(self, text: str, value: float, step: int) -> None:
        """Logs text and value to console and to tensorboard."""
        print(f"{text} = {value:.4f}")
        self.summary_writer.add_scalar(text, value, step)
        self._log.setdefault(text, {})[step] = value

    def train_step(self) -> None:
        """Run a training step over the training data."""
        self.model.train()
        self.optimizer.zero_grad()

        if self._step > self.epsilon_aneal_start and self.epsilon_aneal_start > 0:
            print("using strange anealing process")
            self.epsilon = (
                self.args.epsilon
                / (self._step - self.epsilon_aneal_start) ** self.epsilonpower
            )
        else:
            self.epsilon = self.args.epsilon

        with torch.enable_grad():
            for i, batch in enumerate(self.train_sampler()):
                # Zero the gradients
                # self._examples += len(batch)
                self._examples += len(batch[0])

                preds, targets, logits = self.model(
                    *batch, threshold=0.0, epsilon=self.epsilon
                )

                loss = 0
                if self._step < self.args.stop_cls_step:
                    loss += self.loss_fn(logits, targets)

                if self.args.cost_regulate:
                    loss += self.regu_fn(preds, targets, self._step)

                if self.args.train_rationale:
                    rationale_loss = self.rationale_loss(preds, targets)
                    loss = (loss + self.args.rationale_weight * rationale_loss) / (
                        1 + self.args.rationale_weight
                    )

                # Optimize
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if i % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    if i % self.log_frequency == 0:
                        self.log(
                            "Stats/Grad_Norm", self.model.gradient_norm, self._examples
                        )
                    if self.scheduler is not None:
                        self.scheduler.step()  # Update learning rate schedule
                    self.optimizer.zero_grad()

                # Log loss and norms
                if i % self.log_frequency == 0:
                    self.log(
                        "Stats/Learning_Rate",
                        self.scheduler.get_lr()[0],
                        self._examples,
                    )
                    self.log(f"Train/Loss/{self.loss_fn}", loss.item(), self._examples)
                    self.log(
                        "Stats/Param_Norm", self.model.parameter_norm, self._examples
                    )
                    self.log("Stats/epsilon", self.epsilon, self._examples)

                    if self.args.train_rationale:
                        self.log(
                            f"Train/Loss/rationale_loss",
                            rationale_loss.item(),
                            self._examples,
                        )

                    self.log(
                        f"Train/Metric/{self.metric_fn}",
                        self.metric_fn(logits, targets).item(),
                        self._examples,
                    )

                    (
                        premise_p,
                        premise_r,
                        premise_f1,
                        hypothesis_p,
                        hypothesis_r,
                        hypothesis_f1,
                        p,
                        r,
                        f,
                        r_ratio,
                    ) = compute_raionale_metrics(preds, targets)
                    self.log(f"Train/Metric/rationale_ratio", r_ratio, self._examples)
                    self.log(f"Train/Metric/precision", p, self._examples)
                    self.log(f"Train/Metric/recall", r, self._examples)
                    self.log(f"Train/Metric/rationalef1", f, self._examples)
                    if self.model.domain == "snli":
                        self.log(
                            f"Train/Metric/premise_precision",
                            premise_p,
                            self._examples,
                        )
                        self.log(
                            f"Train/Metric/premise_recall", premise_r, self._examples,
                        )
                        self.log(f"Train/Metric/premise_f1", premise_f1, self._examples)
                        self.log(
                            f"Train/Metric/hypothesis_precision",
                            hypothesis_p,
                            self._examples,
                        )
                        self.log(
                            f"Train/Metric/hypothesis_recall",
                            hypothesis_r,
                            self._examples,
                        )
                        self.log(
                            f"Train/Metric/hypothesis_f1",
                            hypothesis_f1,
                            self._examples,
                        )

                    for metric in self.extra_training_metrics:
                        if str(metric) in ["F1", "Accuracy"]:
                            self.log(
                                f"Train/Metric/{metric}",
                                metric(logits, targets).item(),
                                self._examples,
                            )
                        else:
                            self.log(
                                f"Train/Metric/{metric}",
                                metric(preds, targets).item(),
                                self._examples,
                            )

            # Zero the gradients when exiting a train step
            self.optimizer.zero_grad()

    def postprocess(
        self,
        preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        targets: List[Dict[str, torch.LongTensor]],
        num_preds: int = 0,
    ) -> Tuple[
        List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        List[Dict[str, torch.LongTensor]],
        int,
    ]:
        """Post-processes predictions and targets by moving them to cpu and fixing indexing."""
        # Fix indexing
        for target in targets:
            target["scope"] += num_preds
            if "positives" in target:
                target["positives"] += num_preds
                target["negatives"] += num_preds

        def mapcpu(val):
            if torch.is_tensor(val):
                return val.cpu()
            else:
                return val

        # Move to cpu
        preds = [(cost.cpu(), alignment.cpu()) for (cost, alignment) in preds]
        targets = [
            {key: mapcpu(value) for key, value in target.items()} for target in targets
        ]

        # Compute new num_preds
        num_preds += len(preds)

        return preds, targets, num_preds

    def eval_step(self, flavor="Validation") -> None:
        """Run an evaluation step over the dev set."""

        assert flavor in ["Test", "Validation"]

        self.model.eval()

        with torch.no_grad():
            for scaling in self.sparsity_thresholds:
                all_preds, all_targets = [], []
                all_logits = []
                num_preds = 0

                # compute all the predictions
                sampler = self.test_sampler if flavor == "Test" else self.dev_sampler
                for batch in sampler():
                    preds, targets, logits = self.model(
                        *batch, threshold=scaling, epsilon=self.epsilon
                    )
                    all_logits.append(logits)
                    preds, targets, num_preds = self.postprocess(
                        preds, targets, num_preds
                    )
                    all_preds += preds
                    all_targets += targets

                # Log metrics
                all_logits = torch.cat(all_logits, dim=0).cpu()
                dev_loss = self.dev_loss_fn(
                    all_logits, all_targets
                ).item()  # only report the loss of max_hinge_loss
                if self.args.train_rationale:
                    rationale_loss = self.rationale_loss(all_preds, all_targets)

                dev_metric = self.metric_fn(all_logits, all_targets).item()
                check_dev_metric = dev_metric

                check_dev = (
                    scaling == 0.1
                )  # and self.model.alignment=='attention') or ((scaling==0.1) and self.model.alignment=='attention')
                if check_dev:
                    check_dev_metric = dev_metric
                (
                    premise_p,
                    premise_r,
                    premise_f1,
                    hypothesis_p,
                    hypothesis_r,
                    hypothesis_f1,
                    p,
                    r,
                    f,
                    r_ratio,
                ) = compute_raionale_metrics(all_preds, all_targets, scaling)
                self.log(
                    f"{flavor}_{scaling}/Metric/rationale_ratio", r_ratio, self._step,
                )
                self.log(f"{flavor}_{scaling}/Metric/precision", p, self._step)
                self.log(f"{flavor}_{scaling}/Metric/recall", r, self._step)
                self.log(f"{flavor}_{scaling}/Metric/rationalef1", f, self._step)
                if self.model.domain == "snli":
                    self.log(
                        f"{flavor}_{scaling}/Metric/premise_precision",
                        premise_p,
                        self._step,
                    )
                    self.log(
                        f"{flavor}_{scaling}/Metric/premise_recall",
                        premise_r,
                        self._step,
                    )
                    self.log(
                        f"{flavor}_{scaling}/Metric/premise_f1", premise_f1, self._step,
                    )
                    self.log(
                        f"{flavor}_{scaling}/Metric/hypothesis_precision",
                        hypothesis_p,
                        self._step,
                    )
                    self.log(
                        f"{flavor}_{scaling}/Metric/hypothesis_recall",
                        hypothesis_r,
                        self._step,
                    )
                    self.log(
                        f"{flavor}_{scaling}/Metric/hypothesis_f1",
                        hypothesis_f1,
                        self._step,
                    )

                # Log metrics
                # self.log('Stats/Learning_Rate', self.scheduler.get_lr()[0], self._step)
                print(f"size of all preds", len(all_preds))
                print(f"size of all targets", len(all_targets))
                self.log(
                    f"{flavor}_{scaling}/Loss/{self.dev_loss_fn}", dev_loss, self._step
                )
                if self.args.train_rationale:
                    self.log(
                        f"{flavor}_{scaling}/Loss/rationale_loss",
                        rationale_loss,
                        self._step,
                    )
                self.log(
                    f"{flavor}_{scaling}/Metric/{self.metric_fn}",
                    dev_metric,
                    self._step,
                )
                for metric in self.extra_validation_metrics:
                    if str(metric) in ["F1", "Accuracy"]:
                        self.log(
                            f"{flavor}_{scaling}/Metric/{metric}",
                            metric(all_logits, all_targets).item(),
                            self._step,
                        )
                    else:
                        self.log(
                            f"{flavor}_{scaling}/Metric/{metric}",
                            metric(all_preds, all_targets).item(),
                            self._step,
                        )

        # Update best model
        sign = (-1) ** self.lower_is_better
        if self._best_metric is None or (
            sign * check_dev_metric > sign * self._best_metric
        ):
            self._best_metric = check_dev_metric
            self._best_model = deepcopy(self.model.state_dict())

        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(dev_loss)
            else:
                self.scheduler.step()

    def test_step(
        self, flavor="Test", thresholds=[0.0, 0.01, 0.1, 0.5, 1, 3, 5]
    ) -> None:
        """Run an evaluation step over the dev set."""
        self.model.eval()

        assert flavor in ["Test", "Val"]

        with torch.no_grad():
            for i, scaling in enumerate(thresholds):
                all_preds, all_targets = [], []
                all_logits = []
                num_preds = 0

                # compute all the predictions
                sampler = self.dev_sampler if flavor == "Val" else self.test_sampler
                for batch in sampler():
                    preds, targets, logits = self.model(
                        *batch, threshold=scaling, epsilon=self.epsilon
                    )

                    all_logits.append(logits)
                    preds, targets, num_preds = self.postprocess(
                        preds, targets, num_preds
                    )
                    all_preds += preds
                    all_targets += targets

                # Log metrics
                all_logits = torch.cat(all_logits, dim=0).cpu()
                dev_loss = self.dev_loss_fn(
                    all_logits, all_targets
                ).item()  # only report the loss of max_hinge_loss
                dev_metric = self.metric_fn(all_logits, all_targets).item()

                (
                    premise_p,
                    premise_r,
                    premise_f1,
                    hypothesis_p,
                    hypothesis_r,
                    hypothesis_f1,
                    p,
                    r,
                    f,
                    r_ratio,
                ) = compute_raionale_metrics(all_preds, all_targets, scaling)

                self.log(f"{flavor}/Metric/rationale_ratio", r_ratio, i)
                self.log(f"{flavor}/Metric/precision", p, i)
                self.log(f"{flavor}/Metric/recall", r, i)
                self.log(f"{flavor}/Metric/rationalef1", f, i)
                self.log(f"{flavor}/Metric/tradeoff", p, r)
                if self.model.domain == "snli":
                    self.log(f"{flavor}/Metric/premise_precision", premise_p, i)
                    self.log(f"{flavor}/Metric/premise_recall", premise_r, i)
                    self.log(f"{flavor}/Metric/premise_f1", premise_f1, i)
                    self.log(f"{flavor}/Metric/hypothesis_precision", hypothesis_p, i)
                    self.log(f"{flavor}/Metric/hypothesis_recall", hypothesis_r, i)
                    self.log(f"{flavor}/Metric/hypothesis_f1", hypothesis_f1, i)
                    dev_loss = self.dev_loss_fn(
                        all_preds, all_targets, 10
                    ).item()  # only report the loss of max_hinge_loss
                    dev_metric = self.metric_fn(all_preds, all_targets).item()

                # Log metrics
                # self.log('Stats/Learning_Rate', self.scheduler.get_lr()[0], scaling)
                self.log(f"{flavor}/Metric/threshold", scaling, i)
                self.log(f"{flavor}/Loss/{self.dev_loss_fn}", dev_loss, i)
                self.log(f"{flavor}/Metric/{self.metric_fn}", dev_metric, i)
                for metric in self.extra_validation_metrics:
                    if str(metric) in ["F1", "Accuracy"]:
                        self.log(
                            f"{flavor}/Metric/{metric}",
                            metric(all_logits, all_targets).item(),
                            i,
                        )
                    else:
                        self.log(
                            f"{flavor}/Metric/{metric}",
                            metric(all_preds, all_targets).item(),
                            i,
                        )

    def step(self) -> bool:
        """Train until the next checkpoint, and evaluate.
        Returns
        ------
        bool
            Whether the computable has completed.
        """
        self.train_step()
        self.eval_step()
        if self._step % self.args.test_every == 0:
            self.eval_step("Test")

        # Simple stopping rule, if we exceed the max number of steps
        self._step += 1
        done = self._step >= self.epochs
        if done:
            model_name = "model.pt"
            self.model.load_state_dict(self._best_model)

            # Save metrics
            with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
                json.dump(self._log, f, indent=4, sort_keys=True)
        else:
            model_name = f"model_{self._step - 1}.pt"

        if self._step % self.save_every == 0 or done:
            # Save model
            save_model(self.model, os.path.join(self.log_dir, model_name))

        return done

    def predict(
        self, num_predict: int = None, fold="dev", threshold: float = 0.0
    ) -> Tuple[
        List[Tuple[torch.LongTensor, torch.LongTensor]],
        List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        List[Any],
    ]:
        """Run predictions over the dev set and return sentences (as indices), predictions, and targets."""
        self.model.eval()

        if num_predict is not None:
            num_batches = int(math.ceil(num_predict / self.dev_sampler.batch_size))
        else:
            num_batches = float("inf")

        with torch.no_grad():
            all_sentences, all_preds, all_targets = [], [], []
            num_preds = 0

            sampler = self.dev_sampler if fold == "dev" else self.test_sampler

            for i, batch in enumerate(sampler()):
                if i >= num_batches:
                    break

                sentences, preds, targets = self.model.forward_with_sentences(
                    *batch, threshold=threshold
                )
                sentences = [(doc_1.cpu(), doc_2.cpu()) for doc_1, doc_2 in sentences]
                preds, targets, num_preds = self.postprocess(preds, targets, num_preds)

                all_sentences += sentences
                all_preds += preds
                all_targets += targets

        return all_sentences, all_preds, all_targets
