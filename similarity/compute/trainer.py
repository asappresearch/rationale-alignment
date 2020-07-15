from copy import deepcopy
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from similarity.data import Sampler
from similarity.metric import Metric
from utils.utils import prod, save_model, NoamLR


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

        self.epochs = epochs
        self._step = 0
        self._best_metric = None
        self._best_model = None
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        self._log: Dict[str, Dict[int, float]] = {}

    def log(self, text: str, value: float, step: int) -> None:
        """Logs text and value to console and to tensorboard."""
        print(f"{text} = {value:.4f}")
        self.summary_writer.add_scalar(text, value, step)
        self._log.setdefault(text, {})[step] = value

    def train_step(self) -> None:
        """Run a training step over the training data."""
        self.model.train()
        self.optimizer.zero_grad()

        with torch.enable_grad():
            for i, batch in enumerate(self.train_sampler()):
                # Zero the gradients

                preds, targets = self.model(*batch)
                loss = self.loss_fn(preds, targets, self._step)

                # Optimize
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if i % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    if i % self.log_frequency == 0:
                        global_step = (len(self.train_sampler) * self._step) + i
                        self.log(
                            "Stats/Grad_Norm", self.model.gradient_norm, global_step
                        )
                    if self.scheduler is not None:
                        self.scheduler.step()  # Update learning rate schedule
                    self.optimizer.zero_grad()

                # Log loss and norms
                if i % self.log_frequency == 0:
                    global_step = (len(self.train_sampler) * self._step) + i
                    self.log(
                        "Stats/Learning_Rate", self.scheduler.get_lr()[0], global_step
                    )
                    self.log(f"Train/Loss/{self.loss_fn}", loss.item(), global_step)
                    # self.log('Stats/Grad_Norm', self.model.gradient_norm, global_step)
                    self.log("Stats/Param_Norm", self.model.parameter_norm, global_step)
                    for metric in self.extra_training_metrics:
                        self.log(
                            f"Train/Metric/{metric}",
                            metric(preds, targets).item(),
                            global_step,
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
            target["positives"] += num_preds
            target["negatives"] += num_preds

        # Move to cpu
        preds = [(cost.cpu(), alignment.cpu()) for (cost, alignment) in preds]
        targets = [
            {key: value.cpu() for key, value in target.items()} for target in targets
        ]

        # Compute new num_preds
        num_preds += len(preds)

        return preds, targets, num_preds

    def eval_step(self) -> None:
        """Run an evaluation step over the dev set."""
        self.model.eval()

        with torch.no_grad():
            all_preds, all_targets = [], []
            num_preds = 0

            for batch in self.dev_sampler():
                preds, targets = self.model(*batch)
                preds, targets, num_preds = self.postprocess(preds, targets, num_preds)

                all_preds += preds
                all_targets += targets

            dev_loss = self.dev_loss_fn(
                all_preds, all_targets, 10
            ).item()  # only report the loss of max_hinge_loss
            dev_metric = self.metric_fn(all_preds, all_targets).item()

        # Update best model
        sign = (-1) ** self.lower_is_better
        if self._best_metric is None or (sign * dev_metric > sign * self._best_metric):
            self._best_metric = dev_metric
            self._best_model = deepcopy(self.model.state_dict())

        # Log metrics
        # self.log('Stats/Learning_Rate', self.scheduler.get_lr()[0], self._step)
        self.log(f"Validation/Loss/{self.dev_loss_fn}", dev_loss, self._step)
        self.log(f"Validation/Metric/{self.metric_fn}", dev_metric, self._step)
        for metric in self.extra_validation_metrics:
            self.log(
                f"Validation/Metric/{metric}",
                metric(all_preds, all_targets).item(),
                self._step,
            )

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
            all_preds, all_targets = [], []
            num_preds = 0

            for i, scaling in enumerate(thresholds):
                # if i ==0 and self.model.args.alignment =='sinkhorn':
                #     assert scaling == 0

                # compute all the predictions
                sampler = self.dev_sampler if flavor == "Val" else self.test_sampler
                if i != 0 and self.model.args.alignment == "sinkhorn":
                    # Log metrics at different sparsity thresholds
                    costs, alignments = zip(*all_preds)
                    threshold_alignments = [
                        alignment
                        * (alignment >= scaling / prod(alignment.shape[-2:])).float()
                        for alignment in alignments
                    ]
                    all_preds = list(zip(costs, threshold_alignments))

                else:
                    all_preds, all_targets = [], []
                    num_preds = 0
                    for batch in sampler():
                        preds, targets = self.model(*batch, threshold=scaling)
                        preds, targets, num_preds = self.postprocess(
                            preds, targets, num_preds
                        )
                        all_preds += preds
                        all_targets += targets

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

        # Save model
        save_model(self.model, os.path.join(self.log_dir, model_name))

        return done

    def predict(
        self, num_predict: int = None
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

            for i, batch in enumerate(self.dev_sampler()):
                if i >= num_batches:
                    break

                sentences, preds, targets = self.model.forward_with_sentences(*batch)
                sentences = [(doc_1.cpu(), doc_2.cpu()) for doc_1, doc_2 in sentences]
                preds, targets, num_preds = self.postprocess(preds, targets, num_preds)

                all_sentences += sentences
                all_preds += preds
                all_targets += targets

        return all_sentences, all_preds, all_targets
