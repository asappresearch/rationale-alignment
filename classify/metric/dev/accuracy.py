import torch

from classify.metric.abstract import Metric
from sklearn.metrics import f1_score


class Accuracy(Metric):
    def compute(self, logit: torch.Tensor, targets: dict) -> torch.Tensor:
        """Computes the loss.

            Parameters
            ----------
            pred: Tensor
                input logits of shape (B x N)
            target: LontTensor
                target tensor of shape (B) or (B x N)

            Returns
            -------
            accuracy: torch.Tensor
                single label accuracy, of shape (B)

            """
        # If 2-dimensional, select the highest score in each row
        target = [t for target in targets for t in target["targets"]]
        target = torch.stack(target)
        pred = logit

        if len(target.size()) == 2:
            target = target.argmax(dim=1)
        acc = pred.argmax(dim=1) == target
        return acc.float().mean()


class F1(Metric):
    def compute(self, logit: torch.Tensor, targets: dict) -> torch.Tensor:
        target = [t for target in targets for t in target["targets"]]
        target = torch.stack(target)

        pred = logit.argmax(dim=1)

        if len(target.size()) == 2:
            target = target.argmax(dim=1)

        target = target.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        f1 = f1_score(target, pred, average="macro")
        return f1

