from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

from classify.metric.abstract import AlignmentMetric
from classify.metric.abstract import Metric

from utils.utils import prod


class BinaryCrossEntropyLoss(Metric):
    """Computes the hinge loss between aligned and un-aligned document pairs (for AskUbuntu).

    For each document, the loss is sum_ij |negative_similarity_i - positive_similarity_j + margin|
    i.e. sum over all positive/negative pairs
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize the MultiLabelNLLLoss.

        Parameters
        ----------
        weight : Optional[torch.Tensor]
            A manual rescaling weight given to each class.
            If given, has to be a Tensor of size N, where N is the
            number of classes.
        ignore_index : Optional[int], optional
            Specifies a target value that is ignored and does not
            contribute to the input gradient. When size_average is
            True, the loss is averaged over non-ignored targets.
        reduction : str, optional
            Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'.
            'none': no reduction will be applied,
            'mean': the output will be averaged
            'sum': the output will be summed.

        """
        super(BinaryCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def compute(
        self, logits: torch.Tensor, targets: torch.Tensor, step: int = 4
    ) -> torch.Tensor:
        """Computes the Negative log likelihood loss for multilabel.

        Parameters
        ----------
        pred: torch.Tensor
            input logits of shape (B x N)
        target: torch.LontTensor
            target tensor of shape (B x N)

        Returns
        -------
        loss: torch.float
            Multi label negative log likelihood loss, of shape (B)

        """

        targets = [t for target in targets for t in target["targets"]]
        targets = torch.stack(targets).float()

        logits = torch.stack(
            [torch.sum(cost * alignment) for cost, alignment in logits]
        )

        if self.ignore_index is not None:
            targets[:, self.ignore_index] = 0

        # if self.weight is None:
        #     self.weight = torch.ones(logits.size(1)).to(logits)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets
        )  # , weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss
