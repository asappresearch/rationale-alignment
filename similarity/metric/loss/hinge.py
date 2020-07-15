from typing import Dict, List, Tuple

import torch

from similarity.metric.abstract import AlignmentMetric
from utils.utils import prod


class HingeLoss(AlignmentMetric):
    """Computes the hinge loss between aligned and un-aligned document pairs (for AskUbuntu).

    For each document, the loss is sum_ij |negative_similarity_i - positive_similarity_j + margin|
    i.e. sum over all positive/negative pairs
    """

    def __init__(self, margin: float, pooling: str = "max", alpha: float = 0.5):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.pooling = pooling
        self.alpha = alpha

    def compute(
        self,
        preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        targets: List[Dict[str, torch.LongTensor]],
        step: int = 4,
    ) -> torch.float:
        similarities = self._compute_similarities(preds)

        loss = count = 0
        for target in targets:
            positive_similarities, negative_similarities = (
                similarities[target["positives"]],
                similarities[target["negatives"]],
            )
            diff_similarities = negative_similarities.unsqueeze(
                dim=1
            ) - positive_similarities.unsqueeze(
                dim=0
            )  # num_negatives x num_positives

            if self.pooling == "max":
                diff_similarities = diff_similarities.max(dim=0)[
                    0
                ]  # num_positives (max across negatives)
            elif self.pooling == "smoothmax":
                alpha = self.alpha / (step + 1) if step < 5 else 0
                diff_similarities = (1 - alpha) * diff_similarities.max(dim=0)[
                    0
                ] + alpha * diff_similarities.mean(
                    dim=0
                )  # num_positives (max across negatives)
            elif self.pooling == "average":
                diff_similarities = diff_similarities.mean(
                    dim=0
                )  # num_positives (mean across negatives)
            elif self.pooling == "none":
                pass
            else:
                raise ValueError(f'Pooling type "{self.pooling}" not supported')

            loss += torch.sum(torch.clamp(diff_similarities + self.margin, min=0))
            count += int(prod(diff_similarities.shape))

        loss = loss / count

        return loss
