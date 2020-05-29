from abc import abstractmethod
from typing import Dict, List, Tuple

import torch

from similarity.metric.abstract import AlignmentMetric
from similarity.metric.dev.auc import AUC as RawAUC


class AUC(RawAUC, AlignmentMetric):
    """Computes AUC of aligning documents."""

    def compute(self,
                preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
                targets: List[Dict[str, torch.LongTensor]]) -> torch.float:
        similarities = self._compute_similarities(preds)

        all_true, all_pred = [], []
        for target in targets:
            # Get true and pred for example
            true, pred = target['targets'], similarities[target['scope']]

            all_true += true.numpy().tolist()
            all_pred += pred.numpy().tolist()

        auc = super(AUC, self).compute(all_pred, all_true)

        return auc

    def __str__(self) -> str:
        if self.max_fpr == 1.0:
            return self.__class__.__name__
        return f'{self.__class__.__name__}_{self.max_fpr}'


class SimilarityMetric(AlignmentMetric):
    """Computes the mean of document similarity metrics."""

    @abstractmethod
    def _compute_one(self,
                     true: torch.FloatTensor,
                     pred: torch.FloatTensor) -> torch.float:
        """
        Computes the metric for one example.

        :param true: The true binary values (sorted in order of pred score).
        :param pred: The predicted scores (sorted in order of pred score).
        :return: The metric computed on true and pred.
        """
        pass

    def compute(self,
                preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
                targets: List[Dict[str, torch.LongTensor]]) -> torch.float:
        """
        Computes the mean metric.

        :param preds: A list of (cost, alignment) tuples.
        :param targets: A list of dictionaries mapping to the indices of the targets and scope.
        :return: The mean metric.
        """
        similarities = self._compute_similarities(preds)

        metrics = []
        for target in targets:
            # Get true and pred for example
            true, pred = target['targets'], similarities[target['scope']]

            # Sort based on pred
            argsort = torch.argsort(pred, descending=True)
            true, pred = true[argsort], pred[argsort]

            # Convert true to float
            true = true.float()

            # Compute metric
            metric = self._compute_one(true, pred)
            metrics.append(metric)

        mean_metric = torch.mean(torch.FloatTensor(metrics))

        return mean_metric


class MAP(SimilarityMetric):
    """Computes mean average precision."""

    def _compute_one(self,
                     true: torch.FloatTensor,
                     pred: torch.FloatTensor) -> torch.float:
        cumsum = torch.cumsum(true, dim=0)
        rank = torch.arange(len(true), dtype=torch.float) + 1
        precisions = true * cumsum / rank
        average_precision = torch.sum(precisions) / torch.sum(true)

        return average_precision


class MRR(SimilarityMetric):
    """Computes mean reciprocal rank."""

    def _compute_one(self,
                     true: torch.FloatTensor,
                     pred: torch.FloatTensor) -> torch.float:
        # The rank is the index of the first nonzero element + 1
        rank = torch.nonzero(true)[0, 0] + 1
        reciprocal_rank = 1 / rank.float()

        return reciprocal_rank


class Precision(SimilarityMetric):
    """Computes precision at n."""

    def __init__(self, n: int):
        super(Precision, self).__init__()
        self.n = n

    def _compute_one(self,
                     true: torch.FloatTensor,
                     pred: torch.FloatTensor) -> torch.float:
        return torch.mean(true[:self.n])

    def __str__(self) -> str:
        return f'{self.__class__.__name__}_at_{self.n}'
