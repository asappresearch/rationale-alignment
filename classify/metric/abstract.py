from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch

from sinkhorn import compute_alignment_cost, compute_entropy


class Metric:
    @abstractmethod
    def compute(self, preds, targets, *argv) -> torch.float:
        pass

    def __call__(self, preds, targets, *argv) -> torch.float:
        return self.compute(preds, targets, *argv)

    def __str__(self) -> str:
        return self.__class__.__name__


class AlignmentMetric(Metric):
    """Computes the metric for saying one document is aligned with another."""

    @staticmethod
    def _compute_entropy(preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]]) -> torch.FloatTensor:
        """Computes the entropy term (epislon * H(P)) of each (cost, alignment) tuple in preds."""
        return torch.stack([compute_entropy(alignment) for cost, alignment in preds], dim=0)

    @staticmethod
    def _compute_cost(preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]]) -> torch.FloatTensor:
        """Computes the alignment cost of each (cost, alignment) tuple in preds."""
        return torch.stack([compute_alignment_cost(C=cost, P=alignment) for cost, alignment in preds], dim=0)

    @staticmethod
    def _compute_similarities(preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]]) -> torch.FloatTensor:
        """Computes the alignment similarities (i.e. -cost) of each (cost, alignment) tuple in preds."""
        return -AlignmentMetric._compute_cost(preds)

    @abstractmethod
    def compute(self,
                preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
                targets: Union[List[torch.LongTensor], List[Dict[str, torch.LongTensor]]],
                step: Optional[int]) -> torch.float:
        pass


class AlignmentAverageMetric(AlignmentMetric):
    """Computes a metric and averages it across documents."""

    def __init__(self, similar: Optional[bool] = None):
        self.similar = similar  # Whether to only include similar or only dissimilar examples

    @abstractmethod
    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        """
        Computes the metric and count of aligning two documents.

        :param cost: The cost of aligning sentence i with sentence j (matrix is n x m).
        :param alignment: The probability of aligning sentence i with sentence j (matrix is n x m).
        :param target: Whether the documents are similar or not.
        :return: The value.
        """
        pass

    def _compute_count(self,
                       cost: torch.FloatTensor,
                       alignment: torch.FloatTensor,
                       target: int) -> int:
        """
        Computes the count of items associated with the documents for the purpose of averaging.

        :param cost: The cost of aligning sentence i with sentence j (matrix is n x m).
        :param alignment: The probability of aligning sentence i with sentence j (matrix is n x m).
        :param target: Whether the documents are similar or not.
        :return: The count (typically either # of sentences or 1).
        """
        if self.similar is None:
            return 1

        return target == self.similar

    def compute(self,
                preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
                targets: List[Dict[str, torch.LongTensor]]) -> torch.float:
        """
        Computes metric across a list of instances of two sets of objects.

        :param preds: A list of (cost, alignment) tuples (each is n x m).
        :param targets: A list of LongTensors indicating the correct alignment.
        :return: The metric of the alignments.
        """
        # Initialize
        metric, count = 0, 0

        # Extract targets
        targets = [t.item() for target in targets for t in target['targets']]

        # Check lengths
        assert len(preds) == len(targets)

        # Loop over alignments and add metric and count
        for (cost, alignment), target in zip(preds, targets):
            new_count = self._compute_count(cost, alignment, target)

            if new_count == 0:
                continue

            count += new_count
            metric += self._compute_one(cost, alignment, target)

        # Average metric
        metric = metric / count if count != 0 else 0

        return metric

    def __str__(self) -> str:
        super_str = super(AlignmentAverageMetric, self).__str__()

        if self.similar is None:
            return super_str

        return ('Similar' if self.similar else 'Dissimilar') + super_str
