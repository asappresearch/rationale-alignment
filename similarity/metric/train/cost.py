import torch

from similarity.metric.abstract import AlignmentAverageMetric


class CostRange(AlignmentAverageMetric):
    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return cost.max() - cost.min()


class CostMin(AlignmentAverageMetric):
    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return cost.min()


class CostMax(AlignmentAverageMetric):
    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return cost.max()


class CostMean(AlignmentAverageMetric):
    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return cost.mean()


class CostMedian(AlignmentAverageMetric):
    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return cost.median()


class CostSign(AlignmentAverageMetric):
    def __init__(self, positive: bool, similar: bool):
        super(CostSign, self).__init__(similar=similar)
        self.positive = positive  # Whether to count positive or negative costs

    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        if self.positive:
            return torch.sum(cost > 0).float()
        return torch.sum(cost < 0).float()

    def __str__(self) -> str:
        return f'Num{"Positive" if self.positive else "Negative"}CostWhen{"Similar" if self.similar else "Dissimilar"}'
