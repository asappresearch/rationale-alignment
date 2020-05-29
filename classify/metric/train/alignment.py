from typing import Optional

from sinkhorn import compute_alignment_cost, compute_entropy
import torch

from classify.metric.abstract import AlignmentAverageMetric


class AlignmentSum(AlignmentAverageMetric):
    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return alignment.sum()


class AlignmentCount(AlignmentAverageMetric):
    """Returns the number of non-zero entries (possibly normalized)."""

    def __init__(self,
                 normalize: str = 'none',  # 'none', 'min' to norm by min(n, m), 'full' to norm by (nm)
                 threshold_scaling: float = 1.0,  # how much to scale the 1 / (n * m) threshold
                 similar: Optional[bool] = None):
        assert normalize in ['none', 'min', 'full']

        super(AlignmentCount, self).__init__(similar=similar)
        self.normalize = normalize

    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        n, m = alignment.shape[-2:]
        count = torch.sum(alignment != 0).float()

        if self.normalize == 'full':
            return count / (n * m)
        elif self.normalize == 'min':
            return count / min(n, m)

        return count

    def __str__(self) -> str:
        string = super(AlignmentCount, self).__str__()

        if self.normalize != 'none':
            string += '_normalized_by_' + ('min_nm' if self.normalize == 'min' else 'nm')

        return string


class AlignmentMarginalError(AlignmentAverageMetric):
    """Returns the average absolute error of either the row or column marginal, assuming uniform marginal."""

    def __init__(self, side: int, similar: Optional[bool] = None):
        super(AlignmentMarginalError, self).__init__(similar=similar)
        assert side in {0, 1}
        self.side = side

    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        device = alignment.device
        marginal_dim = -2 if self.side == 0 else -1
        sum_dim = -1 if self.side == 0 else -2
        marginal = torch.ones(alignment.size(marginal_dim), device=device) / alignment.size(marginal_dim)
        marginal_hat = alignment.sum(sum_dim)
        error = torch.abs(marginal - marginal_hat).mean()

        return error


class AlignmentRowMarginalError(AlignmentMarginalError):
    def __init__(self, similar: Optional[bool] = None):
        super(AlignmentRowMarginalError, self).__init__(side=0, similar=similar)


class AlignmentColumnMarginalError(AlignmentMarginalError):
    def __init__(self, similar: Optional[bool] = None):
        super(AlignmentColumnMarginalError, self).__init__(side=1, similar=similar)


class AlignmentCost(AlignmentAverageMetric):
    """Computes the cost of aligning two objects."""

    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return compute_alignment_cost(C=cost, P=alignment)


class AlignmentEntropy(AlignmentAverageMetric):
    """Computes the entropy of aligning two objects."""

    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return compute_entropy(P=alignment)


class AlignmentEpsilonEntropy(AlignmentAverageMetric):
    """Computes epsilon times the entropy of aligning two objects."""

    def __init__(self, epsilon: float, similar: Optional[bool] = None):
        super(AlignmentEpsilonEntropy, self).__init__(similar=similar)
        self.epsilon = epsilon

    def _compute_one(self,
                     cost: torch.FloatTensor,
                     alignment: torch.FloatTensor,
                     target: int) -> torch.float:
        return self.epsilon * compute_entropy(P=alignment)
