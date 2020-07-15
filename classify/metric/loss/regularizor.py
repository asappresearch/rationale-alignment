from typing import Dict, List, Tuple

import torch

from classify.metric.abstract import AlignmentMetric
from utils.utils import prod


class ReguCost(AlignmentMetric):
    """Computes the hinge loss between aligned and un-aligned document pairs (for AskUbuntu).

    For each document, the loss is sum_ij |negative_similarity_i - positive_similarity_j + margin|
    i.e. sum over all positive/negative pairs
    """

    def __init__(self, cost_lambda, ltype, device):
        super(ReguCost, self).__init__()

        self.lmbd = cost_lambda
        self.type = ltype
        self.device = device

    def compute(
        self,
        preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        targets: List[Dict[str, torch.LongTensor]],
        step: int = 4,
    ) -> torch.float:
        reg = 0
        for (cost, alignment) in preds:
            # reg += torch.sum(cost<0) /prod(cost.shape[-2:])
            reg += self.cost_reg(cost)  # ideally, cost needs to be larger,

        return reg * self.lmbd

    def cost_reg(self, cost):
        if self.type == "l0":
            reg = torch.mean((cost < 0).float().to(self.device))
        if self.type == "l0.5":
            reg = -torch.mean(cost * (cost < 0).float().to(self.device))
        if self.type == "l1":
            reg = -torch.mean(cost)
        # if self.type == 'l2':
        #     reg = 0

        return reg
