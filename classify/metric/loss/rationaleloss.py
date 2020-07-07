from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

from classify.metric.abstract import Metric


class RationaleBCELoss(Metric):
    def __init__(self, domain):
        self.domain = domain

    def compute(
        self,
        preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        targets: List[Dict[str, torch.LongTensor]],
    ) -> torch.float:
        rationale_loss = 0
        for (c, alignment), target in zip(preds, targets):
            # gold_column_r = torch.stack([target['column_evidence'] for target in targets]).to(self.device)

            # rationale_pred = []
            if len(alignment.shape) == 3:
                # attention:
                # row_alignment = (-cost).softmax(dim=1); column_alignment = (-cost).softmax(dim=0)
                row_alignment, column_alignment = alignment[0], alignment[1]
                predict_row_r = column_alignment.sum(1)
                predict_column_r = row_alignment.sum(0)
            else:
                predict_row_r = alignment.sum(1)
                predict_column_r = alignment.sum(0)

            gold_column_r = target["column_evidence"].float()
            rationale_loss += F.binary_cross_entropy_with_logits(
                predict_column_r, gold_column_r
            ) / len(targets)
            if self.domain == "snli":
                gold_row_r = target["row_evidence"].float()
                rationale_loss += F.binary_cross_entropy_with_logits(
                    predict_row_r, gold_row_r
                ) / len(targets)

        return rationale_loss
