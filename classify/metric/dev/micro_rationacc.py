from typing import Optional, List, Tuple, Dict

from sinkhorn import compute_alignment_cost, compute_entropy
import torch
import numpy as np
from classify.metric.abstract import AlignmentAverageMetric
from utils.utils import prod
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_raionale_metrics(
    preds: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
    targets: List[Dict[str, torch.LongTensor]],
    threshold: float = 0.1,
    absolute_threshold: bool = False,
) -> torch.float:
    """
        Computes metric across a list of instances of two sets of objects.

        :param preds: A list of (cost, alignment) tuples (each is n x m).
        :param targets: A list of LongTensors indicating the correct alignment.
        :return: The metric of the alignments.
        """
    # Initialize
    # Check lengths
    epsilon = 1e-10
    with torch.no_grad():
        assert len(preds) == len(targets)

        target_true = predicted_true = correct_true = 0
        target_true_c = predicted_true_c = correct_true_c = 0
        rationale_count = total_count = 0

        # Loop over alignments and add metric and count
        all_column_r = []
        all_predict_column_r = []
        ps = []
        rs = []
        fs = []

        for (cost, alignment), target in zip(preds, targets):
            if len(alignment.shape) == 3:
                # attention:
                # row_alignment = (-cost).softmax(dim=1); column_alignment = (-cost).softmax(dim=0)
                row_alignment, column_alignment = alignment[0], alignment[1]
                if absolute_threshold:
                    column_alignment = column_alignment >= threshold  # n
                    row_alignment = row_alignment >= threshold  # m
                else:
                    column_alignment = (
                        column_alignment
                        >= threshold / prod(column_alignment.shape[-2:])
                    ).float()  # n
                    row_alignment = (
                        row_alignment >= threshold / prod(column_alignment.shape[-2:])
                    ).float()  # m
                predict_row_r = column_alignment.sum(1).cpu().numpy() >= 1  # n
                predict_column_r = row_alignment.sum(0).cpu().numpy() >= 1  # m
            else:
                if absolute_threshold:
                    alignment = alignment >= threshold
                else:
                    alignment = (
                        alignment >= threshold / prod(alignment.shape[-2:])
                    ).float()  # n

                predict_row_r = alignment.sum(1).cpu().numpy() >= 1  # n
                predict_column_r = alignment.sum(0).cpu().numpy() >= 1  # m

            row_r = target["row_evidence"].cpu().numpy()  # n
            column_r = target["column_evidence"].cpu().numpy()  # m

            # print(len(column_r))
            # print(len(row_r))

            # For multirc, needs to chagne from sentence annotation to token annotation
            if "lengths" in target:
                # print('converting sent rationale to word rationale')
                # print(column_r)
                # print(sum(column_r))
                # print(predict_column_r)
                # print(sum(predict_column_r))
                column_r = rationale_sent_to_token(target["lengths"], column_r)
                predict_column_r = rationale_sent_to_token(
                    target["lengths"], predict_column_r
                )
                # print('after converting')
                # print(column_r)
                # print(predict_column_r)
                # print(sum(column_r))
                # print(sum(predict_column_r))
                # import sys; sys.exit()
            assert len(row_r) == len(predict_row_r)
            assert len(column_r) == len(predict_column_r)
            # print(f'predicted row: {predict_row_r }')
            # print(f'real row: {row_r }')
            f1 = f1_score(column_r, predict_column_r)
            fs.append(f1)

            all_column_r.extend(column_r)
            all_predict_column_r.extend(predict_column_r)

            if sum(column_r):
                rationale_count += sum(predict_column_r)
                total_count += len(predict_column_r)
            if sum(row_r):
                rationale_count += sum(predict_row_r)
                total_count += len(predict_row_r)
            target_true += np.sum(row_r == 1)  # .float()
            predicted_true += np.sum(predict_row_r == 1)  # .float()
            correct_true += np.sum((row_r == 1) * (predict_row_r == 1))  # .float()

            target_true_c += np.sum(column_r == 1)  # .float()
            predicted_true_c += np.sum(predict_column_r == 1)  # .float()
            correct_true_c += np.sum(
                (column_r == 1) * (predict_column_r == 1)
            )  # .float()

        precision = correct_true / (predicted_true + epsilon)
        recall = correct_true / (target_true + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)

        precision_c = correct_true_c / (predicted_true_c + epsilon)
        recall_c = correct_true_c / (target_true_c + epsilon)
        # f1_score_c = 2 * precision_c * recall_c / (precision_c + recall_c + epsilon)s
        f1_score_c = sum(fs) / len(fs)

        p_all = (correct_true + correct_true_c) / (
            predicted_true + predicted_true_c + epsilon
        )
        r_all = (correct_true + correct_true_c) / (
            target_true + target_true_c + epsilon
        )
        f1_all = 2 * p_all * r_all / (p_all + r_all + epsilon)

        rationale_ratio = rationale_count / total_count
        # print(f'p:',precision_c)
        # print(f'r:',recall_c)
        # print(f'f1:',f1_score_c)
        # for av in ['micro', 'macro', 'weighted' ]:
        #     print(av)
        #     print(f'p:',precision_score(all_column_r, all_predict_column_r)) #, average=av))
        #     print(f'r:',recall_score(all_column_r, all_predict_column_r)) #, average=av))
        #     print(f'f1:',f1_score(all_column_r, all_predict_column_r)) #, average=av))
        # import sys; sys.exit()

        precision_c = precision_score(
            all_column_r, all_predict_column_r, average="macro"
        )
        recall_c = recall_score(all_column_r, all_predict_column_r, average="macro")
        f1_score_c = f1_score(all_column_r, all_predict_column_r, average="macro")

    return (
        precision,
        recall,
        f1,
        precision_c,
        recall_c,
        f1_score_c,
        p_all,
        r_all,
        f1_all,
        rationale_ratio,
    )


def rationale_sent_to_token(lengths, rationales):
    total_l = sum(lengths)
    r_tk = np.zeros(total_l)
    for i, s in enumerate(list(rationales)):
        if s != 0:
            r_tk[sum(lengths[:i]) : sum(lengths[: i + 1])] = 1
    return r_tk
