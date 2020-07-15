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

        # target_true = predicted_true = correct_true = 0
        # target_true_c = predicted_true_c = correct_true_c = 0
        rationale_count = total_count = 0

        # Initialize the result list for row and columns
        p_r = []
        r_r = []
        f_r = []
        p_c = []
        r_c = []
        f_c = []

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
                column_r = rationale_sent_to_token(target["lengths"], column_r)
                predict_column_r = rationale_sent_to_token(
                    target["lengths"], predict_column_r
                )

            assert len(row_r) == len(predict_row_r)
            assert len(column_r) == len(predict_column_r)

            if (sum(column_r) + sum(predict_column_r)) != 0:
                p_instance = precision_score(
                    column_r, predict_column_r
                )  # , average=av))
                r_instance = recall_score(column_r, predict_column_r)  # , average=av))
                f_instance = f1_score(column_r, predict_column_r)
                p_c.append(p_instance)
                r_c.append(r_instance)
                f_c.append(f_instance)
                rationale_count += sum(predict_column_r)
                total_count += len(predict_column_r)
                # if (p_instance+r_instance) !=0:
                #     assert p_instance!=r_instance
            else:
                print("zero rationale annotatino")
            if not "lengths" in target:
                # if sum(row_r): # + sum(predict_row_r):
                if "lengths" in target:
                    print("multirc has no ratioanle anotation on QA pairs")
                    import sys

                    sys.exit()
                p_instance = precision_score(row_r, predict_row_r)  # , average=av))
                r_instance = recall_score(row_r, predict_row_r)  # , average=av))
                f_instance = f1_score(row_r, predict_row_r)
                p_r.append(p_instance)
                r_r.append(r_instance)
                f_r.append(f_instance)
                rationale_count += sum(predict_row_r)
                total_count += len(predict_row_r)

    rationale_ratio = rationale_count / total_count

    pc = sum(p_c) / (len(p_c) + epsilon)
    rc = sum(r_c) / (len(r_c) + epsilon)
    fc = sum(f_c) / (len(f_c) + epsilon)

    pr = sum(p_r) / (len(p_r) + epsilon)
    rr = sum(r_r) / (len(r_r) + epsilon)
    fr = sum(f_r) / (len(f_r) + epsilon)

    p_all = p_c + p_r
    r_all = r_c + r_r
    f_all = f_c + f_r

    p = sum(p_all) / (len(p_all) + epsilon)
    r = sum(r_all) / (len(r_all) + epsilon)
    f = sum(f_all) / (len(f_all) + epsilon)

    if "lengths" in targets[0]:
        # For multirc, there is no annotation for row, aks q+a
        assert len(p_r) == 0
        assert p == pc
        assert f == fc
        assert r == rc
        if rc == pc:
            # print(p_c)
            # print(r_c)
            print(rc)
            print(pc)
            print()
    # return precision, recall, f1, precision_c, recall_c, f1_score_c, p_all, r_all, f1_all, rationale_ratio
    return pr, rr, fr, pc, rc, fc, p, r, f, rationale_ratio


def rationale_sent_to_token(lengths, rationales):
    total_l = sum(lengths)
    r_tk = np.zeros(total_l)
    for i, s in enumerate(list(rationales)):
        if s != 0:
            r_tk[sum(lengths[:i]) : sum(lengths[: i + 1])] = 1
    return r_tk
