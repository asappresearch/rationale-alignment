from typing import List, Tuple

from similarity.metric import Metric
from similarity.metric.dev import *
from similarity.metric.loss import *
from similarity.metric.train import *
from rationale_alignment.parsing import Arguments


def load_loss_and_metrics(
    args: Arguments,
) -> Tuple[Metric, Metric, List[Metric], List[Metric]]:
    """
    Defines the loss and metric functions that will be used during AskUbuntu training.

    :param args: Arguments.
    :return: A tuple consisting of:
    1) Training loss function
    2) Dev metric function
    3) A list of additional training metrics
    4) A list of additional validation metrics
    """
    # Loss
    loss_fn = HingeLoss(
        margin=args.margin, pooling=args.hinge_pooling, alpha=args.hinge_alpha
    )

    # Metrics
    metric_fn = AUC()
    extra_training_metrics = []

    if args.alignment != "average":
        extra_training_metrics += [
            CostRange(),
            CostMin(),
            CostMax(),
            CostMean(),
            CostMedian(),
            AlignmentCount(),
            AlignmentCount(normalize="min"),
            AlignmentCount(normalize="full"),
            AlignmentSum(),
            AlignmentRowMarginalError(),
            AlignmentColumnMarginalError(),
            AlignmentCost(similar=True),
            AlignmentCost(similar=False),
            AlignmentEntropy(similar=True),
            AlignmentEntropy(similar=False),
        ]

    if args.cost_fn in ["dot_product", "scaled_dot_product", "cosine_similarity"]:
        extra_training_metrics += [
            CostSign(positive=True, similar=True),
            CostSign(positive=True, similar=False),
            CostSign(positive=False, similar=True),
            CostSign(positive=False, similar=False),
        ]

    extra_validation_metrics = [
        AUC(max_fpr=0.1),
        AUC(max_fpr=0.05),
        MAP(),
        MRR(),
        Precision(n=1),
        Precision(n=5),
    ]

    if args.alignment != "average":
        extra_validation_metrics += [
            AlignmentCount(),
            AlignmentCount(normalize="min"),
            AlignmentCount(normalize="full"),
        ]

    return loss_fn, metric_fn, extra_training_metrics, extra_validation_metrics
