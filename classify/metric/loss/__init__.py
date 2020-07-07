# from classify.metric.loss.hinge import HingeLoss
from classify.metric.loss.ce import CrossEntropyLoss
from classify.metric.loss.bce import BinaryCrossEntropyLoss

# from classify.metric.loss.multimarginloss import MultiMarginLoss
from classify.metric.loss.f1loss import F1Loss

__all__ = [
    # "HingeLoss",
    "CrossEntropyLoss",
    "BinaryCrossEntropyLoss",
    # "MultiMarginLoss",
    "F1Loss",
]
