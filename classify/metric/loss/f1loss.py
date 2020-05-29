from typing import Dict, List, Tuple

import torch

from classify.metric.abstract import Metric
import torch.nn.functional as F


class F1Loss(Metric):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, 
                 epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()


    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the Negative log likelihood loss for multilabel.

        Parameters
        ----------
        pred: torch.Tensor
            input logits of shape (B x N)
        target: torch.LontTensor
            target tensor of shape (B x N)

        Returns
        -------
        loss: torch.float
            Multi label negative log likelihood loss, of shape (B)

        """
        # TODO: Need to compute this logits
        # pred = torch.Tensor(pred).cuda()
        # target = torch.Tensor(target)
        targets = [t for target in targets for t in target['targets']]
        targets = torch.stack(targets)
        loss = self.forward(logits, targets)
        
        return loss
        
