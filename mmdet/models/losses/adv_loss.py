import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..registry import LOSSES


@LOSSES.register_module
class AdversarialLoss(nn.Module):

    def __init__(self,
                 avg_factor=None,
                 reduction='mean',
                 loss_weight=torch.FloatTensor([1])):
        super(AdversarialLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=loss_weight,
                                                    size_average=avg_factor)


    def forward(self,
                cls_score,
                label,
                weight=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            **kwargs)
        return loss_cls