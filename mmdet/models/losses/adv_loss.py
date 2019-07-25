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
                 loss_weight=torch.FloatTensor([1]).cuda()):
        super(AdversarialLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        #self.criterion = torch.nn.BCEWithLogitsLoss(weight=loss_weight,
        #                                            size_average=avg_factor)
    def forward(self,
                score,
                label,
                weight=None,
                ):
        #assert reduction_override in (None, 'none', 'mean', 'sum')
        #reduction = (
        #    reduction_override if reduction_override else self.reduction)

        loss_adv = F.binary_cross_entropy_with_logits(score, label,
                                           self.loss_weight,
                                           pos_weight=weight,
                                           reduction=self.reduction)
        return loss_adv