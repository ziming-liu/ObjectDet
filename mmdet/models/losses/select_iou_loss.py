import torch
import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..registry import LOSSES
from .utils import weighted_loss



def select_iou_loss(pred, target, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = pred.size(0)
    assert pred.size(0) == target.size(0)
    target = target.clamp(min=0.)
    area_pred = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    area_gt = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
    area_i = ((torch.min(pred[:, 0], target[:, 0]) +
               torch.min(pred[:, 2], target[:, 2])) *
              (torch.min(pred[:, 1], target[:, 1]) +
               torch.min(pred[:, 3], target[:, 3])))
    area_u = area_pred + area_gt - area_i
    iou = area_i / area_u
    loc_losses = -torch.log(iou.clamp(min=1e-7))
    return torch.sum(weight * loc_losses) / avg_factor
@LOSSES.register_module
class Select_Iou_Loss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(Select_Iou_Loss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        #if weight is not None and not torch.any(weight > 0):
        #    return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * select_iou_loss(
            pred,
            target,
            weight,
            avg_factor
           )
        return loss