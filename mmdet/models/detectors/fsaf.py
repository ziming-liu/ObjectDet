from .single_stage import SingleStageDetector
from ..registry import DETECTORS
import torch.nn as nn
import torch
import numpy as np
from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result, delta2bbox, multiclass_nms


@DETECTORS.register_module
class FSAFNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FSAFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)


