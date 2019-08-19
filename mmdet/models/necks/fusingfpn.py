import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class fusingFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 interval=2,
                 num_stage=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(fusingFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.num_stage = num_stage
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.interval = interval
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.out_strides = []
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        reshaped_laterals = [[] for _ in range(len(laterals))]
        idx = 0
        for i in range(self.num_stage):
            num = self.num_stage - i
            for j in range(i, i + num):
                reshaped_laterals[j].append(laterals[idx])
                idx += 1
        for idx_bran in range(0,self.num_stage,1): # 4 3 2 branch
            laterals = reshaped_laterals[idx_bran]
            # for branch idx bran
            # build top-down path
            # brantch自己内部相加
            used_backbone_levels = len(laterals)
            for i in range(1, used_backbone_levels , 1):

                scale_factor = laterals[i].shape[3] / laterals[i-1].shape[3]
                next = F.interpolate(
                    laterals[i-1], scale_factor=scale_factor, mode='nearest')
                w = min(next.shape[3],laterals[i].shape[3])
                h = min(next.shape[2],laterals[i].shape[2])
                next = next[:,:,:h,:w]
                laterals[i]= laterals[i][:,:,:h,:w]
                laterals[i] = (laterals[i] + next) / 2
            # branch之间相加
            if idx_bran!=self.num_stage-1:
                for ii in range(len(reshaped_laterals[idx_bran])):
                    smaller = reshaped_laterals[idx_bran+1][ii] # 对应stage的下一层
                    biger = reshaped_laterals[idx_bran][ii]
                    scale_factor = smaller.shape[3] / biger.shape[3]
                    next = F.interpolate(
                        biger, scale_factor=scale_factor, mode='nearest')
                    w = min(next.shape[3], smaller.shape[3])
                    h = min(next.shape[2], smaller.shape[2])
                    next = next[:, :, :h, :w]
                    smaller = smaller[:, :, :h, :w]
                    reshaped_laterals[idx_bran+1][ii] = (smaller + next)/2

        path_outs = []
        for i in range(self.num_stage): # col
            num = self.num_stage - i
            for j in range(i, i + num): # row
                path_outs.append(reshaped_laterals[j][i])
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](path_outs[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
