import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
from .local_attention import LocalSelfAttention,MultiHeadAttention
@NECKS.register_module
class FAM(nn.Module):

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
                 num_stage = 4,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FAM, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.interval = interval
        self.num_stage = num_stage
        self.horizontal_upers  = nn.ModuleList(nn.ModuleList() for _ in range(self.num_stage))
        #for stream in range(1,self.num_stage):
         #   for numnode in range(self.num_stage-stream):
        #        local_attention = LocalSelfAttention(heads=3, d_model=256, dv=256, dk=256,
        #                                                  neighbors=5, rate=1)
       #         self.horizontal_upers[stream].append(local_attention)
        #self.vertical_upers =  nn.ModuleList(nn.ModuleList() for _ in range(self.num_stage))
        #for stream in range(0,self.num_stage):
       #     for numnode in range(self.num_stage-stream-1):
       #         local_attention = LocalSelfAttention(heads=3, d_model=256, dv=256, dk=256,
       #                                                   neighbors=5, rate=1)
        #        self.vertical_upers[stream].append(local_attention)
        self.attention = LocalSelfAttention(heads=1, d_model=256, dv=32, dk=32,
                                                          neighbors=4, rate=1)
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

        self.laterals_convs = nn.ModuleList()
        self.fpns_convs = nn.ModuleList()

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

            self.laterals_convs.append(l_conv)
            self.fpns_convs.append(fpn_conv)

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
                self.fpns_convs.append(extra_fpn_conv)

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
            for i, lateral_conv in enumerate(self.laterals_convs)
        ]
        # turn to level list
        # turn to level list
        start_Idx = 0
        level_list = [[] for _ in range(self.num_stage)]
        for ii in range(self.num_stage):
            length = self.num_stage - ii
            level_list[ii] = laterals[start_Idx:start_Idx + length]
            start_Idx += length

        # build top-down path
        used_backbone_levels = len(level_list)
        for i in range(used_backbone_levels - 1, -1, -1):  # lvl
            num_add = len(level_list[i])  # == this level's nodes
            for node_I in range(num_add):
                # horizon
                idx = num_add - node_I - 1
                if i!=0:
                    #uper_layer = self.horizontal_upers[i][node_I]
                    uper_layer= self.attention
                    b,c,h,w = level_list[i-1][idx+1].shape
                    big= level_list[i - 1][idx+1].reshape(b,c,-1).contiguous().permute(0,2,1)
                    small = level_list[i][idx].reshape(b,c,-1).contiguous().permute(0,2,1)
                    uper = uper_layer(big,small,small
                                      ).permute(0,2,1).reshape(b,c,h,w)
                    #print(level_list[i][idx].shape)
                    #print(level_list[i - 1][idx].shape)
                    #uper = F.interpolate(level_list[i][idx],scale_factor=2)
                    level_list[i - 1][idx+1]  = (level_list[i-1][idx+1] + uper)/2.0
                # vertical
                idx = num_add - node_I - 1
                if idx!=0:
                    #uper_layer = self.vertical_upers[i][node_I]
                    uper_layer = self.attention
                    idx = num_add - node_I - 1
                    b,c,h,w = level_list[i][idx - 1].shape
                    big = level_list[i][idx - 1].reshape(b,c,-1).contiguous().permute(0,2,1)
                    small = level_list[i][idx].reshape(b,c,-1).contiguous().permute(0,2,1)
                    uper = uper_layer(big,small,small
                                      ).permute(0,2,1).reshape(b,c,h,w)
                    #uper = F.interpolate(level_list[i][idx], scale_factor=2)
                    level_list[i][idx - 1] = (level_list[i][idx - 1] + uper) / 2.0
        return level_list[0]
        """ 
        # turn to small to big list
        laterals = []
        for ii in range(len(level_list)):
            laterals.extend(level_list[ii])
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpns_convs[i](laterals[i]) for i in range(len(laterals))
        ]
        used_backbone_levels = len(laterals)
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
                    outs.append(self.fpns_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpns_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpns_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpns_convs[i](outs[-1]))
        #print(len(outs))
        #print(outs[-1].shape)
        return tuple(outs)
        """
