import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class FPNalign(nn.Module):

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
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPNalign, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
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
    def split(self,feat,num_grid=4):
        assert num_grid % 4 ==0
        b,c,h,w = feat.shape
        #print("split")
        def single(unit):
            b, c, h, w = unit.shape
            halfH, halfW = h // 2, w // 2
            tl = unit[:, :, :halfH, :halfW]
            tr = unit[:, :, :halfH, halfW:w]
            dl = unit[:, :, halfH:h, :halfW]
            dr = unit[:, :, halfH:h, halfW:w]
            #print(tl.shape)
           # print(tr.shape)
          #  print(dl.shape)
          #  print(dr.shape)
            return torch.cat((tl,tr,dl,dr),0)

        while(num_grid!=1):
            feat = single(feat)
            num_grid = num_grid / 4

        return feat

    def gridcat(self,batch_feat,num_grid=4):
        times = num_grid //4
        #print("cat ")
        def single(patches):
            up_ = torch.cat((patches[0].unsqueeze(0), patches[1].unsqueeze(0)), 3)
            #(up_.shape)
            down_ = torch.cat((patches[2].unsqueeze(0), patches[3].unsqueeze(0)), 3)
            #print(down_.shape)
            cat_ = torch.cat((up_, down_), 2)
            return cat_
        while(num_grid!=1):
            b,c,h,w = batch_feat.shape
            lower_grid = []
            for i in range(b//4):
                lower_grid.append(single(batch_feat[i:i+4]))
            batch_feat = torch.cat(tuple(lower_grid),0)
            #print(batch_feat[0].shape)
            num_grid = num_grid // 4
        return batch_feat
    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            x = lateral_conv(inputs[i + self.start_level])
            splitx = self.split(inputs[i + self.start_level], num_grid=4)
            splitx = lateral_conv(splitx)
            catx = self.gridcat(splitx)
            x += catx
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            #print(laterals[i].shape)
           # print(F.interpolate(
            #    laterals[i], scale_factor=1.36, mode='nearest').shape)
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=self.interval, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = []
        for i in range(len(laterals)):
            x = self.fpn_convs[i](laterals[i])
            splitx = self.split(laterals[i], num_grid=4)
            splitx = self.fpn_convs[i](splitx)
            catx = self.gridcat(splitx)
            x += catx
            outs.append(x)
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
