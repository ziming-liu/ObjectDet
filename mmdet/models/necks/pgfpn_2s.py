import os
import random

import cv2
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class PGFPN2s(nn.Module):

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
                 keep=0,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(PGFPN2s, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.keep = keep
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.interval = interval
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            #assert num_outs >= self.num_ins - start_level
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
        self.PGFPN2s_convs = nn.ModuleList()

        for i in range(self.start_level, len(in_channels)):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)

        for i in range(6):
            PGFPN2s_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.PGFPN2s_convs.append(PGFPN2s_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - 6
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[-1]
                else:
                    in_channels = out_channels
                extra_PGFPN2s_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.PGFPN2s_convs.append(extra_PGFPN2s_conv)

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
        large_path = list()
        #mid_path  =list()
        small_path = list()
        large_path.extend(laterals[:4])
        #mid_path.extend(laterals[4:8])
        small_path.extend(laterals[4:8])
        """ 
        randstr = ''
        for zz in range(5):
            randstr = randstr + str(random.randint(0, 9))
        path = "/home/share2/ziming/2sfpnv2/"
        if not os.path.exists(path):
            os.makedirs(path)
        for mm in range(len(large_path)):
            tem = large_path[mm].detach()[0].mean(0)
            h, w = tem.shape
            tem = tem.unsqueeze(2).repeat(1, 1, 1).cpu().numpy()
            import numpy as np
            tem = tem - np.min(tem)
            tem = tem / np.max(tem)
            tem = np.uint8(255 * tem)
            #print(tem.shape)
            heatmap = cv2.applyColorMap(cv2.resize(tem, (w, h)), cv2.COLORMAP_JET)
            cv2.imwrite(path + randstr + 'shallow' + '_' + str(mm) + '.jpg', heatmap)

        for mm in range(len(small_path)):
            tem = small_path[mm].detach()[0].mean(0)
            h, w = tem.shape
            tem = tem.unsqueeze(2).repeat(1, 1, 1).cpu().numpy()
            import numpy as np
            tem = tem - np.min(tem)
            tem = tem / np.max(tem)
            tem = np.uint8(255 * tem)
            #print(tem.shape)
            heatmap = cv2.applyColorMap(cv2.resize(tem, (w, h)), cv2.COLORMAP_JET)
            cv2.imwrite(path + randstr + 'deep' + '_' + str(mm) + '.jpg', heatmap)
        """
        # build top-down path
        for i in range(len(small_path) - 1, 0, -1):
            small_path[i - 1] = (small_path[i - 1] + F.interpolate(
                small_path[i], scale_factor=self.interval, mode='nearest')) /1

        large_path[-1] = (large_path[-1] + F.interpolate(
                small_path[-1], scale_factor=self.interval, mode='nearest'))/1
        for i in range(len(large_path) - 1, 0, -1):
                large_path[i - 1] = (large_path[i - 1] + F.interpolate(
                    large_path[i]+F.interpolate(small_path[i], scale_factor=self.interval, mode='nearest')
                    , scale_factor=self.interval, mode='nearest') )/1

        laterals=large_path


        #laterals.append(mid_path[-1])
        #laterals.append(small_path[-1])
        used_backbone_levels = len(laterals)
        # build outputs
        # part 1: from original levels
        #print(used_backbone_levels)
        outs = [
            self.PGFPN2s_convs[i](laterals[i]) for i in range(used_backbone_levels)
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
                    orig = inputs[-1]
                    outs.append(self.PGFPN2s_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.PGFPN2s_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.PGFPN2s_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.PGFPN2s_convs[i](outs[-1]))
        """ 
        for mm in range(len(outs)):
            tem = outs[mm].detach()[0].mean(0)
            h, w = tem.shape
            tem = tem.unsqueeze(2).repeat(1, 1, 1).cpu().numpy()
            import numpy as np
            tem = tem - np.min(tem)
            tem = tem / np.max(tem)
            tem = np.uint8(255 * tem)
            # print(tem.shape)
            heatmap = cv2.applyColorMap(cv2.resize(tem, (w, h)), cv2.COLORMAP_JET)
            cv2.imwrite(path + randstr + 'fusing' + '_' + str(mm) + '.jpg', heatmap)
        """
        return tuple(outs)
