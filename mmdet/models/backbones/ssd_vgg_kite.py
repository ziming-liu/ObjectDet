import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import VGG, constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES


@BACKBONES.register_module
class SSDVGGkite(VGG):
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 input_size,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 l2_norm_scale=20.):
        super(SSDVGGkite, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices)
        assert input_size in (300, 512)
        self.input_size = input_size

        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.out_feature_indices = out_feature_indices

        self.inplanes = 1024
        self.extra = self._make_extra_layers(self.extra_setting[input_size])
        self.l2_norm = L2Norm(
            self.features[out_feature_indices[0] - 1].out_channels,
            l2_norm_scale)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        constant_init(self.l2_norm, self.l2_norm.scale)

    def forward(self, input):
        edge_len = 7
        parimad = []
        b, c, h, w = input.shape
        map_edge = max(h, w)
        range_ = map_edge - map_edge / (edge_len-1)
        scale_factor = []
        for i in range(edge_len):
            scale_factor.append((map_edge-range_*i/(edge_len-1))/map_edge)
        print(scale_factor)
        for i in range(edge_len):
            parimad.append(F.interpolate(input.clone(), scale_factor=scale_factor[i], mode='nearest'))

        outs = [[] for _ in range(7)]
        for lvl in range(len(self.out_feature_indices)):
            for nodeidx in range(edge_len-lvl):
                n_layers = self.out_feature_indices[lvl]
                if lvl ==0:
                    x = parimad[lvl]
                    for layeridx in range(self.out_feature_indices[0]+1):
                        x=self.features[layeridx](x)
                    outs[lvl].append(x)
                else:
                    left_Node = outs[lvl-1][nodeidx+1]
                    down_Node = outs[lvl-1][nodeidx]
                    scale_factor = left_Node.shape[2] / down_Node.shape[2]
                    next = F.interpolate(down_Node, scale_factor=scale_factor, )
                    w = min(next.shape[3], left_Node.shape[3])
                    h = min(next.shape[2], left_Node.shape[2])
                    left_Node = left_Node[:, :, :h, :w]
                    left_Node = (left_Node + next[:, :, :h, :w]) / 2.0
                    for layeridx in range(self.out_feature_indices[lvl-1]+1,self.out_feature_indices[lvl]+1):
                        left_Node = self.features[layeridx](left_Node)
                    out = left_Node
                    outs[lvl].append(out)
        for lvl in range(len(self.out_feature_indices),edge_len):
            for nodeidx in range(edge_len-lvl):
                left_Node = outs[lvl - 1][nodeidx + 1]
                down_Node = outs[lvl - 1][nodeidx]
                scale_factor = left_Node.shape[2] / down_Node.shape[2]
                next = F.interpolate(down_Node, scale_factor=scale_factor, )
                w = min(next.shape[3], left_Node.shape[3])
                h = min(next.shape[2], left_Node.shape[2])
                left_Node = left_Node[:, :, :h, :w]
                left_Node = (left_Node + next[:, :, :h, :w]) / 2.0
                #for layeridx in range(self.out_feature_indices[lvl - 1] + 1, self.out_feature_indices[lvl] + 1):
                # 每两层输出一个结果
                left_Node = F.relu(self.extra[2*(lvl-len(self.out_feature_indices))](left_Node), inplace=True)
                out = F.relu(self.extra[1+2*(lvl-len(self.out_feature_indices))](left_Node), inplace=True)
                outs[lvl].append(out)
        for i in range(len(outs[0])):
            outs[0][i] = self.l2_norm(outs[0][i])
        finalouts= []
        for i in range(len(outs)):
            finalouts.extend(outs[i])

        if len(finalouts) == 1:
            return finalouts[0]
        else:
            return tuple(finalouts)

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return nn.Sequential(*layers)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
