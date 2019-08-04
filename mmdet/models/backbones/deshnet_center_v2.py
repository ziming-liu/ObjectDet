import logging
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.runner.checkpoint import open_mmlab_model_urls, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm

import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module

import mmcv
import torch
import torchvision
from torch.utils import model_zoo
from mmcv.runner import get_dist_info

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.utils import model_zoo

from mmdet.ops import DeformConv, ModulatedDeformConv, ContextBlock
from mmdet.models.plugins import GeneralizedAttention

from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
model_urls = {
    's1': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    's2': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    's3': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    's6': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        #assert not with_cp
        self.with_cp = with_cp
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class DSNetcenter(nn.Module):
    """deep shallow backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        's1': (BasicBlock, (2, 2, 2, 2)),
        's2': (BasicBlock, (3, 4, 6, 3)),
        's3': (Bottleneck, (3, 4, 6, 3)),
        's4': (Bottleneck, (3, 4, 23, 3)),
        's5': (Bottleneck, (3, 8, 36, 3)),
        's6': (Bottleneck, (2, 2, 2, 2))
    }
    channel_setting = {
        's1': (64,128,256,512),
        's2': (64,128,256,512),
        's3': (256,512,1024,2048),
        's4': (256,512,1024,2048),
        's5': (256,512,1024,2048),
        's6': (256, 512, 1024, 2048),
    }

    def __init__(self,
                 depth=('s1','s3'),
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        super(DSNetcenter, self).__init__()
        for o in range(len(depth)):
            if depth[o] not in self.arch_settings:
                raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        self.zero_init_residual = zero_init_residual
        self.num_stream = len(depth)
        self.blocks = []
        self.stage_blocks = []

        self.streams = []
        for ii in range(self.num_stream):
            stream = []
            block, stage_blocks = self.arch_settings[depth[ii]]
            self.blocks.append(block)
            self.stage_blocks.append(stage_blocks[:num_stages])
            self.inplanes = 64

            pre_block_name = self._make_stem_layer(prefix=depth[ii])
            stream.append(pre_block_name)

            self.res_layers = []
            for i, num_blocks in enumerate(self.stage_blocks[ii]):
                stride = strides[i]
                dilation = dilations[i]
                dcn = self.dcn if self.stage_with_dcn[i] else None
                gcb = self.gcb if self.stage_with_gcb[i] else None
                planes = 64 * 2**i
                res_layer = make_res_layer(
                    self.blocks[ii],
                    self.inplanes,
                    planes,
                    num_blocks,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    gcb=gcb,
                    gen_attention=gen_attention,
                    gen_attention_blocks=stage_with_gen_attention[i])
                self.inplanes = planes * self.blocks[ii].expansion
                layer_name = str(self.depth[ii]) +  '_layer{}'.format(i + 1)
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)
            stream.extend(self.res_layers)
            self.streams.append(stream)
        self._freeze_stages()

        # interaction

        self.fusing_layers_de = []
        self.fusing_layers_sh = []

        fusing_layer = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                64,
                64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            build_norm_layer(self.norm_cfg, 64)[1]
        )
        layer_name = 'de_fusing_layer{}'.format(0)
        self.add_module(layer_name, fusing_layer)
        self.fusing_layers_de.append(layer_name)

        fusing_layer = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                64,
                64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            build_norm_layer(self.norm_cfg, 64)[1]
        )
        layer_name = 'sh_fusing_layer{}'.format(0)
        self.add_module(layer_name, fusing_layer)
        self.fusing_layers_sh.append(layer_name)

        for l in range(self.num_stages):
            fusing_layer = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.channel_setting[self.depth[0]][l],
                    self.channel_setting[self.depth[1]][l],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                build_norm_layer(self.norm_cfg, self.channel_setting[self.depth[1]][l])[1]
                )
            layer_name = 'de_fusing_layer{}'.format(l + 1)
            self.add_module(layer_name, fusing_layer)
            self.fusing_layers_de.append(layer_name)


        self.out_layers = []
        for ii in range(self.num_stages):
            out_layer = nn.Sequential(build_conv_layer(self.conv_cfg,
                                         self.channel_setting[self.depth[-1]][ii],
                                         256,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         ),
                                      build_norm_layer(self.norm_cfg, 256)[1]
                                      )
            layer_name = self.depth[-1]+'_out_layer{}'.format(ii + 1)
            self.add_module(layer_name, out_layer)
            self.out_layers.append(layer_name)
        for ii in range(self.num_stream-2,-1,-1):
            out_layer = nn.Sequential(build_conv_layer(self.conv_cfg,
                                                       self.channel_setting[self.depth[ii]][-1],
                                                       256,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0,
                                                       ),
                                      build_norm_layer(self.norm_cfg, 256)[1]
                                      )
            layer_name = self.depth[ii] + '_out_layer{}'.format(0)
            self.add_module(layer_name, out_layer)
            self.out_layers.append(layer_name)




    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self,prefix='s1'):
        conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.add_module(prefix+'_conv1',conv1)
        norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(prefix+'_'+norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #pre = nn.Sequential(conv1,norm1,relu,maxpool)
        #pre_block_name = 'pre_'+postfix
        #self.add_module(pre_block_name,pre)
        return [prefix+'_conv1',prefix+'_'+norm1_name]

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for jj in range(len(self.streams)):
                conv = getattr(self,self.streams[jj][0][0])
                norm = getattr(self,self.streams[jj][0][1])
                norm.eval()
                for m in [conv,norm]:
                    for param in m.parameters():
                        param.requires_grad = False
        for jj in range(len(self.depth)):
            for i in range(1, self.frozen_stages + 1):
                m = getattr(self, self.depth[jj]+'_layer{}'.format(i))
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        if pretrained is None:
            print("no pretrain params")
        for ii in range(len(self.depth)):
            stream_name = self.depth[ii]
            url = model_urls[self.depth[ii]]
            #http = {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth'}
            pretrained_dict = model_zoo.load_url(url)
            model_dict = self.state_dict()
            pretrained_dict = {stream_name + '_'+ k: v for k, v in pretrained_dict.items() if stream_name + '_'+ k in model_dict}  # filter out unnecessary keys
            #print("for {} stream".format(stream_name))
            #print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("load param for {}".format(stream_name))

    def forward(self, input):
        # input is same as the backbone feature output scale
        num_s = len(self.depth)
        n_inputs = []
        n_inputs.append(input)# big to small
        for _ in range(num_s-1):
            n_inputs.append(F.interpolate(n_inputs[-1],scale_factor=0.5))
        outs = []
        tem_outs = [[] for _ in range(num_s)]
        #for lv in range(num_s):
        x = n_inputs[1]
        #print(x.shape)
        conv = getattr(self, self.streams[0][0][0])
        norm = getattr(self,self.streams[0][0][1])
        #print(pre)
        x = conv(x)
        x = norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        xde = x

        #print("--------")
        x = n_inputs[0]
        # print(x.shape)
        conv = getattr(self, self.streams[1][0][0])
        norm = getattr(self, self.streams[1][0][1])
        # print(pre)
        x = conv(x)
        x = norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        xsh = x

        de_outs = []
        for i in range(1,len(self.streams[0])):
            layer_name = self.streams[0][i]
            res_layer = getattr(self, layer_name)
            xde = res_layer(xde)
            #if i in self.out_indices:
            de_outs.append(xde)

        sh_outs = []
        for i in range(1, len(self.streams[1])):
            layer_name = self.streams[1][i]
            res_layer = getattr(self, layer_name)
            xsh = res_layer(xsh)
            #if i in self.out_indices:
            sh_outs.append(xsh)

        outs = []
        for jj in range(len(sh_outs)-1,-1,-1):
            some_level = sh_outs[jj] + F.interpolate(de_outs[jj], scale_factor=2)
            out_layer = getattr(self, self.out_layers[jj])
            some_level = out_layer(some_level)
            if jj== len(sh_outs)-1:
                outs.append(some_level)
            else:
                outs.append(F.interpolate(outs[-1],scale_factor=2)+some_level)
        return tuple(outs)

    def train(self, mode=True):
        super(DSNetcenter, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
