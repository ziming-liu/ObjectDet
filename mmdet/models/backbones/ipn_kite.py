import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
import mmcv
from mmdet.models.necks.self_attention import MultiHeadAttention
from mmdet.models.utils import ConvModule

from mmdet.ops import DeformConv, ModulatedDeformConv, ContextBlock
from mmdet.models.plugins import GeneralizedAttention

from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


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
        assert not with_cp

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
class IPN_kite(nn.Module):
    """ResNet backbone.

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
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_att=False,
                 without_ip=False,
                 without_dconv=False,
                 num_branch=4,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True):
        super(IPN_kite, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.without_ip = without_ip
        self.with_att = with_att
        self.without_dconv = without_dconv
        self.num_branch = num_branch
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
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        if with_att:
            #self.attention2048 = MultiHeadAttention(n_head=3,d_model=2048,d_v=2048,d_k=2048)
            self.attention1024 = MultiHeadAttention(n_head=3,d_model=1024,d_v=64,d_k=64)
            self.attention512 = MultiHeadAttention(n_head=3,d_model=512,d_v=64,d_k=64)
            self.attention256 = MultiHeadAttention(n_head=3,d_model=256,d_v=64,d_k=64)
            self.maxpooling = nn.MaxPool2d(kernel_size=6,stride=6,padding=0)

        self._make_stem_layer()
        self.channel_setting = [256, 512, 1024, 2048]
        self.dconvs = nn.ModuleList()
        for i in range(self.num_stages-1):
            #for j in range(self.num_stages - i - 1):
            dconv = ConvModule(
                self.channel_setting[i],
                self.channel_setting[i],
                3,
                stride=1,
                padding=2,
                dilation=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=None,
                inplace=False)
            self.dconvs.append(dconv)
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
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
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input):

        parimad = []

        b, c, h, w = input.shape
        #print("INPUT")

        #print(input.shape)
        edge = max(h,w)
        range_ = edge - edge/2
        #print("inter {}".format(range_/4))
        scale_factor = [(edge-range_*i/self.num_branch-1)/edge for i in range(self.num_branch)]
        #print(scale_factor)
        if self.without_ip:
            for i in range(self.num_stages):
                parimad.append(input.clone())
        else:
            for i in range(self.num_branch):
                parimad.append(F.interpolate(input.clone(),scale_factor=scale_factor[i],mode='nearest'))

        parimad_feats = []
        for i in range(self.num_branch):
            x = parimad[i]
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            parimad_feats.append(x)

        outs = [[] for _ in range(self.num_stages)]
        for lvl in range(self.num_stages):
            if lvl < self.num_stages - self.num_branch+1:
                for nodeidx in range(0, self.num_branch):

                    layer_name = self.res_layers[lvl]
                    res_layer = getattr(self, layer_name)
                    if lvl == 0:
                        outs[lvl].append(res_layer(parimad_feats[nodeidx]))
                    elif nodeidx==0:
                        outs[lvl].append(res_layer(outs[lvl-1][nodeidx]))
                    else:
                        left_Node = outs[lvl - 1][nodeidx]
                        if self.without_dconv:
                            down_Node = outs[lvl - 1][nodeidx-1]
                        else:
                            down_Node = self.dconvs[lvl - 1](outs[lvl - 1][nodeidx-1])
                        # 1 得到插值结果
                        scale_factor = left_Node.shape[2] / down_Node.shape[2]
                        next = F.interpolate(down_Node, scale_factor=scale_factor, )

                        if self.with_att:
                            # 2 得到pooling的小的全局特征
                            global_feat = self.maxpooling(down_Node)
                            # 3 根据全局特征 去 补充插值结果缺失的特征信息
                            b1, c1, h1, w1 = next.shape
                            # print(h1*w1)
                            next = next.reshape(b1, c1, -1).permute(0, 2, 1)
                            # 索引缩小 过大的特征图
                            threshold = 30
                            if h1 > threshold and w1 > threshold:
                                hidx = torch.LongTensor([i for i in range(0, h1, h1 // threshold)]).cuda()
                                widx = torch.LongTensor([i for i in range(0, w1, w1 // threshold)]).cuda()
                                idx = w1 * hidx.repeat(len(widx), 1).t().reshape(-1) + widx.repeat(1,
                                                                                                   len(hidx)).reshape(
                                    -1)
                                # print(idx)
                                sparse_Next = next[:, idx, :]
                                # print(len(idx))
                            else:
                                sparse_Next = next[:, :, :]

                            b2, c2, h2, w2 = global_feat.shape
                            global_feat = global_feat.reshape(b2, c2, -1).permute(0, 2, 1)
                            if lvl - 1 == 0:
                                sparse_Next = self.attention256(sparse_Next, global_feat, global_feat)
                            elif lvl - 1 == 1:
                                sparse_Next = self.attention512(sparse_Next, global_feat, global_feat)
                            elif lvl - 1 == 2:
                                sparse_Next = self.attention1024(sparse_Next, global_feat, global_feat)
                            # elif lvl-1==3:
                            #    next = self.attention2048(next, global_feat, global_feat)
                            if h1 > threshold and w1 > threshold:
                                next[:, idx, :] = sparse_Next
                            else:
                                next = sparse_Next
                            next = next.permute(0, 2, 1).reshape(b1, c1, h1, w1)

                        w = min(next.shape[3], left_Node.shape[3])
                        h = min(next.shape[2], left_Node.shape[2])
                        left_Node = left_Node[:, :, :h, :w]
                        left_Node = (left_Node + next[:, :, :h, :w]) / 2.0
                        # update left node
                        outs[lvl - 1][nodeidx] = left_Node
                        outs[lvl].append(res_layer(left_Node))

            else:
                for nodeidx in range(0,self.num_stages-lvl):

                    layer_name = self.res_layers[lvl]
                    res_layer = getattr(self, layer_name)
                    if lvl==0:
                        outs[lvl].append(res_layer(parimad_feats[nodeidx]))
                    else:
                        left_Node = outs[lvl-1][nodeidx+1]
                        if self.without_dconv:
                            down_Node = outs[lvl - 1][nodeidx]
                        else:
                            down_Node = self.dconvs[lvl-1](outs[lvl-1][nodeidx])
                        # 1 得到插值结果
                        scale_factor = left_Node.shape[2] / down_Node.shape[2]
                        next = F.interpolate(down_Node, scale_factor=scale_factor, )

                        if self.with_att:
                            # 2 得到pooling的小的全局特征
                            global_feat = self.maxpooling(down_Node)
                            # 3 根据全局特征 去 补充插值结果缺失的特征信息
                            b1, c1, h1, w1 = next.shape
                           # print(h1*w1)
                            next = next.reshape(b1, c1, -1).permute(0, 2, 1)
                            # 索引缩小 过大的特征图
                            threshold = 30
                            if h1>threshold and w1>threshold:
                                hidx = torch.LongTensor([i for i in range(0,h1,h1//threshold)]).cuda()
                                widx = torch.LongTensor([i for i in range(0, w1, w1 // threshold)]).cuda()
                                idx = w1 * hidx.repeat(len(widx),1).t().reshape(-1) + widx.repeat(1,len(hidx)).reshape(-1)
                                #print(idx)
                                sparse_Next = next[:, idx, :]
                                #print(len(idx))
                            else:
                                sparse_Next = next[:,:,:]

                            b2, c2, h2, w2 = global_feat.shape
                            global_feat = global_feat.reshape(b2, c2, -1).permute(0, 2, 1)
                            if lvl-1==0:
                                sparse_Next = self.attention256(sparse_Next,global_feat,global_feat)
                            elif lvl-1==1:
                                sparse_Next = self.attention512(sparse_Next, global_feat, global_feat)
                            elif lvl-1==2:
                                sparse_Next = self.attention1024(sparse_Next, global_feat, global_feat)
                            #elif lvl-1==3:
                            #    next = self.attention2048(next, global_feat, global_feat)
                            if h1 > threshold and w1 > threshold:
                                next[:,idx,:] = sparse_Next
                            else:
                                next = sparse_Next
                            next = next.permute(0,2,1).reshape(b1,c1,h1,w1)

                        w = min(next.shape[3], left_Node.shape[3])
                        h = min(next.shape[2], left_Node.shape[2])
                        left_Node = left_Node[:, :, :h, :w]
                        left_Node = (left_Node + next[:, :, :h, :w]) / 2.0
                        # update left node
                        outs[lvl - 1][nodeidx + 1] = left_Node
                        outs[lvl].append(res_layer(left_Node))

            #col_outs =[]
        #row_outs = []
        #for lvl in range(len(outs)):
        #    for bran in range(len(outs[lvl])):
        #        col_outs.append(outs[lvl][0])
        #        row_outs.append(outs[lvl][-1])
        finalouts = []
        for lvl in range(self.num_stages):
            if lvl not in self.out_indices:
                continue
            finalouts.extend(outs[lvl])
            #n_Node = self.num_stages - lvl
            #for nodeidx in range(n_Node):
            #    finalouts.append(outs[lvl][nodeidx])
        return tuple(finalouts)

    def train(self, mode=True):
        super(IPN_kite, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

