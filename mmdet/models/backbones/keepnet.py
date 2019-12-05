import logging

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
from .resnet import BasicBlock, Bottleneck


class HRModule(nn.Module):
    """ High-Resolution Module for HRNet. In this module, every branch
    has 4 BasicBlocks/Bottlenecks. Fusion/Exchange is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(HRModule, self).__init__()
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(in_channels))
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, num_channels[branch_index] *
                                 block.expansion)[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)
    def _cat(self,patches):
        assert isinstance(patches,list)
        print(patches[0].shape)
        up_ = torch.cat((patches[0],patches[1]),3)
        print(up_.shape)
        down_ = torch.cat((patches[2],patches[3]),3)
        print(down_.shape)
        cat_ = torch.cat((up_,down_),2)
        print("cat shape{}".format(cat_.shape))
        return cat_

    def _split(self,unit):
        assert isinstance(unit,torch.Tensor)
        b,c,h,w = unit.shape
        halfH,halfW = h//2,w//2
        tl = unit[:,:,:halfH,:halfW]
        tr = unit[:,:,:halfH,halfW:w]
        dl = unit[:,:,halfH:h,:halfW]
        dr = unit[:,:,halfH:h,halfW:w]
        return tuple([tl,tr,dl,dr])
    def _split_N(self,unit,n):
        for i in range(n):
            boxes = []
            for x in unit:
                boxes.append(self._split(x))
            unit = boxes
        return unit
    cat_all = []
    def _cat_N(self,patches):

        if len(patches)==4:
            return self._cat(patches)
        else:
            tl= self._cat_N(patches[:(1//4)*len(patches)])
            tr=self._cat_N(patches[(1//4)* len(patches):(2//4)*len(patches)])
            dl=self._cat_N(patches[(2//4)* len(patches):(3 // 4) * len(patches)])
            dr=self._cat_N(patches[(3//4)* len(patches):(4 // 4) * len(patches)])
        return tuple([tl,tr,dl,dr])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            #for j in range(i-1,i+2,1):
            for j in range(num_branches):

                if j<0 or j>=num_branches or j<i-1 or j>=i+2:# no existing branch
                    fuse_layer.append(None)
                    continue
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],))
                            #nn.Upsample(
                            #    scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):# x is list of num branches
        if self.num_branches == 1:
            return [self.branches[0](x[0][0])]
        #print(type(x))
        #print(len(x))
        for i in range(self.num_branches):
            for j in range(len(x[i])):
                #print(type(x[i][j]))
                #print(x[i][j].shape)
                x[i][j] = self.branches[i](x[i][j])
        """ 
        x_fuse = []
        for i in range(len(x)):# fuse layer contain i branches and
            y = 0
            for j in range(self.num_branches):# each i contain j branches
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
        """
        x_fuse = []
        for i in range(len(x)):  #for braches, fuse layer contain i branches and
            y = [0 for _ in range(len(x[i]))] # for brachi , we have how many lines
            for j in range(i-1,i+2):  # each i contain j branches
                if j<0 or j>=len(x):
                    continue
                if i == j :
                    y = [y[k]+x[j][k] for k in range(len(x[i]))]
                    #y += x[j]
                elif i>j:# jsize >isize
                    processed_j =[]
                    splited = []
                    for k in range(len(x[j])):
                        processed_j.append(self.fuse_layers[i][j](x[j][k]))# j-->i
                        splited.extend(self._split(processed_j[k]))
                    assert len(splited) == len(x[i])
                    y = [y[k] + splited[k] for k in range(len(x[i]))]
                else: # j>i
                    processed_j = []
                    cated = []
                    for k in range(len(x[j])):
                        processed_j.append(self.fuse_layers[i][j](x[j][k]))  # j-->i
                    print(type(processed_j))
                    print(len(processed_j))
                    print(processed_j[0].shape)
                    for k in range(0,len(processed_j),4):
                        print("计数器")
                        cated.append(self._cat(processed_j[k:k+4]))
                    print("changdu")
                    print(len(x[i]))
                    print(len(cated))
                    print(cated[0].shape)
                    assert len(cated) == len(x[i])
                    y = [y[k] + cated[k] for k in range(len(x[i]))]

            x_fuse.append([self.relu(y[k]) for k in range(len(y))])
        return x_fuse


@BACKBONES.register_module
class KEEPNet(nn.Module):
    """HRNet backbone.

    High-Resolution Representations for Labeling Pixels and Regions
    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=False):
        super(KEEPNet, self).__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*hr_modules), in_channels

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

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def _cat(self, patches):
        assert isinstance(patches, list)
        up_ = torch.cat((patches[0], patches[1]), 3)
        down_ = torch.cat((patches[2], patches[3]), 3)
        cat_ = torch.cat((up_, down_), 2)
        print("cat shape{}".format(cat_.shape))

    def _split(self, unit):
        assert isinstance(unit, torch.Tensor)
        b, c, h, w = unit.shape
        halfH, halfW = h // 2, w // 2
        tl = unit[:, :, :halfH, :halfW]
        tr = unit[:, :, :halfH, halfW:w]
        dl = unit[:, :, halfH:h, :halfW]
        dr = unit[:, :, halfH:h, halfW:w]
        return tuple([tl, tr, dl, dr])
    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = [[x]]
        #x_list = [[] for _ in range(self.stage2_cfg['num_branches'])]
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            #for k in range(len(x)): # how many lines
            if self.transition1[i] is not None:
                tmp = []
                for k in range(len(x[-1])):
                    tmp.append(self.transition1[i](x[-1][k]))
                    # next branch
                x_list.append(tmp)
            else:
                x_list.append(x[i])
        branch_last = []
        for k in range(len(x_list[-1])):
            branch_last.extend(self._split(x_list[-1][k]))
        x_list[-1] = branch_last
        #print(x_list)
        #print("stage1 out ")
        #print(len(x_list[0]))
        #print(len(x_list[1]))
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                #x_list.append(self.transition2[i](y_list[-1]))
                tmp = []
                for k in range(len(y_list[-1])):
                    tmp.append(self.transition2[i](y_list[-1][k]))
                    # next branch

                x_list.append(tmp)
            else:
                x_list.append(y_list[i])
        branch_last = []
        for k in range(len(x_list[-1])):
            branch_last.extend(self._split(x_list[-1][k]))
        x_list[-1] = branch_last
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                tmp = []

                for k in range(len(y_list[-1])):
                    tmp.append(self.transition3[i](y_list[-1][k]))

                x_list.append(tmp)
            else:
                x_list.append(y_list[i])
        branch_last = []
        for k in range(len(x_list[-1])):
            branch_last.extend(self._split(x_list[-1][k]))
        x_list[-1] = branch_last
        y_list = self.stage4(x_list)
        assert len(y_list) == 4
        outs = []
        outs.append(y_list[0][0])
        outs.append(self._cat(y_list[1]))
        tmp=[]
        for k in range(0,16,4):
            tmp.append(self._cat(y_list[2][k:k+4]))
        outs.append(self._cat(tmp))

        tmp = []
        for k in range(0, 16*4, 4):
            tmp.append(self._cat(y_list[3][k:k + 4]))
        tmp2 = []
        for k in range(0,16,4):
            tmp2.append(self._cat(tmp[k:k+4]))
        outs.append(self._cat(tmp2))
        return outs

    def train(self, mode=True):
        super(KEEPNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
