import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmdet.models.necks.self_attention import MultiHeadAttention
from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class customFPN(nn.Module):

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
                 with_att=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(customFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.interval = interval
        self.with_att = with_att
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
        self.outs_stides = []
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        if with_att:
            # self.attention2048 = MultiHeadAttention(n_head=3,d_model=2048,d_v=2048,d_k=2048)
            # self.attention1024 = MultiHeadAttention(n_head=3,d_model=1024,d_v=1024,d_k=1024)
            # self.attention512 = MultiHeadAttention(n_head=3,d_model=512,d_v=512,d_k=512)
            # self.attentionlv0 = MultiHeadAttention(n_head=3,d_model=256,d_v=256,d_k=256)
            self.attentionlv1 = MultiHeadAttention(n_head=3, d_model=256, d_v=64, d_k=64)
            self.attentionlv2 = MultiHeadAttention(n_head=3, d_model=256, d_v=64, d_k=64)
            self.attentionlv3 = MultiHeadAttention(n_head=3, d_model=256, d_v=64, d_k=64)

            self.maxpooling = nn.AdaptiveMaxPool2d((6, 6))  # nn.MaxPool2d(kernel_size=6,stride=6,padding=0)

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

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):

            #print(laterals[i].shape)
           # print(F.interpolate(
            #    laterals[i], scale_factor=1.36, mode='nearest').shape)
            scale_factor = laterals[i-1].shape[3] / laterals[i].shape[3]
            next = F.interpolate(
                laterals[i], scale_factor=scale_factor, mode='nearest')
            if self.with_att:
                # 2 得到pooling的小的全局特征
                global_feat = self.maxpooling(laterals[i])
                # 3 根据全局特征 去 补充插值结果缺失的特征信息
                b1, c1, h1, w1 = next.shape
                next = next.reshape(b1, c1, -1).permute(0, 2, 1)
                # 索引缩小 过大的特征图
                threshold = 36
                if h1 > threshold and w1 > threshold:
                    hidx = torch.LongTensor([i for i in range(0, h1, h1 // threshold)]).cuda()
                    widx = torch.LongTensor([i for i in range(0, w1, w1 // threshold)]).cuda()
                    idx = w1 * hidx.repeat(len(widx), 1).t().reshape(-1) + widx.repeat(1, len(hidx)).reshape(-1)
                    # print(idx)
                    sparse_Next = next[:, idx, :]
                    # print(len(idx))
                else:
                    sparse_Next = next[:, :, :]
                b2, c2, h2, w2 = global_feat.shape
                global_feat = global_feat.reshape(b2, c2, -1).permute(0, 2, 1)
                # if i == 0:
                #    next = self.attentionlv0(next, global_feat, global_feat)
                if i == 1:
                    sparse_Next = self.attentionlv1(sparse_Next, global_feat, global_feat)
                elif i == 2:
                    sparse_Next = self.attentionlv2(sparse_Next, global_feat, global_feat)
                elif i == 3:
                    sparse_Next = self.attentionlv3(sparse_Next, global_feat, global_feat)
                if h1 > threshold and w1 > threshold:
                    next[:, idx, :] = sparse_Next
                else:
                    next = sparse_Next
                next = next.permute(0, 2, 1).reshape(b1, c1, h1, w1)
            w = min(next.shape[3],laterals[i-1].shape[3])
            h = min(next.shape[2],laterals[i-1].shape[2])
            next = next[:,:,:h,:w]
            laterals[i-1]= laterals[i-1][:,:,:h,:w]
            laterals[i-1] += next

            #laterals[i - 1] += F.interpolate(
             #   laterals[i], scale_factor=self.interval, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
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
