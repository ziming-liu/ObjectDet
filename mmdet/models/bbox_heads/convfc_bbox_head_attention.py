import torch
import torch.nn as nn
from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
from ..losses import accuracy
from .self_attention import MultiHeadAttention
@HEADS.register_module
class ATConvFCBBoxHead(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=2,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=2,
                 num_reg_convs=0,
                 num_reg_fcs=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ATConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        print(num_shared_convs)
        print(num_shared_fcs)
        print(num_cls_convs)
        print(num_cls_fcs)
        print(num_reg_convs)
        print(num_reg_fcs)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.attention = MultiHeadAttention(3,1024,512,512)

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls2 = nn.Sequential(nn.Linear(self.cls_last_dim, self.cls_last_dim),
                                        nn.Linear(self.cls_last_dim, self.num_classes),)
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ATConvFCBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #print(x.shape)
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        #print(x.shape)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        #print(x.shape)
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)


        #assert x_cls.dim() >2
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            # self attention
            #b,c,h,w = x_cls_at.shape
            #x_cls_at = x_cls_at.continous().view(b,c,h*w).permute(0,2,1)
            #x_cls_at,_ = self.attention(x_cls_at,x_cls_at,x_cls_at)
            #x_cls_at = x.permute(0,2,1).view(b,c,h,w)

            x_cls = x_cls.view(x_cls.size(0), -1)
            #x_cls_at = x_cls_at.view(x_cls_at.size(0),-1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
            #x_cls_at = self.relu(fc(x_cls_at))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        #x_cls_at = x_cls.unsqueeze(0)
        #x_cls_at, _ = self.attention(x_cls_at, x_cls_at, x_cls_at)
        #x_cls_at = x_cls_at.squeeze(0)
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        cls_at_score = self.fc_cls2(x_cls) if self.with_cls else None
        #cls_at_score =cls_score
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score,cls_at_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             cls_at_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,big_label_weights,small_label_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:

            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            """
            avg_factor = max(torch.sum(small_label_weights > 0).float().item(), 1.)
            small_loss_cls = self.loss_cls(
                cls_score,
                labels,
                small_label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)

            avg_factor = max(torch.sum(big_label_weights > 0).float().item(), 1.)
            big_loss_cls= self.loss_cls(
                cls_score,
                labels,
                big_label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            avg_factor = max(torch.sum(small_label_weights > 0).float().item(), 1.)
            """
            losses['at_small_loss_cls'] = self.loss_cls(
                cls_at_score,
                labels,
                small_label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            print("attention-- {}".format(losses['at_small_loss_cls'].item()))
            #print("1-- {}".format(max(0,losses['at_small_loss_cls'].item()-small_loss_cls)))
           # print("2-- {}".format(max(0, losses['at_small_loss_cls'].item() - big_loss_cls)))
            print("cls --  {}".format(max(0,losses['at_small_loss_cls'].item()-losses['loss_cls'].item())))
            losses['at_small_loss_cls'] = losses['at_small_loss_cls']+ max(0,losses['at_small_loss_cls'].item()-losses['loss_cls'].item())
                                         # max(0,losses['at_small_loss_cls'].item()-small_loss_cls)+\
                        #    max(0, losses['at_small_loss_cls'].item() - big_loss_cls)

            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

@HEADS.register_module
class ATSharedFCBBoxHead(ATConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(ATSharedFCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
