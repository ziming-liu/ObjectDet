import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import multi_apply
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class DeshHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(DeshHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        #print(labels[:50])
        label_weights = label_weights.reshape(-1)
        b,c,h,w = cls_score.shape
        cls_score = cls_score.permute(0, 2, 3,
                                     1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score - cls_score.mean(1).reshape(-1,1)
        # 归一化 到-500 +500
        #print(cls_score)
        #min_score,_ = torch.min(cls_score,1)
        #max_score,_ = torch.max(cls_score,1)
       # k = (1000-(-1000)) / (max_score.reshape(-1,1) - min_score.reshape(-1,1))
        #cls_score = -1000 + k*(cls_score-min_score.reshape(-1,1))
        #print(cls_score)
        cls_score = cls_score.reshape(b,h,w,c)
        #cls_score = cls_score.scatter_(1,torch.zeros(cls_score.size(0),1).long().cuda(),1)
        cls_score_h = cls_score.mean(2).reshape(-1, self.cls_out_channels)
        #cls_score_h = cls_score_h.scatter_(1,torch.zeros(cls_score_h.size(0),1).long().cuda(),1)

        cls_score_w = cls_score.mean(1).reshape(-1, self.cls_out_channels)
        #cls_score_w = cls_score_w.scatter_(1,torch.zeros(cls_score_w.size(0),1).long().cuda(),1)

        #cls_score_w = cls_score.permute(0, 2, 3,
        #                                1).softmax(3).mean(1).reshape(-1, self.cls_out_channels)
        labels_grid = labels.contiguous().view(b,h,w,self.num_anchors)
        labels_grid_h = labels_grid.contiguous().permute(0,1,3,2).contiguous().view(-1,w)
        labels_grid_w = labels_grid.contiguous().permute(0,2,3,1).contiguous().view(-1,h)

        def to_onehot(labels):
            one_hots = []
            batchsize,n_lable = labels.shape
            one_hot = torch.zeros(batchsize, self.cls_out_channels).cuda().scatter_(1, labels, 1)
            """ 
            for bi in range(batchsize):
                bi_label = set(labels[bi].tolist())
                bi_label = list(bi_label)
                bi_label = [bi_label[i] for i in range(len(bi_label)) if bi_label[i]!=0]

                #bi_label = torch.LongTensor(list(set(labels[bi].tolist()))).cuda()-1
                #bi_label = bi_label[bi_label>=0]
                if len(bi_label) ==0:
                    one_hots.append(torch.zeros(1, self.cls_out_channels).cuda())
                else:
                    idx = torch.LongTensor(list(bi_label)).reshape(1,-1).cuda() - 1
                    one_hots.append(torch.zeros(1, self.cls_out_channels).cuda().scatter_(1,idx,1))
                    #one_hots.append(torch.zeros(1, self.cls_out_channels).cuda().scatter_(1, bi_label.reshape(1,-1), 1))


            #labels = labels.contiguous().view(-1,n_lable)
            #one_hot = torch.zeros(batchsize, self.cls_out_channels).cuda().scatter_(1, labels, 1)
            one_hots = torch.cat(one_hots,0)
            """
            return one_hot

        labels_grid_h = to_onehot(labels_grid_h)#.contiguous().view(-1,self.cls_out_channels)
        labels_grid_w = to_onehot(labels_grid_w)#.contiguous().view(-1,self.cls_out_channels)
        #print(labels_grid_w[:5,40])
        #print(labels_grid_h[:5,40:])
        #cls_score_h = cls_score_h.sigmoid()
        #cls_score_w = cls_score_w.sigmoid()
        #print(cls_score_w[100:111])
        #print(labels_grid_w[100:111])
        #print(cls_score_w.shape[0])
        label_weights_h = torch.ones(labels_grid_h.size(0),self.cls_out_channels).cuda() \
            .scatter_(1, torch.zeros(labels_grid_h.size(0), 1).long().cuda(), 0)
        label_weights_w = torch.ones(labels_grid_w.size(0), self.cls_out_channels).cuda() \
            .scatter_(1, torch.zeros(labels_grid_w.size(0), 1).long().cuda(), 0)
        #print(label_weights_h)
        loss_cls_h = self.loss_cls(
            cls_score_h, labels_grid_h, label_weights_h, avg_factor=cls_score_h.shape[0]*self.cls_out_channels)
        loss_cls_w = self.loss_cls(
            cls_score_w, labels_grid_w, label_weights_w, avg_factor=cls_score_w.shape[0]*self.cls_out_channels)

        #loss_cls = self.loss_cls(
        #    cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls_h,loss_cls_w, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls_h,losses_cls_w, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_cls_h=losses_cls_h,loss_cls_w=losses_cls_w, loss_bbox=losses_bbox)

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            """ 
            c,h,w = cls_score.shape
            assert c//self.num_anchors==self.cls_out_channels
            cls_score_h = cls_score.permute(1,2,0).mean(1).reshape(-1,c)
            cls_score_w = cls_score.permute(1,2,0).mean(0).reshape(-1,c)
            assert self.use_sigmoid_cls is True
            assert cls_score_h.shape[0]==h
            assert cls_score_w.shape[0]==w
            scores_h = cls_score_h.sigmoid()
            scores_w = cls_score_w.sigmoid()
            cls_score = torch.zeros(h,w,c).cuda()
            for hi in range(h):
                for wi in range(w):
                    cls_score[hi][wi] = scores_h[hi] + scores_w[wi]
            cls_score = cls_score.reshape(-1,self.cls_out_channels)
            """
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            cls_score = cls_score[:,1:]
            #print(cls_score)
            if self.use_sigmoid_cls:
                scores = cls_score.softmax(-1)
            else:

                scores = cls_score.softmax(-1)
           # print(scores)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
