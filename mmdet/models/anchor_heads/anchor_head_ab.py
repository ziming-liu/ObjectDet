from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch

from mmdet.core.bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from mmdet.core.utils import multi_apply

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from mmdet.datasets import builder
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        #print(labels.shape)
        #print(label_weights.shape)
        #print(cls_score.shape)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        #print(loss_cls)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        #print(loss_bbox)
        return loss_cls, loss_bbox

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
        #print(len(img_metas))
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        """
        if cfg.online_select:
            cls_reg_targets = self.anchor_target(
                cls_scores.copy(),
                bbox_preds.copy(),
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
        else:
         """
        cls_reg_targets = anchor_target(
            #cls_scores.copy(),
            #bbox_preds.copy(),
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
        #print(num_total_samples)
        losses_cls, losses_bbox = multi_apply( # levels
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

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
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
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
"""
    def anchor_target(self,cls_scores,
                      bbox_preds,
                      anchor_list,
                      valid_flag_list,
                      gt_bboxes_list,
                      img_metas,
                      target_means,
                      target_stds,
                      cfg,
                      gt_bboxes_ignore_list=None,
                      gt_labels_list=None,
                      label_channels=1,
                      sampling=True,
                      unmap_outputs=True):
        """ """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            valid_flag_list (list[list]): Multi level valid flags of each image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            target_means (Iterable): Mean value of regression targets.
            target_stds (Iterable): Std value of regression targets.
            cfg (dict): RPN train configs.

        Returns:
            tuple
        """ """ 
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # split net outputs w.r.t. images
        num_levels = len(anchor_list[0])
        assert len(cls_scores) == len(bbox_preds) == num_levels
        cls_score_list = []
        bbox_pred_list = []
        # change the sort
        for img_id in range(num_imgs):
            #print("cls shape {}".format(cls_scores[0][img_id].shape))
            cls_score_list.append(
                [cls_scores[i][img_id].detach() for i in range(num_levels)])
            bbox_pred_list.append(
                [bbox_preds[i][img_id].detach() for i in range(num_levels)])

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        # for i in range(num_imgs):
        #    assert len(anchor_list[i]) == len(valid_flag_list[i])
        #    anchor_list[i] = torch.cat(anchor_list[i])
        #    valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list) = multi_apply(
            self.anchor_target_single,
            cls_score_list,
            bbox_pred_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            target_means=target_means,
            target_stds=target_stds,
            cfg=cfg,
            label_channels=label_channels,
            sampling=sampling,
            unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = self.images_to_levels(all_labels, num_level_anchors)

        label_weights_list = self.images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = self.images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = self.images_to_levels(all_bbox_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def images_to_levels(self,target, num_level_anchors):
       #Convert targets by image to targets by feature level.

       # [target_img0, target_img1] -> [target_level0, target_level1, ...]
       # 
        #print(target[0].shape)
        #print(target[1].shape)
        target = torch.stack(target, 0)
        level_targets = []
        start = 0
        for n in num_level_anchors:
            end = start + n
            level_targets.append(target[:, start:end].squeeze(0))
            start = end
        return level_targets

    def anchor_target_single(self,cls_score_list,
                             bbox_pred_list,
                             ml_anchors,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             img_meta,
                             target_means,
                             target_stds,
                             cfg,
                             label_channels=1,
                             sampling=True,
                             unmap_outputs=True):
        feat_lvls = self.feat_level_select(cls_score_list.copy(), bbox_pred_list.copy(), ml_anchors,
                                      valid_flags,
                                      gt_bboxes, gt_bboxes_ignore, gt_labels, img_meta, cfg,
                                      unmap_outputs=True)
        labels_ = []
        label_weights_ = []
        bbox_targets_ = []
        bbox_weights_ = []
        pos_inds_ = []
        neg_inds_ = []
        device = bbox_pred_list[0].device
        img_h, img_w, _ = img_meta['pad_shape']
        #print("ml achor {}".format(len(ml_anchors)))
        for lvl, flat_anchors in enumerate(ml_anchors):  # each level
            inds = torch.nonzero(feat_lvls == lvl).squeeze(-1)

            valid_flag = valid_flags[lvl]
            inside_flags = self.anchor_inside_flags(flat_anchors, valid_flag,
                                                    img_meta['img_shape'][:2],
                                                    cfg.allowed_border)

            if not inside_flags.any():
                return (None,) * 6
            # assign gt and sample anchors
            anchors = flat_anchors[inside_flags, :]
            #print(flat_anchors.shape)

            num_valid_anchors = anchors.shape[0]
            bbox_targets = torch.zeros_like(anchors)
            bbox_weights = torch.zeros_like(anchors)
            labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
            label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
            pos_inds = None
            neg_inds = None#torch.LongTensor([0]).cuda()

            if len(inds) > 0:
                boxes = gt_bboxes[inds, :]
                classes = gt_labels[inds]
                ##
                num_levels = len(ml_anchors)
                assert len(cls_score_list) == len(bbox_pred_list) == num_levels

                if self.sampling:
                    assign_result, sampling_result = assign_and_sample(
                        anchors, boxes, gt_bboxes_ignore, None, cfg)
                else:
                    bbox_assigner = build_assigner(cfg.assigner)
                    assign_result = bbox_assigner.assign(anchors, boxes,
                                                         gt_bboxes_ignore, classes)
                    bbox_sampler = PseudoSampler()
                    sampling_result = bbox_sampler.sample(assign_result, anchors,
                                                          boxes)

                pos_inds = sampling_result.pos_inds
                neg_inds = sampling_result.neg_inds
                if len(pos_inds) > 0:
                    pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                                  sampling_result.pos_gt_bboxes,
                                                  target_means, target_stds)
                    bbox_targets[pos_inds, :] = pos_bbox_targets
                    bbox_weights[pos_inds, :] = 1.0
                    if classes is None:
                        labels[pos_inds] = 1
                    else:
                        labels[pos_inds] = classes[sampling_result.pos_assigned_gt_inds]
                    if cfg.pos_weight <= 0:
                        label_weights[pos_inds] = 1.0
                    else:
                        label_weights[pos_inds] = cfg.pos_weight
                if len(neg_inds) > 0:
                    label_weights[neg_inds] = 1.0

                # map up to original set of anchors
            if unmap_outputs:
                num_total_anchors = flat_anchors.size(0)
                labels = self.unmap(labels, num_total_anchors, inside_flags)
                label_weights = self.unmap(label_weights, num_total_anchors, inside_flags)
                bbox_targets = self.unmap(bbox_targets, num_total_anchors, inside_flags)
                bbox_weights = self.unmap(bbox_weights, num_total_anchors, inside_flags)
            labels_.append(labels)
            label_weights_.append(label_weights)
            bbox_targets_.append(bbox_targets)
            bbox_weights_.append(bbox_weights)
            if pos_inds is not None:
                pos_inds_.append(pos_inds)
            if neg_inds is not None:
                neg_inds_.append(neg_inds)
        labels_ = torch.cat(tuple(labels_), 0)
        label_weights_ = torch.cat(tuple(label_weights_), 0)
        bbox_targets_ = torch.cat(tuple(bbox_targets_), 0)
        bbox_weights_ = torch.cat(tuple(bbox_weights_), 0)
        pos_inds_ = torch.cat(tuple(pos_inds_), 0)
        neg_inds_ = torch.cat(tuple(neg_inds_), 0)
        return (labels_, label_weights_, bbox_targets_, bbox_weights_, pos_inds_,
                neg_inds_)

    def feat_level_select(self,cls_score_list, bbox_pred_list, ml_anchors,
                          valid_flags, gt_bboxes_, gt_bboxes_ignore_,
                          gt_labels_, img_meta, cfg,
                          unmap_outputs=True):

        # cfg['online_select'] = True
        # if cfg.online_select:
        num_levels = len(cls_score_list)
        #print("num level {}".format(num_levels))
        #for jj in range(num_levels):
        #    print(cls_score_list[jj].shape)
        num_boxes = gt_bboxes_.size(0)
        feat_losses = gt_bboxes_.new_zeros((num_boxes, num_levels))
        device = bbox_pred_list[0].device
        for lvl in range(num_levels):

            for i in range(num_boxes):
                cls_score = cls_score_list[lvl]
                bbox_pred = bbox_pred_list[lvl]
                flat_anchors = ml_anchors[lvl]
                valid_flag = valid_flags[lvl]

                gt_bboxes = gt_bboxes_[i].unsqueeze(0)
                gt_labels = gt_labels_[i].unsqueeze(0)
                gt_bboxes_ignore = gt_bboxes_ignore_
                inside_flags = self.anchor_inside_flags(flat_anchors, valid_flag,
                                                   img_meta['img_shape'][:2],
                                                   cfg.allowed_border)
                if not inside_flags.any():
                    return (None,) * 6
                # assign gt and sample anchors
                anchors = flat_anchors[inside_flags, :]
                ##
                num_levels = len(ml_anchors)
                assert len(cls_score_list) == len(bbox_pred_list) == num_levels

                if self.sampling:
                    assign_result, sampling_result = assign_and_sample(
                        anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
                else:
                    bbox_assigner = build_assigner(cfg.assigner)
                    assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                         gt_bboxes_ignore, gt_labels)
                    bbox_sampler = PseudoSampler()
                    sampling_result = bbox_sampler.sample(assign_result, anchors,
                                                          gt_bboxes)

                num_valid_anchors = anchors.shape[0]
                bbox_targets = torch.zeros_like(anchors)
                bbox_weights = torch.zeros_like(anchors)
                labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
                label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

                pos_inds = sampling_result.pos_inds
                neg_inds = sampling_result.neg_inds
                #print("pos inds {}".format(pos_inds))
                if len(pos_inds) > 0:
                    pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                                  sampling_result.pos_gt_bboxes,
                                                  )
                    bbox_targets[pos_inds, :] = pos_bbox_targets
                    bbox_weights[pos_inds, :] = 1.0
                    if gt_labels is None:
                        labels[pos_inds] = 1
                    else:
                        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
                    if cfg.pos_weight <= 0:
                        label_weights[pos_inds] = 1.0
                    else:
                        label_weights[pos_inds] = cfg.pos_weight
                if len(neg_inds) > 0:
                    label_weights[neg_inds] = 1.0

                # map up to original set of anchors
                # if unmap_outputs:
                num_total_anchors = flat_anchors.size(0)
                labels = self.unmap(labels, num_total_anchors, inside_flags)
                label_weights = self.unmap(label_weights, num_total_anchors, inside_flags)
                bbox_targets = self.unmap(bbox_targets, num_total_anchors, inside_flags)
                bbox_weights = self.unmap(bbox_weights, num_total_anchors, inside_flags)
                #print("pos indx :{}".format(pos_inds.shape))

                num_total_samples = (
                    pos_inds.size(0) + neg_inds.size(0) if self.sampling else pos_inds.size(0))
                #print(" toto sample {}".format(num_total_samples.shape))
                # compute loss
                # classification loss
                labels = labels.reshape(-1)
                label_weights = label_weights.reshape(-1)
                #print(label_weights)
                #print("cls sorce {}".format(cls_score.shape))
                cls_score = cls_score.permute(1,2,0).reshape(-1,
                                                              self.cls_out_channels)
                loss_cls = self.loss_cls(
                    cls_score, labels, label_weights, avg_factor=1.0)
                # regression loss
                bbox_targets = bbox_targets.reshape(-1, 4)
                bbox_weights = bbox_weights.reshape(-1, 4)
                bbox_pred = bbox_pred.permute(1,2,0).reshape(-1, 4)
                loss_bbox = self.loss_bbox(
                    bbox_pred,
                    bbox_targets,
                    bbox_weights,
                    avg_factor=1.0)
                #print(torch.sum(loss_cls))
                #print(torch.sum(loss_bbox) )
                feat_losses[i, lvl] = (torch.sum(loss_cls)+torch.sum(loss_bbox))/num_total_samples
        #print("feat losses {}".format(feat_losses))
        feat_levels = torch.argmin(feat_losses, dim=1)
        #print("num bbox {}".format(num_boxes))
        #print("feat level \n {}".format(feat_levels))
        return feat_levels

    def anchor_inside_flags(self,flat_anchors, valid_flags, img_shape,
                            allowed_border=0):
        img_h, img_w = img_shape[:2]
        if allowed_border >= 0:
            inside_flags = valid_flags & \
                           (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
                           (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
                           (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
                           (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
        else:
            inside_flags = valid_flags
        return inside_flags

    def unmap(self, data, count, inds, fill=0):
       #  Unmap a subset of item (data) back to the original set of items (of
       # size count) 
        if data.dim() == 1:
            ret = data.new_full((count,), fill)
            ret[inds] = data
        else:
            new_size = (count,) + data.size()[1:]
            ret = data.new_full(new_size, fill)
            ret[inds, :] = data
        return ret
"""