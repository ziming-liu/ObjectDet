model = dict(
    type='PGCascadeRCNNmod2',
    num_stages=3,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        pretrained='modelzoo://resnet50',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',with_cp=True),
    backbone_deep=dict(
        type='ResNet',
        pretrained='modelzoo://resnet101',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',with_cp=True),
    backbone_shallow=dict(
        type='ResNet',
        pretrained='modelzoo://resnet18',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',with_cp=True),
    neck=dict(
        type='PGFPN',
        in_channels=[64, 128, 256, 512, 256, 512, 1024, 2048, 256, 512, 1024, 2048,],
        out_channels=256,
        end_level=7,
        num_outs=7),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[2,4, 8, 16, 32, 64,128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[2,4, 8, 16, 32,64 ]),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=11,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=11,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=11,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0))
    ])
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100),
    keep_all_stages=False)
# dataset settings
#task_type='image'
dataset_type = 'VisDroneDataset'
data_root = '/home/share2/VisDrone2019/TASK1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-train-patches/img_size_anno.json',
        img_prefix=data_root + 'VisDrone2019-DET-train-patches/',
        img_scale=(1000,600),
        #extra_aug=dict(
        #    slidewin=dict(
        #        winsize=(0.3,0.3))),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32*2,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-val/img_size_anno.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/',
        img_scale=[(1000 * 2, 600 * 2)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32*2,
        #slide_test = dict(winsize=(0.5,0.5),
        #                overlapratio = 0.3,
        #                ),
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-test-challenge/img_size_anno.json',
        img_prefix=data_root + 'VisDrone2019-DET-test-challenge/',
        img_scale=[(1500, 1000), (1500 * 1.5, 1000 * 1.5), (1500 * 2, 1000 * 2)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32*2,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
fp16 = dict(coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 distributed=True)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[7-4, 11-6])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 6
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pgnet_visdrone_cascade_rcnn_r50_pgfpn_1x_patch'
load_from = './work_dirs/pg_cascade_rcnn_r50_fpn_1x_visdrone_pathch/epoch_12.pth'
resume_from = None
workflow = [('train', 1)]
