model = dict(
    type='RetinaNet',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='IPN_kite',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='kiteFPN',
        in_channels=[256, 256, 256, 256, 512, 512, 512, 1024, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=False,
        num_outs=11),
    bbox_head=dict(
        type='FSAFHead',
        num_classes=21,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_factor=4.0,
        feat_strides=[4,4/0.8333333,4/0.66666666,4/0.5,8/0.8333333,8/0.6666666,8/0.5,16/0.666666666,16/0.5,32/0.5,128]))
# training and testing settings
train_cfg = dict(
    pos_scale=0.2,
    ignore_scale=0.5,
    gamma=2.0,
    alpha=0.25,
    canonical_scale=224.0,
    canonical_level=2,
    bbox_reg_weight=1.0,
    online_select=True,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=3000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=1000)
## dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',  # to avoid reloading datasets frequently
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            img_scale=(1000, 600),
            img_norm_cfg=img_norm_cfg,
            size_divisor=32*8,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=True,
            with_label=True)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(1000, 600),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32*8,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        img_scale=(1000, 600),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32*8,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[3])  # actual epoch = 3 * 3 = 9
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fsaf_r50_fpn_1x_kite_voc/'
load_from = None#'/home/share2/VisDrone2019/TASK1/outfile/retinanet_r50_fpn_1x/epoch_12.pth' #'/home/share2/VisDrone2019/TASK1/outfile/retinanet_r50_fpn_1x_second/epoch_1.pth'
resume_from = None#'/home/share2/VisDrone2019/TASK1/outfile/retinanet_bpn_r50_fpn_1x/epoch_10.pth'
workflow = [('train', 1)]
