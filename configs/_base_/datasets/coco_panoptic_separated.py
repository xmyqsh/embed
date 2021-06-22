dataset_type = 'CocoPanopticSeparatedDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800),],
            multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='AddArea', area_type='bbox'),
    dict(type='AddCenter', center_type='mask'),
    dict(type='MaskRescale', scale_factor=1 / 4),
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='PanopticFCNFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_areas', 'gt_centers', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
'''
train=dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/instances_val2017.json',
    img_prefix=data_root + 'val2017/',
    seg_prefix=data_root + 'panoptic_stuff_val2017/',
    pipeline=train_pipeline),
'''
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        seg_prefix=data_root + 'panoptic_stuff_train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'panoptic_stuff_val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'panoptic_stuff_val2017/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm', 'stuff', 'panoptic'])
