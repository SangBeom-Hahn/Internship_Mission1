# dataset settings
dataset_type = 'KFashionDataset'
data_root = '../data'
# classes = ('blouse', 'cardigan', 'coat', 'jacket', 'jumper', 'shirt')
classes = ('blouse', 'cardigan', 'coat', 'jacket', 'jumper', 'shirt', 'sweater', 't-shirt', 'vest', 'pants', 'skirt', 'onepiece(dress)', 'onepiece(jumpsuite)')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/test.json',
        img_prefix=data_root + 'test/',
        classes=classes,
        pipeline=train_pipeline),        
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/val.json',
        img_prefix=data_root + 'val/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/test_2.json',
        img_prefix=data_root + '/images/test_image/',
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(classwise=True, metric=['segm'])
