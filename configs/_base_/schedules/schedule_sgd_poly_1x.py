# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
total_epochs = 12
