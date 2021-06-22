_base_ = './panoptic_fcn_r50_400_fast_3x_coco.py'
find_unused_parameters=True
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8)
total_epochs = 12
log_config = dict(interval=20)
optimizer = dict(lr=0.01)
