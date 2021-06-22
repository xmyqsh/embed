_base_ = [
    '../_base_/models/panoptic_fcn_r50.py',
    '../_base_/datasets/coco_panoptic_separated.py',
    '../_base_/schedules/schedule_sgd_poly_1x.py', '../_base_/default_runtime.py'
]
find_unused_parameters=True
