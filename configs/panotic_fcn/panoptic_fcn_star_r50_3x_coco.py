_base_ = './panoptic_fcn_r50_3x_coco.py'
model = dict(
    feature_encoding=dict(
        feature_encoder=dict(
            conv_cfg=dict(type='DCNv2', deform_groups=1))))
