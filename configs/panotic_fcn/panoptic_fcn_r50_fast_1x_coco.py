_base_ = './panoptic_fcn_r50_1x_coco.py'
model = dict(
    position_decoding_head=dict(
        thing_head=dict(thres=0.2),
        stuff_head=dict(thres=0.1)))