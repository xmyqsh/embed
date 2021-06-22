_base_ = './panoptic_fcn_star_r50_3x_coco.py'
model = dict(
    position_decoding_head=dict(
        thing_head=dict(thres=0.2),
        stuff_head=dict(thres=0.1)))
