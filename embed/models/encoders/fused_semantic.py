import torch.nn as nn
from embed.cv.cnn import ConvModule
from embed.cv.runner import auto_fp16

from embed.models import ENCODERS, build_encoder

@ENCODERS.register_module()
class FusedSemantic(nn.Module):
    r"""Fused Semantic FPN.

    This is an implementation of paper `Panoptic FPN Network
    <https://arxiv.org/abs/xxxx.xxxxx>`_.
    It takes a list of FPN features as input, and applies a sequence of
    3x3 convs and upsampling to scale all of them to the stride defined by
    ``common_stride``. Then these features are added and used to make final
    predictions by another 1x1 conv layer.

    Args:
        in_channels (int):
        out_channels (int):
        num_ins (int):
        start_level (int):
        fusion_level (int):

        ...

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_ins,
                 start_level=0,
                 fusion_level=0,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 feature_encoder=None):
        super(FusedSemantic, self).__init__()
        assert 'scale_factor' not in upsample_cfg

        self.num_ins = num_ins
        self.start_level = start_level

        self.scale_heads = []
        for cur_level in range(start_level, start_level + num_ins):
            head_ops = []
            head_length = max(
                1, abs(cur_level - fusion_level)
            )
            for k in range(head_length):
                conv = ConvModule(
                    in_channels if k == 0 else out_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                head_ops.append(conv)
                if cur_level != fusion_level:
                    head_ops.append(
                        nn.Upsample(
                            scale_factor=2 if cur_level > fusion_level else 0.5,
                            **upsample_cfg
                        ),
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(f'S{cur_level}', self.scale_heads[-1])

        if feature_encoder is not None:
            self.feature_encoder = build_encoder(feature_encoder)

    @property
    def with_feature_encoder(self):
        """bool: whether the detector has a feature_encoder"""
        return hasattr(self, 'feature_encoder') and self.feature_encoder is not None

    def init_weights(self):
        # NOTE: use the default msra initialization method
        if self.with_feature_encoder:
            self.feature_encoder.init_weights()

    def forward(self, inputs):
        """Forward function."""
        #for i, x in enumerate(inputs[self.start_level : self.start_level + self.num_ins]):
        for i, x in enumerate(inputs):
            feature_fused = self.scale_heads[i](x) if i == 0 else feature_fused + self.scale_heads[i](x)
        return self.feature_encoder(feature_fused) if self.with_feature_encoder else feature_fused

    def forward_train(self, inputs):
        return self(inputs[self.start_level : self.start_level + self.num_ins])

    def simple_test(self, inputs):
        return self(inputs[self.start_level : self.start_level + self.num_ins])
