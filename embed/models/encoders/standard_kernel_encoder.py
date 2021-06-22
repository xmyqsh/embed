import torch.nn as nn
import torch.nn.functional as F
from embed.cv.cnn import normal_init
from embed.cv.runner import auto_fp16

from embed.models import ENCODERS, build_encoder

@ENCODERS.register_module()
class StandardKernelEncoder(nn.Module):
    r"""Fused Semantic FPN.
    """
    def __init__(self,
                 start_level,
                 num_ins,
                 weight_encoder,
                 in_channels,
                 out_channels):
        super(StandardKernelEncoder, self).__init__()
        self.start_level = start_level
        self.num_ins = num_ins
        self.weight_encoder = build_encoder(weight_encoder)
        self.pred_weight = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def init_weights(self):
        self.weight_encoder.init_weights()
        normal_init(self.pred_weight, std=0.01, bias=0)

    def forward_single_level(self, x):
        x = self.weight_encoder(x)
        return self.pred_weight(x)

    def forward(self, inputs):
        return list(map(self.forward_single_level,
                        inputs))
#                        inputs[self.start_level : self.start_level + self.num_ins]))

    def forward_train(self, inputs):
        return self(inputs[self.start_level : self.start_level + self.num_ins])

    def simple_test(self, inputs):
        return self(inputs[self.start_level : self.start_level + self.num_ins])
