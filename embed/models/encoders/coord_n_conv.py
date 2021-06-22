import torch
import torch.nn as nn
import torch.nn.functional as F
from embed.cv.cnn import ConvModule
from embed.cv.runner import auto_fp16

from embed.models import ENCODERS

@ENCODERS.register_module()
class CoordNConv(nn.Module):
    r"""N Convs with Coord as extra input.

    Args:
        in_channels  (int):
        out_channels (int):
        num_convs    (int):
        use_coord   (bool):
        name      (string):
        conv_cfg
        norm_cfg
        act_cfg
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs,
                 use_coord=True,
                 name='',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(CoordNConv, self).__init__()
        self.use_coord = use_coord
        self.name = name

        self.encoders = []
        for k in range(num_convs):
            conv = ConvModule(
                (in_channels + 2 if self.use_coord else in_channels) if k == 0 else out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.encoders.append(conv)
            self.add_module(f'CoordNConv_{name}_{k}', self.encoders[-1])

    def init_weights(self):
        # NOTE: use the default msra initialization method
        pass

    def forward(self, x):
        if self.use_coord:
            x = self.add_coord(x)
        for encoder in self.encoders:
            x = encoder(x)
        return x

    def add_coord(self, feat):
        with torch.no_grad():
            y_pos = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
            x_pos = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
            grid_y, grid_x = torch.meshgrid(y_pos, x_pos)
            grid_yx = torch.stack([grid_y, grid_x], dim=0) \
                           .unsqueeze(0).expand(feat.shape[0], -1, -1, -1)
            feat = torch.cat([feat, grid_yx], dim=1)
        return feat
