import torch.nn as nn
import torch.nn.functional as F
from embed.cv.cnn import ConvModule, bias_init_with_prob, normal_init
from embed.cv.runner import auto_fp16

from embed.models import build_loss

class BasePositionEmbeddingHead(nn.Module):
    r"""Base Position Embedding Head.

    Args:
        in_channels (int):
        num_classes (int):
        thres       (float):
        loss_pos    (str):
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 thres,
                 loss_pos,
                 **kwargs):
        super(BasePositionEmbeddingHead, self).__init__()
        self.num_classes = num_classes
        self.thres = thres

        self.pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.loss_pos = build_loss(loss_pos)

    def init_weights(self):
        # Note: default prob in RetinaNet Focalloss is 0.01
        bias = bias_init_with_prob(0.1)
        normal_init(self.pred, std=0.01, bias=bias)


    def forward_single_level(self, x):
        return self.pred(x)

    def forward(self, x):
        return list(map(self.forward_single_level, x))

    def forward_train(self, x, pred_weights, gt_semantic_seg):
        pass

    def simple_test(self, x, pred_weights):
        preds = self(x)
        idx_feats, classes, scores, nums = \
            self.get_positions(preds, pred_weights)

        return dict(idx_feats=idx_feats,
                    classes=classes,
                    scores=scores,
                    nums=nums)
