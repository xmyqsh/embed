import torch.nn as nn
import torch.nn.functional as F
from embed.cv.cnn import ConvModule
from embed.cv.runner import auto_fp16

from embed.models import POSITION_EMBEDDING_HEADS, build_position_embedding_head, build_encoder

@POSITION_EMBEDDING_HEADS.register_module()
class CombinedPositionEmbeddingHead(nn.Module):
    r"""Combined Position Embedding Head.

    Args:
        thing_head (dict):
        stuff_head (dict):
    """

    def __init__(self,
                 start_level,
                 num_ins,
                 position_encoder,
                 thing_head,
                 stuff_head,
                 **kwargs):
        super(CombinedPositionEmbeddingHead, self).__init__()
        self.start_level = start_level
        self.num_ins = num_ins

        self.position_encoder = build_encoder(position_encoder)
        thing_head.update(**kwargs)
        stuff_head.update(**kwargs)
        self.thing_head = build_position_embedding_head(thing_head)
        self.stuff_head = build_position_embedding_head(stuff_head)

    def init_weights(self):
        self.position_encoder.init_weights()
        self.thing_head.init_weights()
        self.stuff_head.init_weights()

    def forward_single_level(self, x):
        return self.position_encoder(x)

    def forward(self, inputs):
        return list(map(self.forward_single_level,
                        inputs))
#                        inputs[self.start_level : self.start_level + self.num_ins]))

    def forward_train(self, features, pred_weights, input_shape,
                                      gt_semantic_seg,
                                      **kwargs):
        position_encoded = self(features[self.start_level : self.start_level + self.num_ins])
        losses_th, positions_th = \
            self.thing_head.forward_train(position_encoded, pred_weights,
                                          input_shape=input_shape,
                                          **kwargs)
        losses_st, positions_st = \
            self.stuff_head.forward_train(position_encoded, pred_weights,
                                          gt_semantic_seg=gt_semantic_seg)
        losses = dict()
        losses.update(losses_th)
        losses.update(losses_st)
        return losses, dict(positions_th=positions_th,
                            positions_st=positions_st)

    def simple_test(self, features, pred_weights):
        position_encoded = self(features[self.start_level : self.start_level + self.num_ins])
        positions_th = self.thing_head.simple_test(position_encoded, pred_weights)
        positions_st = self.stuff_head.simple_test(position_encoded, pred_weights)
        return dict(positions_th=positions_th,
                    positions_st=positions_st)
