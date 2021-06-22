import torch
from .base_position_decoding_generator import BasePositionDecodingGenerator
from embed.models import POSITION_DECODING_GENERATORS

@POSITION_DECODING_GENERATORS.register_module()
class ClsScoreBasedGenerator(BasePositionDecodingGenerator):
    def __init__(self):
        super(ClsScoreBasedGenerator, self).__init__()

    def kernel_fusion(self, meta_weight, cls, score, **kwargs):
        unique_cls = torch.unique(cls)
        fuse_mat = (unique_cls.unsqueeze(1) == cls.unsqueeze(0)).float()
        fuse_mat = fuse_mat * score.unsqueeze(0)
        score = fuse_mat.max(dim=1)
        meta_weight = torch.mm(fuse_mat, meta_weight) / fuse_mat.sum(dim=1, keepdim=True)
        return meta_weight, unique_cls, score
