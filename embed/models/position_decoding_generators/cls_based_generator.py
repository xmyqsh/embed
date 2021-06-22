import torch
from .base_position_decoding_generator import BasePositionDecodingGenerator
from embed.models import POSITION_DECODING_GENERATORS

@POSITION_DECODING_GENERATORS.register_module()
class ClsBasedGenerator(BasePositionDecodingGenerator):
    def __init__(self):
        super(ClsBasedGenerator, self).__init__()

    def kernel_fusion(self, meta_weight, cls, score, **kwargs):
        unique_cls = torch.unique(cls)
        fuse_mat = (unique_cls.unsqueeze(1) == cls.unsqueeze(0)).float()
        fuse_mat = fuse_mat / fuse_mat.sum(dim=1, keepdim=True)
        meta_weight = torch.mm(fuse_mat, meta_weight)
        score = torch.mm(fuse_mat, score.unsqueeze(-1)).squeeze(-1)
        return meta_weight, unique_cls, score
