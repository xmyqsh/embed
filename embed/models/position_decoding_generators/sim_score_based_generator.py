import torch
from .sim_based_generator import SimBasedGenerator
from embed.models import POSITION_DECODING_GENERATORS

@POSITION_DECODING_GENERATORS.register_module()
class SimScoreBasedGenerator(SimBasedGenerator):
    def __init__(self, *args, **kwargs):
        super(SimScoreBasedGenerator, self).__init__(*args, **kwargs)
        assert not self.fuse_score, 'SimScoreBasedGenerator does not support fuse score'

    def kernel_fusion(self, meta_weight, cls, score, **kwargs):
        if self.force_resort:
            idxs = torch.argsort(score, descending=True)
            meta_weight, cls, score = meta_weight[idxs], cls[idxs], score[idxs]
        similarity = self.similarity(meta_weight, sim_type=self.sim_type)
        fuse_mat = similarity.triu(diagonal=0) >= self.sim_thres
        fuse_mat = (fuse_mat & (cls.unsqueeze(1) == cls.unsqueeze(0))) \
                        if self.cls_spec else fuse_mat
        keep = torch.cumsum(fuse_mat, dim=0).diagonal(0) == 1
        fuse_mat, cls = fuse_mat[keep].float(), cls[keep]
        fuse_mat = fuse_mat * score.unsqueeze(0)
        # TODO(ljm): comment the following line
        assert fuse_mat.max(dim=1) == score[keep]
        score = score[keep]
        meta_weight = torch.mm(fuse_mat, meta_weight) / fuse_mat.sum(dim=1, keepdim=True)
        return meta_weight, cls, score

@POSITION_DECODING_GENERATORS.register_module()
class SimScoreBasedGeneratorV2(SimScoreBasedGenerator):
    def __init__(self, *args, **kwargs):
        super(SimScoreBasedGeneratorV2, self).__init__(*args, **kwargs)
        del self.force_resort

    def kernel_fusion(self, meta_weight, cls, score, **kwargs):
        similarity = self.similarity(meta_weight, sim_type=self.sim_type)
        fuse_mat = similarity.triu(diagonal=0) >= self.sim_thres
        fuse_mat = (fuse_mat & (cls.unsqueeze(1) == cls.unsqueeze(0))) \
                        if self.cls_spec else fuse_mat
        keep = torch.cumsum(fuse_mat, dim=0).diagonal(0) == 1
        fuse_mat = fuse_mat[keep] * score.unsqueeze(0)
        meta_weight = torch.mm(fuse_mat, meta_weight) / fuse_mat.sum(dim=1, keepdim=True)
        keep = fuse_mat.argmax(dim=1)
        cls, score = cls[keep], score[keep]
        return meta_weight, cls, score
