import torch
from .base_position_decoding_generator import BasePositionDecodingGenerator
from embed.models import POSITION_DECODING_GENERATORS

@POSITION_DECODING_GENERATORS.register_module()
class SimBasedGenerator(BasePositionDecodingGenerator):
    def __init__(self, sim_type,
                       sim_thres,
                       cls_spec=True,
                       fuse_score=False,
                       force_resort=True):
        super(SimBasedGenerator, self).__init__()
        self.sim_type = sim_type
        self.sim_thres = sim_thres
        self.cls_spec = cls_spec
        self.fuse_score = fuse_score
        self.force_resort = force_resort

        assert self.force_resort, 'we need force_resort for the naive implementation, \
                                   or use SimScoreBasedGeneratorV2'

    def kernel_fusion(self, meta_weight, cls, score):
        if self.force_resort:
            idxs = torch.argsort(score, descending=True)
            meta_weight, cls, score = meta_weight[idxs], cls[idxs], score[idxs]
        similarity = self.similarity(meta_weight, sim_type=self.sim_type)
        fuse_mat = similarity.triu(diagonal=0) >= self.sim_thres
        fuse_mat = (fuse_mat & (cls.unsqueeze(1) == cls.unsqueeze(0))) \
                        if self.cls_spec else fuse_mat
        keep = torch.cumsum(fuse_mat, dim=0).diagonal(0) == 1
        fuse_mat, cls = fuse_mat[keep].float(), cls[keep]
        fuse_mat = fuse_mat / fuse_mat.sum(dim=1, keepdim=True)
        meta_weight = torch.mm(fuse_mat, meta_weight)
        # TODO(ljm): try to set fuse_score=True as ClsBasedGenerator does.
        #            NOTE: the sim_thres should be further tuned for it specifically.
        score = torch.mm(fuse_mat, score.unsqueeze(-1)).squeeze(-1) \
                            if self.fuse_score else score[keep]
        return meta_weight, cls, score

    def similarity(self, w, sim_type='cos'):
        return eval(f'self.{sim_type}_similarity')(w)

    @staticmethod
    def cos_similarity(w):
        w_norm = w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.mm(w_norm, w_norm.t())

    @staticmethod
    def L2_similarity(w):
        # NOTE: the sim_thres should be tuned here specifically.
        return -(w.unsqueeze(1) - w.unsqueeze(0)).norm(dim=-1)
