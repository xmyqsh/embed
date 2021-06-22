import torch
from torch import nn

class BasePositionDecodingGenerator(nn.Module):
    def __init__(self):
        super(BasePositionDecodingGenerator, self).__init__()

    def __call__(self, x, fusion=False, **kwargs):
        return eval('self.forward_{}'.format('test' if fusion else 'train'))(x, **kwargs)

    def forward_train(self, x, meta_weight):
        sem = self.sem_generate(x, meta_weight)
        return sem

    def forward_test(self, x, **kwargs):
        # NOTE: self.training is not alway True here
        meta_weight, cls, score = self.kernel_fusion(**kwargs)
        sem = self.sem_generate(x, meta_weight)
        return sem, cls, score

    def sem_generate(self, x, meta_weight):
        h, w = x.shape[-2:]
        x = x.reshape(*x.shape[:-2], -1)
        sem = torch.matmul(meta_weight, x).reshape(*meta_weight.shape[:-1], h, w)
        return sem

    def kernel_fusion(self, meta_weight, cls, score, **kwargs):
        pass
