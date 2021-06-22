import torch
from torch import nn
from embed.cv.cnn import normal_init
from embed.core.utils import multi_apply
from functools import partial

from embed.models import build_position_decoding_generator, build_loss

class BasePositionDecodingHead(nn.Module):
    r"""Base Position Decoding Head.
    """
    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 sem_generator=None,
                 loss_sem=None,
                 **kwargs):
        super(BasePositionDecodingHead, self).__init__()
        #self.embed_extractor = nn.Linear(in_channels, out_channels)
        self.embed_extractor = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # TODO(ljm): registe generator and add builder of it
        self.sem_generator = build_position_decoding_generator(sem_generator)
        # TODO(ljm): regist loss decode and add builder of it
        self.loss_sem = build_loss(loss_sem)

    def init_weights(self):
        '''
        nn.init.normal_(self.embed_extractor.weight, 0, 0.01)
        nn.init.constant_(self.embed_extractor.bias, 0)
        '''
        normal_init(self.embed_extractor, std=0.01, bias=0)

    def forward(self, x):
        return self.embed_extractor(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def forward_train(self, *args, **kwargs):
        pass

    def simple_test(self, x, input_shape, img_metas, **kwargs):
        sems, classes, scores = self.get_sems(x, **kwargs)
        return self.post_process(sems, classes, scores, input_shape, img_metas)

    def post_process(self, sems, classes, scores, input_shape, img_metas):
        # sem = torch.cat(sems, dim=0)
        nums = list(map(lambda x: x.shape[0], classes))
        sem, cls, score = tuple(map(partial(torch.cat, dim=0), [sems, classes, scores]))
        sem, cls, score, keeps = self.post_process_pre(sem, cls, score, nums)
        sem = F.interpolate(sem, size=input_shape[-2:], mode='bilinear', align_corners=False)
        # sems = torch.split(sem, [cls.shape[0] for cls in classes], dim=0)
        sems, classes, scores = tuple(map(lambda x: torch.split(x, nums, dim=0),
                                            [sem, cls, score]))
        sems, classes, scores = tuple(map(list,
                                          zip(*list(map(lambda sem, cls, score, keep: (sem[keep], cls[keep], score[keep]),
                                               sems, classes, scores, keeps)))))

        return list(map(self.post_process_single_image, sems, classes, scores, img_metas))

    def post_process_single_image_rescale(self, sem, img_meta):
        img_shape, ori_shape = img_meta['img_shape'], img_meta['ori_shape']
        sem = F.interpolate(sem[:, img_shape[0], img_shape[1]].unsqueeze(0),
                            size=ori_shape[:2], mode='bilinear', align_corners=False).squeeze(0)
        return sem

    def loss(self, x, gt, *args, **kwargs):
        return self.loss_sem(x, gt, *args, **kwargs)

    def get_gt_guided_sems(self, x, nums, gt_guided_idx_feat):
        gt_guided_idx_feats = self.get_sems_pre(gt_guided_idx_feat, nums)
        b, max_num = len(gt_guided_idx_feats), max(nums)
        dim, device = gt_guided_idx_feats[0].shape[-1], gt_guided_idx_feat.device
        # TODO(ljm): check the args of torch.zeros
        gt_guided_idx_feat = torch.zeros((b, max_num, dim), device=device)

        '''
        map(lambda idx_b, num: gt_guided_idx_feat[idx_b, :num] = gt_guided_idx_feats[idx_b],
            *enumerate(nums))
        '''

        for idx_b, num in enumerate(nums):
            gt_guided_idx_feat[idx_b, :num] = gt_guided_idx_feats[idx_b]

        gt_guided_sem = self.sem_generator(x, meta_weight=gt_guided_idx_feat, fusion=False)
        gt_guided_sems = list(map(lambda sem, num: sem[:num, ...], gt_guided_sem, nums))
        return torch.cat(gt_guided_sems, dim=0)

    def get_targets(self, nums, *args):
        rets = self.image_level2image(nums, *args)
        nums = rets.pop()
        rets = list(map(partial(torch.cat, dim=0), rets))
        rets.append(nums)
        return rets

    @staticmethod
    def image_level2image(nums, *args):
        rets = \
            list(map(lambda img_level: list(map(partial(torch.cat, dim=0),
                                                      list(map(list, zip(*img_level))))),
                     args))
        nums = list(map(sum, zip(*nums)))
        rets.append(nums)
        return rets

    def get_sems_pre(self, idx_feat, nums):
        idx_feat = self(idx_feat)
        return torch.split(idx_feat, nums, dim=0)

    def get_sems(self, x, idx_feats, classes, scores, nums):
        idx_feats, classes, scores, nums = self.image_level2image(nums, idx_feats, classes, scores)
        idx_feat = torch.cat(idx_feats, dim=0)
        idx_feats = self.get_sems_pre(idx_feat, nums)
        sems, classes, scores = \
            multi_apply(lambda f, idx_feat, cls, score: self.sem_generator(f, meta_weight=idx_feat,
                                                                              cls=cls,
                                                                              score=score,
                                                                              fusion=True),
                        x, idx_feats, classes, scores)
        return sems, classes, scores
