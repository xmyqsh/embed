import torch
import torch.nn as nn
import torch.nn.functional as F
from embed.cv.cnn import ConvModule, bias_init_with_prob, normal_init
from embed.cv.runner import auto_fp16
from embed.core.utils import multi_apply

from embed.models import POSITION_EMBEDDING_HEADS

from .base_position_embedding_head import BasePositionEmbeddingHead

import numpy as np

@POSITION_EMBEDDING_HEADS.register_module()
class AvgEmbeddingHead(BasePositionEmbeddingHead):
    r"""Average Embedding Head.

    Args:
        in_channels (int):
        num_classes (int):
        scale_ranges (List[Tuple]):
        thres       (float):
    """

    def __init__(self,
                 ignore_val=255,
                 *args,
                 **kwargs):
        super(AvgEmbeddingHead, self).__init__(*args, **kwargs)
        self.ignore_val = ignore_val

        # TODO(ljm) add this param into a proper place
        self.scale_factor = 1. / 4

    def forward_train(self, x, pred_weights, gt_semantic_seg):
        pred_regions = self(x)
        gt_scoremap, gt_sem_label, gt_sem_mask, gt_sem_class, num_sts = \
            self.get_targets(pred_regions, gt_semantic_seg)
        loss_pos_st = self.loss(pred_regions, gt_scoremap)
        gt_guided_idx_feat_sts = self.get_gt_guided_positions(pred_weights,
                                                              gt_sem_mask, num_sts)
        return dict(loss_pos_st=loss_pos_st), \
               dict(gt_guided_idx_feat_sts=gt_guided_idx_feat_sts,
                    gt_sem_label=gt_sem_label,
                    gt_sem_class=gt_sem_class,
                    num_sts=num_sts)

    def get_targets(self, pred_regions, gt_semantic_seg):
        gt_semantic_seg[gt_semantic_seg == self.ignore_val] = self.num_classes
        gt_semantic_seg = F.one_hot(gt_semantic_seg.squeeze(1).long(),
                                        num_classes=self.num_classes + 1)[..., :-1]
        gt_semantic_seg = gt_semantic_seg.permute(0, 3, 1, 2).float().contiguous()

        '''
        gt_semantic_seg = F.interpolate(gt_semantic_seg, scale_factor=self.scale_factor,
                                                         mode='bilinear',
                                                         align_corners=False).clamp(max=1.0)
        '''

        return multi_apply(self.get_target_single_level, pred_regions,
                                                         gt_semantic_seg=gt_semantic_seg)

    def get_target_single_level(self, pred_region, gt_semantic_seg):
        gt_scoremap = F.interpolate(gt_semantic_seg, size=pred_region.shape[-2:],
                                                     mode='bilinear',
                                                     align_corners=False).clamp(max=1.0)
        gt_scoremap[gt_scoremap < 0.5] = 0.0
        gt_assign_mask = gt_scoremap.reshape(*gt_scoremap.shape[:-2], -1).sum(dim=-1) > 0
        gt_sem_label, gt_sem_mask, gt_sem_class, num_sts = \
            multi_apply(self.get_target_single_image, gt_semantic_seg, gt_scoremap, gt_assign_mask)

        return gt_scoremap, gt_sem_label, gt_sem_mask, gt_sem_class, num_sts

    def get_target_single_image(self, gt_semantic_seg, gt_scoremap, gt_assign_mask):
        gt_sem_class = torch.nonzero(gt_assign_mask, as_tuple=False).squeeze(-1)
        num_sem = gt_assign_mask.sum().item()
        gt_sem_label = gt_semantic_seg[gt_assign_mask]
        gt_sem_mask = gt_scoremap[gt_assign_mask].bool().float()
        return gt_sem_label, gt_sem_mask, gt_sem_class, num_sem

    def loss(self, pred_regions, gt_scoremap):
        return list(map(self.loss_single_level, pred_regions, gt_scoremap))

    def loss_single_level(self, pred_region, gt_scoremap):
        b, c = pred_region.shape[:2]
        loss_pos = self.loss_pos(pred_region, gt_scoremap, reduction_override='none')
        loss_pos = loss_pos.reshape(b, c, -1).mean(dim=-1)
        loss_pos = loss_pos.sum() / b
        return loss_pos

    def get_gt_guided_positions(self, pred_weights, gt_sem_mask, num_sts):
        return list(map(self.get_gt_guided_position_single_level, pred_weights,
                                                                  gt_sem_mask,
                                                                  num_sts))

    def get_gt_guided_position_single_level(self, pred_weight, gt_sem_mask, num_sts):
        idx_feat_sts = list(map(lambda a, b: a.unsqueeze(0) * b.unsqueeze(1),
                                pred_weight, gt_sem_mask))
        idx_feat_st = torch.cat(idx_feat_sts, dim=0)
        idx_feat_st = F.adaptive_avg_pool2d(idx_feat_st, output_size=1).squeeze(-1).squeeze(-1)
        return torch.split(idx_feat_st, num_sts, dim=0)

    def get_positions(self, pred_regions, pred_weights):
        return multi_apply(self.get_position_single_level, pred_regions, pred_weights)

    def get_position_single_level(self, pred_region, pred_weight):
        pred_region = pred_region.sigmoid()
        pred_cate = pred_region.argmax(dim=1)

        pred_st_mask = F.one_hot(pred_cate, num_classes=self.num_classes)
        pred_st_mask = pred_st_mask.permute(0, 3, 1, 2).contiguous()

        score_st = (pred_region * pred_st_mask).reshape(*pred_region.shape[:2], -1)

        idx_feat_sts, class_sts, score_sts, num_sts = \
            multi_apply(self.get_position_single_image, pred_cate,
                                                        pred_st_mask,
                                                        score_st,
                                                        pred_weight)

        idx_feat_st = torch.cat(idx_feat_sts, dim=0)
        idx_feat_st = F.adaptive_avg_pool2d(idx_feat_st, output_size=1).squeeze(-1).squeeze(-1)
        idx_feat_sts = torch.split(idx_feat_st, num_sts, dim=0)

        return idx_feat_sts, class_sts, score_sts, num_sts

    def get_position_single_image(self, pred_cate, pred_st_mask, score_st, pred_weight):
        class_st, num_class_st = torch.unique(pred_cate, return_counts=True)
        score_st = (score_st[class_st].sum(dim=-1) / num_class_st)
        pred_st_mask = pred_st_mask[class_st]
        keep = score_st > self.thres

        class_st, score_st, pred_st_mask = class_st[keep], score_st[keep], pred_st_mask[keep]
        num_st = keep.sum()

        idx_feat_st = pred_st_mask.unsqueeze(1) * pred_weight.unsqueeze(0)
        return idx_feat_st, class_st, score_st, num_st
