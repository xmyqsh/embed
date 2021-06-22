import torch
import torch.nn as nn
import torch.nn.functional as F
from embed.cv.cnn import ConvModule
from embed.cv.runner import auto_fp16

from embed.models import POSITION_DECODING_HEADS

from .base_position_decoding_head import BasePositionDecodingHead

import numpy as np

@POSITION_DECODING_HEADS.register_module()
class StandardInstancePositionDecodingHead(BasePositionDecodingHead):
    r"""Standard Instance Position Decoding Head.

    Args:
        in_channels     (int):
        out_channels    (int):

    """

    def __init__(self,
                 inst_threshold=None,
                 pos_num=7,
                 top_num=100,
                 **kwargs):
        super(StandardInstancePositionDecodingHead, self).__init__(**kwargs)
        self.inst_threshold = inst_threshold
        self.pos_num = pos_num
        self.top_num = top_num

    def forward_train(self, x, gt_guided_idx_feat_ths,
                               weighted_values,
                               gt_instance,
                               gt_class,
                               num_ths):
        num_ths = list(map(lambda num_th: list(map(lambda num: self.pos_num * num, num_th)), num_ths))
        gt_guided_idx_feat_th, weighted_value, gt_instance, num_ths = \
            self.get_targets(num_ths, gt_guided_idx_feat_ths, weighted_values, gt_instance)

        del gt_guided_idx_feat_ths, weighted_values
        gt_guided_sem = self.get_gt_guided_sems(x, num_ths, gt_guided_idx_feat_th)

        '''
        assert torch.isnan(gt_guided_idx_feat_th).sum() == 0, torch.isnan(gt_guided_idx_feat_th)
        assert torch.isnan(weighted_value).sum() == 0, torch.isnan(weighted_value)
        assert torch.isnan(gt_instance).sum() == 0, torch.isnan(gt_instance)
        assert torch.isnan(gt_guided_sem).sum() == 0, torch.isnan(gt_guided_sem)
        '''

        del gt_guided_idx_feat_th
        loss_sem_th = self.loss(gt_guided_sem, gt_instance, weighted_value, avg_factor=sum(num_ths) / self.pos_num)
        return dict(loss_sem_th=loss_sem_th)

    def get_targets(self, *args):
        rets = super(StandardInstancePositionDecodingHead, self).get_targets(*args)
        weighted_value = rets[1]
        weighted_value = weighted_value.reshape(-1, self.pos_num)
        weighted_value = weighted_value / weighted_value.sum(dim=1, keepdim=True).clamp(min=1e-8)
        rets[1] = weighted_value.reshape(-1)
        gt_instance = rets[2]
        gt_instance = gt_instance.unsqueeze(1).expand(-1, self.pos_num, -1, -1) \
                                              .reshape(-1, *gt_instance.shape[-2:])
        rets[2] = gt_instance
        return rets

    def post_process_pre(self, sem, cls, score, nums):
        sem = sem.sigmoid()
        pred_mask = sem > self.inst_threshold
        # object rescore
        sum_masks = pred_mask.sum((1, 2)).float() + 1e-6
        seg_score = (sem * pred_mask.float()).sum((1, 2)) / sum_masks
        score *= seg_score

        keeps = map(lambda s: torch.argsort(s, descending=True)[: min(self.top_num,
                                                                              (s >= 0.05).sum())],
                         score.split(nums, dim=0))
        return sem, cls.to(torch.int32), score, keeps

    def post_process_single_image(self, sem, cls, score, img_meta):
        sem = self.post_process_single_image_rescale(sem, img_meta)
        # TODO(ljm): check the type of pred_mask(Bool) ok or not for format_result and evaluation
        pred_mask = sem > self.inst_thres
        pred_bbox = self.get_bounding_boxes(pred_mask)
#        return pred_bbox, pred_mask, cls, score
        return dict(pred_bbox=pred_bbox, pred_mask=pred_mask, cls=cls, score=score)

    # TODO(ljm): move this func into a proper file
    #####################################
    # TODO(ljm): check the xy or yx
    #####################################
    @staticmethod
    def get_bounding_boxes(pred_mask):
        x_any, y_any = torch.any(pred_mask, dim=1), torch.any(pred_mask, dim=2)
        def get_bounding_box(x, y):
            x, y = torch.where(x)[0], torch.where(y)[0]
            # TODO(ljm): check the type of x[0], scalar or torch.Tensor, assumed scale here
            return [x[0].item(), y[0].item(), x[-1].item(), y[-1].item()] if len(x) and len(y) else [0, 0, 0, 0]
        '''
        x1, y1, x2, y2 = multi_apply(self.get_bounding_box, *zip(x_any, y_any))
        return torch.hstack(multi_apply(torch.Tensor, [x1, y1, x2, y2], dtype=torch.float32, device=pred_mask.device))
        '''
        '''
        x1, y1, x2, y2 = multi_apply(torch.Tensor,
                                     multi_apply(self.get_bounding_box, *zip(x_any, y_any)),
                                     dtype=torch.float32, device=pred_mask.device)
        return torch.stack([x1, y1, x2, y2], dim=1)
        '''
        #bboxes = map(self.get_bounding_box, *zip(x_any, y_any))
        bboxes = list(map(self.get_bounding_box, x_any, y_any))
#        return np.array(bboxes, dtype=np.float32)
        return torch.tensor(bboxes, dtype=torch.float32)
