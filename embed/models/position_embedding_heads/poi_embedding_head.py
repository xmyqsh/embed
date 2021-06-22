import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from embed.cv.cnn import ConvModule, bias_init_with_prob, normal_init
from embed.cv.runner import auto_fp16
from embed.core.utils import multi_apply
from functools import partial

from embed.models import POSITION_EMBEDDING_HEADS, build_loss

from .base_position_embedding_head import BasePositionEmbeddingHead

@POSITION_EMBEDDING_HEADS.register_module()
class PoiEmbeddingHead(BasePositionEmbeddingHead):
    r"""Position of Interest Embedding Head.

    Args:
        in_channels (int):
        num_classes (int):
        scale_ranges (List[Tuple]):
        thres       (float):
        pos_num     (int):
    """

    def __init__(self,
                 scale_ranges=[],
                 pool_sizes=[3,3,3,5,5],
                 pos_num=7,
                 top_num=100,
                 *args,
                 **kwargs):
        super(PoiEmbeddingHead, self).__init__(*args, **kwargs)
        self.pool_sizes = pool_sizes
        self.scale_ranges = scale_ranges

        self.pos_num = pos_num
        self.top_num = top_num

        # TODO(ljm): register a generator for gassian heatmap generation
        self.min_overlap = 0.7
        self.gaussian_sigma = 3

        self.scale_factor = 1. / 4

    def forward_train(self, x, pred_weights, **kwargs):
        pred_centers = self(x)
        gt_scoremap, gt_instance, gt_class, num_ths = \
            self.get_targets(pred_centers, **kwargs)
        loss_pos_th = self.loss(pred_centers, gt_scoremap, num_ths)
        gt_guided_idx_feat_ths, weighted_values = \
            self.get_gt_guided_positions(pred_centers, pred_weights,
                                         gt_instance, gt_class, num_ths)

        return dict(loss_pos_th=loss_pos_th), \
               dict(gt_guided_idx_feat_ths=gt_guided_idx_feat_ths,
                    weighted_values=weighted_values,
                    gt_instance=gt_instance,
                    gt_class=gt_class,
                    num_ths=num_ths)

    def get_targets(self, pred_centers, input_shape, gt_labels,
                                                     gt_bboxes,
                                                     gt_masks,
                                                     gt_areas,
                                                     gt_centers):
        num_ths = list(map(lambda x: x.shape[0], gt_labels))
        gt_labels, gt_bboxes, gt_masks, gt_areas, gt_centers = \
            tuple(map(partial(torch.cat, dim=0), [gt_labels, gt_bboxes, gt_masks, gt_areas, gt_centers]))

        '''
        gt_masks = F.interpolate(gt_masks.unsqueeze(0).float(), scale_factor = self.scale_factor,
                                                                mode='bilinear',
                                                                align_corners=False).clamp(max=1.0) \
                                                                                    .squeeze(0)
        '''

        return multi_apply(self.get_target_single_level, pred_centers, self.scale_ranges,
                                                         input_shape=input_shape,
                                                         gt_labels=gt_labels,
                                                         gt_bboxes=gt_bboxes,
                                                         gt_masks=gt_masks,
                                                         gt_areas=gt_areas,
                                                         gt_centers=gt_centers,
                                                         num_ths=num_ths)

    def get_target_single_level(self, pred_center, scale_range, input_shape,
                                                                gt_labels,
                                                                gt_bboxes,
                                                                gt_masks,
                                                                gt_areas,
                                                                gt_centers,
                                                                num_ths):
        feat_shape = pred_center.shape[-2:]
        rescale = [feat_shape[-2] / input_shape[-2], feat_shape[-1] / input_shape[-1]]

        gt_assign_mask = ((gt_areas.sqrt() >= scale_range[0]) & (gt_areas.sqrt() <= scale_range[1]))
        centers = gt_centers[gt_assign_mask]
        centers[..., 0] *= rescale[1]
        centers[..., 1] *= rescale[0]
        centers_int = centers.to(torch.int64)
        # TODO(ljm): move this check into a proper place
        #            check out where the cornerNet check this
        centers_int[:, 0].clamp_(min=0, max=feat_shape[1])
        centers_int[:, 1].clamp_(min=0, max=feat_shape[0])

        box_tensor = gt_bboxes[gt_assign_mask]
        wh = torch.zeros_like(centers)
        wh[..., 0] = (box_tensor[..., 2] - box_tensor[..., 0]) * rescale[1]
        wh[..., 1] = (box_tensor[..., 3] - box_tensor[..., 1]) * rescale[0]

        gt_instance = gt_masks[gt_assign_mask].float()
        gt_class = gt_labels[gt_assign_mask]

        num_ths = list(map(lambda x: x.sum().item(), torch.split(gt_assign_mask, num_ths, dim=0)))

        gt_instance, gt_class, centers_int, wh = \
            tuple(map(lambda x: torch.split(x, num_ths, dim=0),
                      [gt_instance, gt_class, centers_int, wh]))

        b = pred_center.shape[0]
        gt_scoremap = torch.zeros(b, self.num_classes,
                                  *feat_shape, device=gt_masks.device)

        list(map(lambda idx: self.get_target_single_image(gt_scoremap[idx],
                                                          gt_class[idx],
                                                          centers_int[idx],
                                                          wh[idx]),
             range(b)))

        return gt_scoremap, gt_instance, gt_class, num_ths

    def get_target_single_image(self, gt_scoremap, gt_class, centers_int, wh):
        PoiEmbeddingHead.generate_score_map(gt_scoremap, gt_class, wh,
                                            centers_int, self.min_overlap,
                                            sigma_factor=self.gaussian_sigma,
                                            device=gt_scoremap.device)

    def loss(self, pred_centers, gt_heatmap, num_ths):
        return list(map(partial(self.loss_single_level, avg_factor=sum(map(sum, num_ths))),
                    pred_centers, gt_heatmap))

    def loss_single_level(self, pred_center, gt_heatmap, avg_factor=1):
        loss_pos = self.loss_pos(pred_center, gt_heatmap,
                                 reduction_override='mean',
                                 avg_factor=avg_factor)
        return loss_pos

    def get_gt_guided_positions(self, pred_centers, pred_weights,
                                      gt_instance, gt_class, num_ths):
        return multi_apply(self.get_gt_guided_position_single_level,
                           pred_centers, pred_weights,
                           gt_instance, gt_class, num_ths)

    def get_gt_guided_position_single_level(self, pred_center, pred_weight,
                                                  gt_instance, gt_class, num_ths):
        if sum(num_ths) == 0:
            b, dim = pred_weight.shape[:2]
            dtype, device = pred_weight.dtype, pred_weight.device
            idx_feat_ths = [torch.empty((0, dim), dtype=dtype, device=device) for _ in range(b)]
            weighted_values = [torch.empty((0), dtype=dtype, device=device) for _ in range(b)]
            return idx_feat_ths, weighted_values
        pred_select = list(map(lambda sub_pred, sub_class: sub_pred[sub_class.to(torch.int64), ...],
                               pred_center, gt_class))
        pred_select, gt_instance = tuple(map(partial(torch.cat, dim=0), [pred_select, gt_instance]))
        guided_inst = F.interpolate(gt_instance.unsqueeze(0), size=pred_center.shape[-2:],
                                    mode='bilinear', align_corners=False).squeeze(0)
        keep = (guided_inst > 0.1) & (guided_inst < 255)
        guidence = torch.zeros_like(guided_inst)
        guidence[keep] = pred_select[keep].sigmoid()

        weighted_values, guided_index = torch.topk(guidence.reshape(guidence.shape[0], -1),
                                                   k=self.pos_num, dim=-1)
        weighted_values, guided_index = \
            tuple(map(lambda x: torch.split(x.reshape(-1),
                                      list(map(lambda num_th: self.pos_num * num_th, num_ths)),
                                      dim=0),
                [weighted_values, guided_index]))
        pred_weight = pred_weight.reshape(*pred_weight.shape[:2], -1).permute(0, 2, 1).contiguous()
        idx_feat_ths = list(map(lambda sub_weight, sub_idx: sub_weight[sub_idx, :],
                                pred_weight, guided_index))
        return idx_feat_ths, weighted_values

    def get_positions(self, pred_centers, pred_weights):
        return multi_apply(self.get_position_single_level, pred_centers,
                                                           pred_weights,
                                                           self.pool_sizes)

    def get_position_single_level(self, pred_center, pred_weight, pool_size):
        pred_center = pred_center.sigmoid()
        center_pool = F.avg_pool2d(pred_center, kernel_size=pool_size,
                                   stride=1, padding=(pool_size-1)//2)
        pred_center = (pred_center + center_pool) / 2.0
        fmap_max = F.max_pool2d(pred_center, 3, stride=1, padding=1)
        keep = ((fmap_max == pred_center) & (pred_center > self.thres)).float()
        pred_center *= keep
        pred_center = pred_center.reshape(*pred_center.shape[:2], -1)
        pred_weight = pred_weight.reshape(*pred_weight.shape[:2], -1).permute(0, 2, 1).contiguous()
        return multi_apply(self.get_position_single_image, pred_center, pred_weight)

    def get_position_single_image(self, pred_center, pred_weight):
        class_th, index_th = torch.nonzero(pred_center, as_tuple=True)
        score_th = pred_center[class_th, index_th]
        keep = torch.argsort(score_th, descending=True)[:self.top_num // 2]
        class_th, score_th, index_th = class_th[keep], score_th[keep], index_th[keep]
        idx_feat_th = pred_weight[index_th, :]
        return idx_feat_th, class_th, score_th, len(index_th)

    # TODO(ljm): register various gaussian heatmap generators
    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap, sigma_factor=6, device=None):
        """
        Generate gaussian-based score map for Things in each stage.
        """
        radius = PoiEmbeddingHead.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            PoiEmbeddingHead.draw_gaussian(fmap[channel_index], centers_int[i], radius[i],
                                           sigma_factor=sigma_factor, device=device)

    @staticmethod
    def get_gaussian_radius(box_tensor, min_overlap):
        """
        Calculate Gaussian radius based on box size.
        This algorithm is copyed from CornerNet.
        box_tensor (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        """
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1, sigma_factor=6, device=None):
        """
        Draw gaussian-based score map.
        """
        diameter = 2 * radius + 1
        gaussian = PoiEmbeddingHead.gaussian2D((radius, radius), sigma=diameter / sigma_factor)
        gaussian = torch.Tensor(gaussian).to(device=device)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap  = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
