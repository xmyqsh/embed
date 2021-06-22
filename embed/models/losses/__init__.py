from mmdet.models.losses import \
    accuracy, Accuracy, cross_entropy, binary_cross_entropy, \
    mask_cross_entropy, CrossEntropyLoss, sigmoid_focal_loss, \
    FocalLoss, smooth_l1_loss, SmoothL1Loss, balanced_l1_loss, \
    BalancedL1Loss, mse_loss, MSELoss, iou_loss, bounded_iou_loss, \
    IoULoss, BoundedIoULoss, GIoULoss, DIoULoss, CIoULoss, GHMC, \
    GHMR, reduce_loss, weight_reduce_loss, weighted_loss, L1Loss, \
    l1_loss, isr_p, carl_loss, AssociativeEmbeddingLoss, \
    GaussianFocalLoss, QualityFocalLoss, DistributionFocalLoss, \
    VarifocalLoss

mms = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'sigmoid_focal_loss',
    'FocalLoss', 'smooth_l1_loss', 'SmoothL1Loss', 'balanced_l1_loss',
    'BalancedL1Loss', 'mse_loss', 'MSELoss', 'iou_loss', 'bounded_iou_loss',
    'IoULoss', 'BoundedIoULoss', 'GIoULoss', 'DIoULoss', 'CIoULoss', 'GHMC',
    'GHMR', 'reduce_loss', 'weight_reduce_loss', 'weighted_loss', 'L1Loss',
    'l1_loss', 'isr_p', 'carl_loss', 'AssociativeEmbeddingLoss',
    'GaussianFocalLoss', 'QualityFocalLoss', 'DistributionFocalLoss',
    'VarifocalLoss'
]

from .binary_dice_loss import BinaryDiceLoss, binary_dice_loss
from .focal_loss import FocalLossV2, sigmoid_focal_loss_v2

__all__ = mms + [
    'BinaryDiceLoss', 'binary_dice_loss',
    'FocalLossV2', 'sigmoid_focal_loss_v2'
]
