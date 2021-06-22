from torch import nn
from embed.models import LOSSES
from embed.models.losses import weighted_loss


@weighted_loss
def binary_dice_loss(pred, target):
    """`Dice Loss <https://arxiv.org/abs/xxxx.xxxxx>`_ for targets in compact
    0/1 format.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The target.
    """
    pred = pred.reshape(pred.shape[0], -1).sigmoid()
    target = target.reshape(target.shape[0], -1)
    loss_part = (pred ** 2).sum(dim=1) + (target ** 2).sum(dim=1)
    loss = 1 - 2 * (target * pred).sum(dim=1) / loss_part
    return loss


@LOSSES.register_module()
class BinaryDiceLoss(nn.Module):
    """BinaryDiceLoss is a variant of dice loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/xxxx.xxxxx>`_
    Code is modified from `loss.py
    <https://github.com/princeton-vl/PanopticFCN/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    where the targets is in the compact 0/1 format.

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(BinaryDiceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * binary_dice_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss
