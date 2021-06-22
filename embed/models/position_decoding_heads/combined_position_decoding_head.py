import torch.nn as nn
import torch.nn.functional as F
from embed.cv.cnn import ConvModule
from embed.cv.runner import auto_fp16

from embed.models import POSITION_DECODING_HEADS, build_position_decoding_head

@POSITION_DECODING_HEADS.register_module()
class CombinedPositionDecodingHead(nn.Module):
    r"""Standard Position Decoding Head.

    Args:
        thing_head (dict):
        stuff_head (dict):
    """

    def __init__(self,
                 thing_head,
                 stuff_head,
                 **kwargs):
        super(CombinedPositionDecodingHead, self).__init__()
        thing_head.update(**kwargs)
        stuff_head.update(**kwargs)
        self.thing_head = build_position_decoding_head(thing_head)
        self.stuff_head = build_position_decoding_head(stuff_head)

    def init_weights(self):
        self.thing_head.init_weights()
        self.stuff_head.init_weights()

    def forward_train(self, x, position_embedded):
        losses = dict()
        losses_thing = self.thing_head.forward_train(x, **position_embedded['positions_th'])
        losses_stuff = self.stuff_head.forward_train(x, **position_embedded['positions_st'])
        losses.update(losses_thing)
        losses.update(losses_stuff)
        return losses

    def simple_test(self, x, input_shape, position_embedded, img_metas):
        results_th = self.thing_head.simple_test(x, input_shape, img_metas, **position_embedded['positions_th'])
        results_st = self.stuff_head.simple_test(x, input_shape, img_metas, **position_embedded['positions_st'])
        #results_pn = map(self.combine_thing_and_stuff, *zip(results_th, results_st))
        results_pn = map(self.combine_thing_and_stuff, results_th, results_st)

        # TODO(ljm): The operation of `.detach().cpu().numpy()` should be implemented in the engine
        #            yeah, in the after_val_iter called to_numpy callback
        def to_numpy(result, key):
            list(map(lambda k: to_numpy(result[key], k), result[key].keys())) if isinstance(result[key], dict) else \
            result[key].detach().cpu().numpy() if isinstance(result[key], torch.Tensor) else result[key]

        def to_numpy(result):
            list(map(lambda k: to_numpy(result, k), result)) if isinstance(result, dict) else result

        return list(map(lambda result_th, result_st, result_pn: dict(instance=to_numpy(result_th),
                                                                sem_seg=to_numpy(result_st),
                                                                panoptic=to_numpy(result_pn)),
                    results_th, results_st, results_pn))

    # TODO(ljm): check this part again
    def combine_thing_and_stuff(self, results_th, results_st, force_resort=True):
        # use contigous id as the medium map between prediction and gt dataset
        # TODO(list):
        #            combine     easy                                        Done
        #            evaluator   copy                                        Done
        #            generator   copy
        #            dataset     contiguous id mapper; result formater       Done
        #
        #            mIoU evaluator: not touched                             Done
        #            Dice loss   design
        panoptic_seg = torch.zeros_like(pred_sem, dtype=torch.int32)
        current_segment_id = 0
        segments_info = []

        pred_mask, cls, score = results_th['pred_mask'], results_th['cls'], results_th['score']
        indexes = torch.argsort(score, descending=True) if force_resort else [_ for _ in range(len(score))]
        indexes = indexes[:(score >= self.inst_threshold).sum()]
        for _idx, index in enumerate(indexes):
            _mask, _cls, _score = pred_mask[index], cls[index], score[index]
            mask_area = _mask.sum().item()
            intersect = _mask & (panoptic_seg > 0)
            intersect_area = intersect.sum().item()
            if mask_area == 0 or intersect_area * 1.0 / mask_area > self.overlap_threshold:
                continue
            if intersect_area > 0:
                _mask = _mask & (panoptic_seg == 0)
            current_segment_id += 1
            panoptic_seg[_mask] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": True,
                    "score": _score.item(),
                    "category_id": _cls.item(),
                    "instance_id": _idx,
                }
            )

        pred_sem = results_st['pred_sem']
        cls = results_st['cls']
        # cls = torch.unique(pred_sem)
        for _cls in cls:
            if _cls == 0:
                continue
            mask = (pred_sem == _cls) & (panoptic_seg == 0)
            mask_area = mask.sum()
            if mask_area < self.stuff_area_limit:
                continue
            current_segment_id += 1
            panoptic_seg[_mask] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": False,
                    "category_id": _cls.item(),
                    "area": mask_area.item(),
                }
            )

        return dict(panoptic_seg=panoptic_seg, segments_info=segments_info)
