import torch
import numpy as np
from embed.datasets import PIPELINES
from embed.datasets.pipelines import DefaultFormatBundle, to_tensor
from embed.cv.parallel import DataContainerWithPad as DC

from embed.core import BitmapMasks

@PIPELINES.register_module()
class PanopticFCNFormatBundle(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_areas: (1)to tensor, (2)to DataContainer
    - gt_centers: (1)to tensor, (2)to DataContainer
    - gt_masks: (0)assert isinstance(gt_masks, BitmapMasks)
                (1)to tensor, (2)to DataContainer
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                    'gt_areas', 'gt_centers']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            assert isinstance(results['gt_masks'], BitmapMasks)
#            results['gt_masks'] = DC(results['gt_masks'].to_tensor(dtype=torch.uint8, device=torch.device('cpu')), pad=True)
            results['gt_masks'] = DC(results['gt_masks'].to_tensor(dtype=torch.float32, device='cpu'), pad=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results
