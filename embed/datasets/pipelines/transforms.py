import numpy as np
from embed.datasets import PIPELINES

@PIPELINES.register_module()
class AddCenter(object):
    """Add center.

    Args:
        center_type (str): The center type to calculate the center.
    """

    def __init__(self, center_type='bbox'):
        self.center_type = center_type

    def __call__(self, results):
        """Call function to calculate the center.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with center calculated.
        """

        if self.center_type == 'bbox':
            gt_bboxes = results['gt_bboxes']
            results['gt_centers'] = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5
        elif self.center_type == 'mask':
            gt_centers = list(map(lambda gt_mask:
                                  np.transpose(np.nonzero(gt_mask))
                                  .mean(axis=0),
                                  results['gt_masks']))

            '''
            for gt_bbox, gt_mask in zip(results['gt_bboxes'], results['gt_masks']):
                if np.transpose(np.nonzero(gt_mask)).shape[0] == 0:
                    print('np.transpose(np.nonzero(gt_mask)).shape[0] = 0')
                    print('np.transpose(np.nonzero(gt_mask)).mean(axis=0): ')
                    print(np.transpose(np.nonzero(gt_mask)).mean(axis=0))
                    print('gt_bbox: ')
                    print(gt_bbox)
                if gt_mask.sum() == 0:
                    print('gt_mask.sum() = 0')
                    print('gt_bbox: ')
                    print(gt_bbox)
            '''

            results['gt_centers'] = np.stack(gt_centers, axis=0)[:, [1, 0]]
        else:
            raise AssertionError(
                f'{self.center_type} is not supported')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(mask_type={self.mask_type})'


@PIPELINES.register_module()
class AddArea(object):
    """Add area.

    Args:
        area_type (str): The area type to calculate the area.
    """

    def __init__(self, area_type='bbox'):
        self.area_type = area_type

    def __call__(self, results):
        """Call function to calculate the area.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with area calculated.
        """

        if self.area_type == 'bbox':
            gt_bboxes = results['gt_bboxes']
            results['gt_areas'] = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * \
                                  (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        elif self.area_type == 'mask':
            results['gt_areas'] = results['gt_masks'].areas()
        else:
            raise AssertionError(
                '{self.area_type} is not supported')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(mask_type={self.mask_type})'


@PIPELINES.register_module()
class MaskRescale(object):
    """Rescale instance segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """

        for key in results.get('mask_fields', []):
            if self.scale_factor != 1:
                results[key] = results[key].rescale(self.scale_factor,
                                                    interpolation='bilinear')
#                                                    interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'
