from embed.models import DETECTORS, build_backbone, build_neck, \
    build_encoder, build_position_embedding_head, build_position_decoding_head
from embed.models.detectors import BaseDetector

@DETECTORS.register_module()
class PanopticFCN(BaseDetector):
    """Implementation of the PanopticFCN Network."""

    def __init__(self,
                 backbone,
                 neck,
                 feature_encoding,
                 kernel_encoding,
                 position_embedding_head,
                 position_decoding_head,
                 pretrained=None,
                 **kwargs):
        super(PanopticFCN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        self.feature_encoding = build_encoder(feature_encoding)
        self.kernel_encoding = build_encoder(kernel_encoding)
        position_embedding_head.update(**kwargs)
        position_decoding_head.update(**kwargs)
        self.position_embedding_head = build_position_embedding_head(position_embedding_head)
        self.position_decoding_head = build_position_decoding_head(position_decoding_head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(PanopticFCN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            self.neck.init_weights()
        self.feature_encoding.init_weights()
        self.kernel_encoding.init_weights()
        self.position_embedding_head.init_weights()
        self.position_decoding_head.init_weights()

    def extract_feat(self, img):
        """Extract features.

        Args:
            img (torch.Tensor): Image tensor with shape (n, c, h ,w).

        Returns:
            list[torch.Tensor]: Multi-level features that may have
                different resolutions.
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward_dummy(self, img, img_metas, **kwargs):
        pass

    '''
    def forward_dummy(self, img):
        """Dummy forward function."""
        x = self.extract_feat(img)
        features = self.extract_feat(img)
        outs = ()
        encode_feat = self.feature_encoding(features)
        position_embedded_outs = self.position_embedding_head.forward_dummy(features)
        outs = outs + (position_embedded_outs)
        positon_decoding_outs = self.position_decoding_head.forward_dummy(encode_feat, position_embedded_outs)
        outs = outs + (position_decoding_outs)
        return outs
    '''

    '''
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_areas=None,
                      gt_centers=None,
                      gt_semantic_seg=None):
    '''
    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # TODO(ljm): testing pad_shape here

        input_shape = img.shape[-2:]

        features = self.extract_feat(img)
        encode_feat = self.feature_encoding.forward_train(features)
        pred_weights = self.kernel_encoding.forward_train(features)

        losses = dict()

        losses_position_embedding, position_embedded = \
            self.position_embedding_head.forward_train(features,
                                                       pred_weights,
                                                       input_shape,
                                                       **kwargs)
        '''
                                                       gt_bboxes,
                                                       gt_labels,
                                                       gt_masks,
                                                       gt_areas,
                                                       gt_centers,
                                                       gt_semantic_seg)
        '''

        losses_position_decoding = \
            self.position_decoding_head.forward_train(encode_feat, position_embedded)

        losses.update(losses_position_embedding)
        losses.update(losses_position_decoding)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        input_shape = img.shape[-2:]
        features = self.extract_feat(img)
        encode_feat = self.feature_encoding.simple_test(features)
        pred_weights = self.kernel_encoding.simple_test(features)
        position_embedded = self.position_embedding_head.simple_test(features, pred_weights)
        return self.position_decoding_head.simple_test(encode_feat, input_shape, position_embedded, img_metas)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        assert False, 'not supported currently'
