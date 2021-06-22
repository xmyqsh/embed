# model settings
model = dict(
    type='PanopticFCN',
    pretrained='torchvision://resnet50',
    pos_num=7,
    top_num=100,
    ignore_val=255,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_input',
        relu_before_extra_convs=True,
        num_outs=6), # p2, p3, p4, p5, p6, p7
    feature_encoding=dict(
        type='FusedSemantic',
        in_channels=256,
        out_channels=256,
        num_ins=4, # p2, p3, p4, p5
        start_level=0,
        fusion_level=0,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        feature_encoder=dict(
            # TODO(ljm): add with_last_relu and finefune other hyper params
            type='CoordNConv',
            name='feature_encoder',
            in_channels=256,
            out_channels=64,
            num_convs=3)),
#            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))),
    kernel_encoding=dict(
        type='StandardKernelEncoder',
        start_level=1,
        num_ins=5, # p3, p4, p5, p6, p7
        weight_encoder=dict(
            type='CoordNConv',
            name='weight_encoder',
            in_channels=256,
            out_channels=256,
            num_convs=3,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        in_channels=256,
        out_channels=256),
    position_embedding_head=dict(
        type='CombinedPositionEmbeddingHead',
        start_level=1,
        num_ins=5, # p3, p4, p5, p6, p7
        position_encoder=dict(
            type='CoordNConv',
            name='position_encoder',
            use_coord=False,
            in_channels=256,
            out_channels=256,
            num_convs=3,
            conv_cfg=dict(type='DCNv2', deform_groups=1),
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        thing_head=dict(
            type='PoiEmbeddingHead',
            in_channels=256,
            num_classes=80,
            scale_ranges=[(1, 64), (32, 128), (64, 256), (128, 512), (256, 2048),],
            pool_sizes=[3,3,3,5,5],
            thres=0.05,
            loss_pos=dict(
                # TODO(ljm): Try GaussianFocalLoss
                # type='GaussianFocalLoss',
                type='FocalLossV2',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0)),
        stuff_head=dict(
            type='AvgEmbeddingHead',
            in_channels=256,
            num_classes=54,
            thres=0.05,
            loss_pos=dict(
                type='FocalLossV2',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0))),
    position_decoding_head=dict(
        type='CombinedPositionDecodingHead',
        thing_head=dict(
            type='StandardInstancePositionDecodingHead',
            in_channels=256,
            out_channels=64,
            inst_threshold=0.4,
            sem_generator=dict(
                type='SimBasedGenerator',
                sim_type='cos',
                sim_thres=0.9,
                cls_spec=True),
            loss_sem=dict(
                type='BinaryDiceLoss',
                loss_weight=3.0)),
        stuff_head=dict(
            type='StandardSemanticPositionDecodingHead',
            in_channels=256,
            out_channels=64,
            sem_generator=dict(type='ClsBasedGenerator'),
            loss_sem=dict(
                type='BinaryDiceLoss',
                loss_weight=3.0))),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
)
