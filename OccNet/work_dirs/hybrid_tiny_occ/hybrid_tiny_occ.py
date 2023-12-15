point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='LoadOccupancyGT'),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='CustomDefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='CustomCollect3D',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'occ_gts'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file=
        'data/occ_gt_release_v1_0/nuscenes_infos_temporal_train_occ_gt.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(type='LoadOccupancyGT'),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='CustomDefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='CustomCollect3D',
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'occ_gts'])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        use_occ_gts=True,
        use_valid_flag=True,
        bev_size=(200, 200),
        queue_length=3),
    val=dict(
        type='CustomNuScenesDataset',
        ann_file=
        'data/occ_gt_release_v1_0/nuscenes_infos_temporal_val_occ_gt.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='CustomCollect3D', keys=['img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='data/nuscenes/',
        bev_size=(200, 200),
        samples_per_gpu=1),
    test=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file=
        'data/occ_gt_release_v1_0/nuscenes_infos_temporal_val_occ_gt.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1600, 900),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='CustomCollect3D', keys=['img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        bev_size=(200, 200)),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1600, 900),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(type='CustomCollect3D', keys=['img'])
            ])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/hybrid_tiny_occ'
load_from = None
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.2, 0.2, 8]
occupancy_size = [0.5, 0.5, 0.5]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
_dim_ = 128
_occupancy_dim_ = 128
_pos_dim_ = 64
_ffn_dim_ = 256
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 200
bev_z_ = 16
queue_length = 3
use_occ_gts = True
only_occ = True
only_det = False
voxel_encoder1_dim = 64
voxel_encoder2_dim = 64
voxel_encoder3_dim = 32
voxel_encoder4_dim = 32
bev_z1 = 2
bev_z2 = 4
bev_z3 = 8
bev_z4 = 16
_pos_dim_1 = 32
_pos_dim_2 = 32
_pos_dim_3 = 16
_pos_dim_4 = 16
last_voxel_dims = 32
decoder_on_bev = True
box_query_dims = 128
model = dict(
    type='HybridFormer',
    use_grid_mask=True,
    video_test_mode=True,
    use_occ_gts=True,
    only_occ=True,
    only_det=False,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='HybridFormerOccupancyHead',
        bev_h=200,
        bev_w=200,
        bev_z=16,
        num_query=900,
        num_classes=10,
        in_channels=128,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        point_cloud_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        occupancy_size=[0.5, 0.5, 0.5],
        occ_dims=128,
        occupancy_classes=16,
        only_occ=True,
        only_det=False,
        last_voxel_dims=32,
        box_query_dims=128,
        transformer=dict(
            type='HybridPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=128,
            decoder_on_bev=True,
            encoder_embed_dims=[128, 64, 64, 32, 32],
            feature_map_z=[1, 2, 4, 8, 16],
            pos_dims=[64, 32, 32, 16, 16],
            position=dict(
                bev=dict(
                    type='LearnedPositionalEncoding',
                    num_feats=64,
                    row_num_embed=200,
                    col_num_embed=200),
                voxel=dict(
                    type='VoxelLearnedPositionalEncoding',
                    num_feats=32,
                    row_num_embed=200,
                    col_num_embed=200,
                    z_num_embed=2)),
            encoder=dict(
                bev=dict(
                    type='BEVFormerEncoder',
                    num_layers=1,
                    pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BEVFormerLayer',
                        attn_cfgs=[
                            dict(
                                type='TemporalSelfAttention',
                                embed_dims=128,
                                num_points=4,
                                num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=128,
                                    num_points=8,
                                    num_levels=1),
                                embed_dims=128)
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=128,
                            feedforward_channels=256,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True)),
                        feedforward_channels=256,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm'))),
                voxel=dict(
                    type='VoxelFormerEncoder',
                    num_layers=1,
                    pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
                    num_points_in_voxel=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='VoxelFormerLayer',
                        attn_cfgs=[
                            dict(
                                type='VoxelTemporalSelfAttention',
                                embed_dims=64,
                                num_points=4,
                                num_levels=1),
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=64,
                                    num_points=8,
                                    num_levels=1),
                                embed_dims=64)
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=64,
                            feedforward_channels=128,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True)),
                        feedforward_channels=128,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm'))))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=64,
            row_num_embed=200,
            col_num_embed=200),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_occupancy=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]))))
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
find_unused_parameters = False
gpu_ids = range(0, 1)
