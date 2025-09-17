# model settings
crop_size = (512, 512)
num_classes = 19
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)
checkpoint_file = "checkpoints/sam_vit_h_converted_512x512.pth"
model = dict(
    type="HRDAEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="SAMViT",
        img_size=512,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        global_attn_indexes=[7, 15, 23, 31],
        out_indices=[7, 15, 23, 31],
        window_size=14,
        use_rel_pos=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=checkpoint_file,
        ),
    ),
    decode_head=dict(
        type="HRDAHead",
        # Use the DAFormer decoder for each scale.
        single_scale_head='DAFormerHead',
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=0.1,
        # in_channels=[64, 128, 320, 512],
        in_channels=[1280, 1280, 1280, 1280],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg),
        ),
        loss_ce=dict(
            type='WeightedCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    # Use the full resolution for the detail crop and half the resolution for
    # the context crop.
    scales=[1, 0.5],
    # Use a relative crop size of 0.5 (=512/1024) for the detail crop.
    hr_crop_size=[512, 512],
    # Use LR features for the Feature Distance as in the original DAFormer.
    feature_scale=0.5,
    # Make the crop coordinates divisible by 8 (output stride = 4,
    # downscale factor = 2) to ensure alignment during fusion.
    crop_coord_divisible=8,
    # Use overlapping slide inference for detail crops for pseudo-labels.
    hr_slide_inference=True,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]),
)
