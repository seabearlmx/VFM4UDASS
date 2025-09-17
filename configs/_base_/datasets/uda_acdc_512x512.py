acdc_type = "CityscapesDataset"
acdc_root = "/data/uda_datasets/acdc/"
acdc_crop_size = (512, 512)
acdc_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(960, 540), keep_ratio=True),
    dict(type="RandomCrop", crop_size=acdc_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
acdc_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(960, 540), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_acdc = dict(
    type=acdc_type,
    data_root=acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/train",
        seg_map_path="gt/train",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    rcs_cfg=dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5),
    pipeline=acdc_train_pipeline,
)
val_acdc = dict(
    type=acdc_type,
    data_root=acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/val",
        seg_map_path="gt/val",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=acdc_test_pipeline,
)
