dz_type = "CityscapesDataset"
dz_root = "/data/uda_datasets/dark_zurich"
dz_crop_size = (512, 512)
dz_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="DZRandomCrop", crop_size=dz_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
dz_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_dz = dict(
    type=dz_type,
    data_root=dz_root,
    data_prefix=dict(
        img_path="rgb_anon/train/night",
    ),
    img_suffix="_rgb_anon.png",
    pipeline=dz_train_pipeline,
)
val_dz = dict(
    type=dz_type,
    data_root=dz_root,
    data_prefix=dict(
        img_path="rgb_anon/val/night",
        seg_map_path="gt/val/night",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=dz_test_pipeline,
)
