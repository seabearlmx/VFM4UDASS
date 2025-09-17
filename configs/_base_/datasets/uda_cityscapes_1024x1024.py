cityscapes_type = "UDACityscapesDataset"
cityscapes_root = "/data/uda_datasets/Cityscapes/"
cityscapes_crop_size = (1024, 1024)
cityscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(2048, 1024)),
    dict(type="RandomCrop", crop_size=cityscapes_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
cityscapes_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/train",
        seg_map_path="gtFine/train",
    ),
    rcs_cfg=dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5),
    pipeline=cityscapes_train_pipeline,
)
val_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/val",
        seg_map_path="gtFine/val",
    ),
    pipeline=cityscapes_test_pipeline,
)
