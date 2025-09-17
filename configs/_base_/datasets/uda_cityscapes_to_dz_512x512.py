_base_ = [
    "./uda_cityscapes_512x512.py",
    "./uda_dz_512x512.py",
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="UDADataset",
        source={{_base_.train_cityscapes}},
        target={{_base_.train_dz}},
    ),
    val={{_base_.val_dz}},
    test={{_base_.val_dz}},
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=data['train']
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_dz}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["dark_zurich"]
)
test_evaluator=val_evaluator
