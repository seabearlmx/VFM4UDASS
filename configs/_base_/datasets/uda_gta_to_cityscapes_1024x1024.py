_base_ = [
    "./uda_gta_1024x1024.py",
    "./uda_cityscapes_1024x1024.py",
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="UDADataset",
        source={{_base_.train_gta}},
        target={{_base_.train_cityscapes}},
    ),
    val={{_base_.val_cityscapes}},
    test={{_base_.val_cityscapes}},
)

train_dataloader = dict(
    batch_size=2,
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
            {{_base_.val_cityscapes}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["Citys"] 
)
test_evaluator=val_evaluator
