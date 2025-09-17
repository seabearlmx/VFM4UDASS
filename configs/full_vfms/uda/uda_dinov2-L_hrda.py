# dataset config
_base_ = [
    "../../_base_/datasets/uda_gta_to_cityscapes_1024x1024.py",
    "../../_base_/default_runtime.py",
    "../../_base_/models/dinov2_hrda.py"
]

model_wrapper_cfg=dict(
        type='UDAMMDistributedDataParallel', find_unused_parameters=True)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys=dict(
            norm=dict(decay_mult=0.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0),
            backbone=dict(lr_mult=0.1)),
        norm_decay_mult=0.0)
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(type="UDAIterBasedTrainLoop", max_iters=40000, val_interval=2000)
val_cfg = dict(type="UDAValLoop")
test_cfg = dict(type="UDATestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=2000, max_keep_ckpts=30
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
