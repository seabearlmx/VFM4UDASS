# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random

import torch
from torch.nn import Module

from ..utils.dacs_transforms import get_mean_std, strong_transform
from ..utils.masking_transforms import build_mask_generator
from mmengine.structures import PixelData


class MaskingConsistencyModule(Module):

    def __init__(self, require_teacher):
        super(MaskingConsistencyModule, self).__init__()

        self.source_only = False
        self.max_iters = 40000
        self.color_jitter_s = 0.2
        self.color_jitter_p = 0.2

        self.mask_mode = 'separatetrgaug'
        self.mask_alpha = 'same'
        self.mask_pseudo_threshold = 'same'
        self.mask_lambda = 1
        self.mask_gen = build_mask_generator(dict(type='block', mask_ratio=0.7, mask_block_size=64))

        assert self.mask_mode in [
            'separate', 'separatesrc', 'separatetrg', 'separateaug',
            'separatesrcaug', 'separatetrgaug'
        ]

    def __call__(self,
                 runner,
                 trg_data_batch,
                 target_img,
                 pseudo_label=None,
                 pseudo_weight=None):

        dev = target_img.device

        masked_img = target_img
        masked_lbl = pseudo_label.unsqueeze(1)
        # masked_seg_weight = pseudo_weight
        # Apply masking to image
        masked_img = self.mask_gen.mask_image(masked_img)

        for i in range(target_img.shape[0]):
            trg_data_batch['data_samples'][i].gt_sem_seg.data = PixelData(**dict(gt_sem_seg=masked_lbl[i].squeeze())).gt_sem_seg.data
            trg_data_batch['inputs'][i] = masked_img[i]

        # Train on masked images
        # masked_outputs, masked_features, masked_logits = runner.model.train_step(
        #     trg_data_batch, optim_wrapper=runner.optim_wrapper, is_masked=True, seg_weight=masked_seg_weight)
        masked_outputs, masked_features, masked_logits = runner.model.train_step(
            trg_data_batch, optim_wrapper=runner.optim_wrapper, is_masked=True)

        return masked_outputs, masked_features, masked_logits
