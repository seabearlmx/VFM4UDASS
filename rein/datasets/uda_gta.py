# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
from .uda_cityscapes import UDACityscapesDataset

import json
import os.path as osp
import numpy as np
import torch


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()



@DATASETS.register_module()
class UDAGTADataset(UDACityscapesDataset):
    CLASSES = UDACityscapesDataset.METAINFO['classes']
    PALETTE = UDACityscapesDataset.METAINFO['palette']

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_labelTrainIds.png',
                 rcs_cfg=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                self.data_root, self.rcs_class_temp)
            print(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            print(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(
                    osp.join(self.data_root,
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.data_list):
                file = dic['seg_map_path']
                self.file_to_idx[file] = i
