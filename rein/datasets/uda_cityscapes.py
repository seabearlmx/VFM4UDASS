# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

import json
import os.path as osp
import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Sequence, Union
import mmengine
import mmengine.fileio as fileio


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
class UDACityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 rcs_cfg=None,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        super().full_init()
        self.data_list = self.load_data_list()
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
                file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = super().__getitem__(i1)
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['data_samples'].gt_sem_seg.data == c)
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s1 = super().__getitem__(i1)
        return s1

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        return super().__getitem__(idx)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list