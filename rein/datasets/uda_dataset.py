# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import json
import os.path as osp
import numpy as np
import torch

from .builder import DATASETS
from .builder import build_uda_dataset
from torch.utils.data import DataLoader

@DATASETS.register_module()
class UDADataset(object):
    def __init__(self, source, target): # , **kwargs
        self.source_cfg = source
        self.target_cfg = target
        self.source_ds = build_uda_dataset(source)
        self.target_ds = build_uda_dataset(target)

    def __len__(self):
        return max(len(self.source_ds), len(self.target_ds))