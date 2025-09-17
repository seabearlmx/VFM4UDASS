# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Support for seg_weight
# - Add debug flag
# - Add return_logits flag
# - Update debug_output from loss

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from ..utils.ops import resize
from ..losses.cross_entropy_loss import WeightedCrossEntropyLoss

from mmseg.utils import ConfigType, SampleList
from torch import Tensor
from typing import List, Tuple
from typing import Dict, Optional, Union
from mmseg.models.builder import MODELS


class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_ce: ConfigType = dict(type='WeightedCrossEntropyLoss', loss_weight=1.0),
                 loss_mmseg_ce: ConfigType = dict(type='mmseg.CrossEntropyLoss', loss_weight=1.0),
                 decoder_params=None,
                 sampler=None,
                 ignore_index=255,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_ce = MODELS.build(loss_ce)
        self.loss_mmseg_ce = MODELS.build(loss_mmseg_ce)
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.out_channels = num_classes
        self.threshold = threshold

        self.city_class_weight = [0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                        1.0865, 1.0955, 1.0865, 1.1529, 1.0507]

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def loss(self, x: Tuple[Tensor], img_metas, is_trg=False, is_masked=False, is_src=False,
             seg_weight=None) -> dict:
        # forward
        seg_logit = self(x)

        trg_gts = []
        for i in range(seg_logit.shape[0]):
            trg_gt = img_metas[i].gt_sem_seg.data
            trg_gts.append(trg_gt.unsqueeze(0))
        trg_lbl = torch.cat(trg_gts, dim=0).squeeze(1).to(seg_logit.device)  # torch.Size([4, 512, 512])

        seg_logit = resize(
            input=seg_logit,
            size=trg_lbl.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        if is_src:
            loss_ce = self.loss_mmseg_ce(seg_logit, trg_lbl, ignore_index=255)
        elif is_trg:
            loss_ce = self.loss_ce(seg_logit, trg_lbl, weight=seg_weight, class_weight=self.city_class_weight, ignore_index=255)
        else:
            loss_ce = self.loss_ce(seg_logit, trg_lbl, weight=seg_weight, class_weight=self.city_class_weight, ignore_index=255)

        # loss
        losses = {}
        if is_src:
            losses['src_ce_loss'] = loss_ce
        elif is_trg:
            losses['trg_ce_loss'] = loss_ce
        elif is_masked:
            losses['masked_ce_loss'] = loss_ce

        losses['logits'] = seg_logit
        losses['features'] = x

        return losses  

    def predict(self, x: Tuple[Tensor], batch_img_metas) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        seg_logit = self(x)

        return seg_logit
