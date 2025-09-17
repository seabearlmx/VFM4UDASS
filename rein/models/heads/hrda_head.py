# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Update debug_output
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy

import torch
from torch.nn import functional as F

from ..utils.ops import add_prefix
from ..utils.ops import resize as _resize
from .. import builder
from mmseg.models.builder import MODELS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead

from ..losses.cross_entropy_loss import WeightedCrossEntropyLoss
from mmseg.utils import ConfigType, SampleList
from torch import Tensor
from typing import List, Tuple
from typing import Dict, Optional, Union
from mmseg.models.builder import MODELS


def scale_box(box, scale):
    y1, y2, x1, x2 = box
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


@MODELS.register_module()
class HRDAHead(BaseDecodeHead):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 align_corners=False,
                 loss_ce: ConfigType = dict(type='WeightedCrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = single_scale_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention
        self.loss_ce = MODELS.build(loss_ce)
        self.align_corners = align_corners

    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            crop_seg_logits = self.head(features)
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        lr_seg = self.head(lr_inp)

        hr_seg = self.decode_hr(hr_inp, batch_size)

        att = self.get_scale_attention(lr_sc_att_inp)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask
        lr_seg = (1 - att) * lr_seg
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,
                      img_metas,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)

        trg_gts = []
        for i in range(inputs[0][0].shape[0]):
            trg_gt = img_metas[i].gt_sem_seg.data
            trg_gts.append(trg_gt.unsqueeze(0))
        gt_semantic_seg = torch.cat(trg_gts, dim=0).squeeze(1).to(inputs[0][0].device)  # torch.Size([4, 512, 512])

        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        losses['logits'] = seg_logits
        losses['features'] = inputs
        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""
        fused_seg, lr_seg, hr_seg = seg_logit

        fused_seg = _resize(
            input=fused_seg,
            size=seg_label.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        loss = {}

        loss['loss_seg'] = self.loss_ce(fused_seg, seg_label, weight=seg_weight, ignore_index=255)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            lr_seg = _resize(
                input=lr_seg,
                size=seg_label.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)
            loss.update(add_prefix({'loss_seg': self.loss_ce(lr_seg, seg_label, weight=seg_weight, ignore_index=255)}, 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            hr_seg = _resize(
                input=hr_seg,
                size=cropped_seg_label.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)
            loss.update(
                add_prefix({'loss_seg': self.loss_ce(hr_seg, cropped_seg_label, weight=cropped_seg_weight, ignore_index=255)}, 'hr'))
        elif self.hr_loss_weight > 0:
            hr_seg = _resize(
                input=hr_seg,
                size=seg_label.shape[1:],
                mode='bilinear',
                align_corners=self.align_corners)
            loss.update(add_prefix({'loss_seg': self.loss_ce(hr_seg, seg_label, weight=seg_weight, ignore_index=255)}, 'hr'))
        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight
        return loss
