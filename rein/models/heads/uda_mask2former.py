from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from torch import Tensor
from typing import List, Tuple
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from mmseg.models.builder import MODELS
from mmseg.structures.seg_data_sample import SegDataSample

import torch.nn.functional as F
from mmengine.structures import InstanceData
from ...utils.dist_utils import reduce_mean
from ...utils.point_sample import get_uncertain_point_coords_with_randomness
from mmcv.ops import point_sample
from ..losses.cross_entropy_loss import WeightedCrossEntropyLoss
import numpy as np
from sklearn.cluster import KMeans
import os
from ..backbones.clip_layers import clip


@MODELS.register_module()
class UDAMask2FormerHead(Mask2FormerHead):
    def __init__(self, loss_ce: ConfigType = dict(type='WeightedCrossEntropyLoss', loss_weight=1.0),
                 loss_mmseg_ce: ConfigType = dict(type='mmseg.CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super().__init__(**kwargs)
        self.is_src_train = True
        self.is_trg_train = False
        self.loss_ce = MODELS.build(loss_ce)
        self.loss_mmseg_ce = MODELS.build(loss_mmseg_ce)

    def forward(
            self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:

        batch_size = x[0].shape[0]
        if isinstance(batch_data_samples[0], SegDataSample):
            batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        else:
            batch_img_metas = batch_data_samples# [data_sample for data_sample in batch_data_samples]

        # use vpt_querys to replace query_embed
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                -2:
                ],
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        return cls_pred_list, mask_pred_list

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    def loss(self, x: Tuple[Tensor], img_metas, is_trg=False, is_masked=False, is_src=False, seg_weight=None) -> dict:  # , layout: Tensor
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            img_metas)

        # forward
        all_cls_scores, all_mask_preds = self(x, img_metas)

        # loss
        losses = {}
        if is_src:
            losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        elif is_trg:
            losses = {}
            img_shape = batch_img_metas[0]['img_shape']
            mask_pred = all_mask_preds[-1]
            mask_pred = F.interpolate(
                mask_pred,
                size=(512, 512),
                mode='bilinear',
                align_corners=False)
            mask_pred = mask_pred.sigmoid()
            cls_score = F.softmax(all_cls_scores[-1], dim=-1)[..., :-1]
            logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
            trg_gts = []
            for i in range(mask_pred.shape[0]):
                trg_gt = img_metas[i].gt_sem_seg.data
                trg_gts.append(trg_gt.unsqueeze(0))
            trg_lbl = torch.cat(trg_gts, dim=0).squeeze(1).to(logits.device)
            loss_ce_trg = self.loss_ce(logits, trg_lbl, weight=seg_weight, ignore_index=255)
            losses['trg_ce_loss'] = loss_ce_trg
        elif is_masked:
            losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                       batch_gt_instances, batch_img_metas)
            img_shape = batch_img_metas[0]['img_shape']
            mask_pred = all_mask_preds[-1]
            mask_pred = F.interpolate(
                mask_pred,
                size=(512, 512),
                mode='bilinear',
                align_corners=False)
            mask_pred = mask_pred.sigmoid()
            cls_score = F.softmax(all_cls_scores[-1], dim=-1)[..., :-1]
            logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
            trg_gts = []
            for i in range(mask_pred.shape[0]):
                trg_gt = img_metas[i].gt_sem_seg.data
                trg_gts.append(trg_gt.unsqueeze(0))
            trg_lbl = torch.cat(trg_gts, dim=0).squeeze(1).to(logits.device)
            loss_ce_masked = self.loss_ce(logits, trg_lbl, weight=seg_weight, ignore_index=255)
            losses['masked_ce_loss'] = loss_ce_masked
           

        img_shape = batch_img_metas[0]['img_shape']
        mask_pred = all_mask_preds[-1]
        mask_pred_results = F.interpolate(
            mask_pred,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)
        mask_pred = mask_pred_results.sigmoid()
        cls_score = F.softmax(all_cls_scores[-1], dim=-1)[..., :-1]
        logits = torch.einsum('bqc, bqhw->bhwc', cls_score, mask_pred)
        losses['logits'] = logits
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
        all_cls_scores, all_mask_preds = self(x, batch_img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        return mask_cls_results, mask_pred_results
