import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import HistoryBuffer, print_log
from mmengine.registry import LOOPS
from mmengine.structures import BaseDataElement
from mmengine.structures import PixelData
from mmengine.utils import is_list_of
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals
from mmengine.runner.loops import IterBasedTrainLoop, _InfiniteDataloaderIterator, ValLoop, TestLoop, _parse_losses, _update_losses
from ..models.utils.dacs_transforms import *
import random
from mmseg.utils import add_prefix
from mmengine.model.wrappers import MMDistributedDataParallel
import copy
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd
from ..models.uda.masking_consistency_module  import MaskingConsistencyModule
import torch.fft as fft



@LOOPS.register_module()
class UDADZIterBasedTrainLoop(IterBasedTrainLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader,
            max_iters: int,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        # super().__init__(runner, dataloader, max_iters)
        self.src_dataloader_cfg = {'dataset': dataloader['dataset']['source'].copy()}
        self.trg_dataloader_cfg = {'dataset': dataloader['dataset']['target'].copy()}
        dataloader.pop('dataset')
        self.src_dataloader_cfg_all = dataloader.copy()
        self.trg_dataloader_cfg_all = dataloader.copy()
        self.src_dataloader_cfg_all.merge(self.src_dataloader_cfg)
        self.trg_dataloader_cfg_all.merge(self.trg_dataloader_cfg)
        super().__init__(runner, self.src_dataloader_cfg_all, max_iters)


        self.src_dataloader = None
        self.trg_dataloader = None
        self.src_dataloader_iterator = None
        self.trg_dataloader_iterator = None
        self._runner = runner
        self.batch_size = dataloader['batch_size']
        self.ema_model = copy.deepcopy(runner.model)
        # self.mic_model = copy.deepcopy(runner.model)
        self.mic = MaskingConsistencyModule(require_teacher=False)

        if isinstance(self.src_dataloader_cfg_all, dict) and isinstance(self.trg_dataloader_cfg_all, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.src_dataloader = runner.build_dataloader(
                self.src_dataloader_cfg_all, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            self.trg_dataloader = runner.build_dataloader(
                self.trg_dataloader_cfg_all, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.src_dataloader = self.src_dataloader_cfg_all
            self.trg_dataloader = self.trg_dataloader_cfg_all

        self._max_iters = int(max_iters)
        assert self._max_iters == max_iters, \
            f'`max_iters` should be a integer number, but get {max_iters}'
        self._max_epochs = 1  # for compatibility with EpochBasedTrainLoop
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.src_dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.src_dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.src_dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        if hasattr(self.trg_dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.trg_dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.trg_dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)
        # get the iterator of the dataloader

        self.src_dataloader_iterator = _InfiniteDataloaderIterator(self.src_dataloader)
        self.trg_dataloader_iterator = _InfiniteDataloaderIterator(self.trg_dataloader)

        self.cur_src_data = None
        self.cur_trg_data = None
        self.cur_src_feats = None
        self.cur_src_logits = None
        self.cur_trg_feats = None
        self.cur_trg_logits = None
        self.is_src_train = True
        self.is_trg_train = False
        self.alpha = 0.999
        self.pseudo_threshold = 0.968
        self.psweight_ignore_top = 15
        self.psweight_ignore_bottom = 120
        self.valid_pseudo_mask = None

        self.mix = 'class'
        self.blur = True
        self.color_jitter_s = 0.2
        self.color_jitter_p = 0.2
        # mean = [
        #     torch.as_tensor([123.675, 116.28, 103.53], device='cuda')
        #     for i in range(self.batch_size)
        # ]
        # self.mean = torch.stack(mean).view(-1, 3, 1, 1)
        # std = [
        #     torch.as_tensor([58.395, 57.12, 57.375], device='cuda')
        #     for i in range(self.batch_size)
        # ]
        # self.std = torch.stack(std).view(-1, 3, 1, 1)
        self.strong_parameters = {
            'mix': None,
            # 'color_jitter': random.uniform(0, 1),
            # 'color_jitter_s': self.color_jitter_s,
            # 'color_jitter_p': self.color_jitter_p,
            # 'blur': random.uniform(0, 1) if self.blur else 0,
            # 'mean': self.mean[0].unsqueeze(0),  # assume same normalization
            # 'std': self.std[0].unsqueeze(0)
        }

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    def get_module(self, module):
        """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

        Args:
            module (MMDistributedDataParallel | nn.ModuleDict): The input
                module that needs processing.

        Returns:
            nn.ModuleDict: The ModuleDict of multiple networks.
        """
        if isinstance(module, MMDistributedDataParallel):
            return module.module

        return module

    def get_model(self):
        return self.get_module(self.runner.model)

    def get_ema_model(self):
        return self.get_module(self.ema_model)

    def _init_ema_weights(self):
        with torch.no_grad():
            for param in self.get_ema_model().parameters():
                param.detach_()
                param.requires_grad = False
            mp = list(self.get_module(self.runner.model).parameters())
            mcp = list(self.get_ema_model().parameters())
            for i in range(0, len(mp)):
                if not mcp[i].data.shape:  # scalar tensor
                    mcp[i].data = mp[i].data.clone()
                else:
                    mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        with torch.no_grad():
            alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
            for ema_param, param in zip(self.get_ema_model().parameters(),
                                        self.get_module(self.runner.model).parameters()):
                if not param.data.shape:  # scalar tensor
                    ema_param.data = \
                        alpha_teacher * ema_param.data + \
                        (1 - alpha_teacher) * param.data
                else:
                    ema_param.data[:] = \
                        alpha_teacher * ema_param[:].data[:] + \
                        (1 - alpha_teacher) * param[:].data[:]

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        # print('filter_valid_pseudo_region: valid_pseudo_mask: ', valid_pseudo_mask)
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def frequency_style_transfer(self, source_img, target_img):
        _, _, H, W = source_img.shape

        source_fft = fft.rfftn(source_img, dim=(-2, -1), norm='backward')
        target_fft = fft.rfftn(target_img, dim=(-2, -1), norm='backward')

        source_mag = torch.abs(source_fft)
        source_phase = torch.angle(source_fft)
        target_mag = torch.abs(target_fft)

        real = target_mag * torch.cos(source_phase)
        imag = target_mag * torch.sin(source_phase)
        merged_fft = torch.complex(real, imag)

        merged_img = fft.irfftn(merged_fft, s=(H, W), dim=(-2, -1), norm='backward')

        return merged_img

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            for _ in range(self._iter):
                next(self.src_dataloader_iterator)
                next(self.trg_dataloader_iterator)
        while self._iter < self._max_iters and not self.stop_training:

            if self._iter == 0:
                self._init_ema_weights()
            if self._iter > 0:
                self._update_ema(self._iter)

            self.runner.model.train()

            src_data_batch = next(self.src_dataloader_iterator)
            trg_data_batch = next(self.trg_dataloader_iterator)
            self.run_iter([src_data_batch, trg_data_batch]) # , is_trg=self.is_trg_train

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()


        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch, is_src=False, is_trg=False) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """

        src_data_batch, trg_data_batch = data_batch
        dev = 'cuda'
        src_img = []
        trg_img = []
        src_gt_semantic_seg = []
        for i in range(self.batch_size):
            src_img.append(src_data_batch['inputs'][i].unsqueeze(0))
            trg_img.append(trg_data_batch['inputs'][i].unsqueeze(0))
            src_gt_semantic_seg.append(src_data_batch['data_samples'][i].gt_sem_seg.data)
        src_img = torch.cat(src_img, dim=0).to(dev)
        trg_img = torch.cat(trg_img, dim=0).to(dev)
        src_gt_semantic_seg = torch.cat(src_gt_semantic_seg, dim=0).unsqueeze(1).to(dev)

        # sty_src_img = self.frequency_style_transfer(src_img, trg_img)

        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=src_data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.


        src_outputs, src_features, src_logits = self.runner.model.train_step(
            src_data_batch, optim_wrapper=self.runner.optim_wrapper, is_src=True)

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        trg_input = self.get_ema_model().data_preprocessor(trg_data_batch, training=True)
        ema_logits = self.get_ema_model().generate_pseudo_label(trg_input['inputs'], trg_input['data_samples']) # , trg_data_batch
        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(ema_logits)
        pseudo_weight = self.filter_valid_pseudo_region(pseudo_weight, self.valid_pseudo_mask)
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
        #
        # # Apply mixing
        mixed_img, mixed_lbl = [None] * self.batch_size, [None] * self.batch_size
        mixed_seg_weight = pseudo_weight.clone()
        mix_masks = get_class_masks(src_gt_semantic_seg)
        #
        for i in range(self.batch_size):
            self.strong_parameters['mix'] = mix_masks[i].to(dev)
            mixed_img[i], mixed_lbl[i] = strong_transform(
                self.strong_parameters,
                data=torch.stack((src_img[i].to(dev), trg_img[i].to(dev))),
                target=torch.stack(
                    (src_gt_semantic_seg[i][0].to(dev), pseudo_label[i].to(dev))))
            _, mixed_seg_weight[i] = strong_transform(
                self.strong_parameters,
                target=torch.stack((gt_pixel_weight[i].to(dev), pseudo_weight[i].to(dev))))
        #
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        for i in range(self.batch_size):
            trg_data_batch['data_samples'][i].gt_sem_seg = PixelData(**dict(data=mixed_lbl[i].squeeze()))
            trg_data_batch['inputs'][i] = mixed_img[i]


        trg_outputs, trg_features, trg_logits = self.runner.model.train_step(
            trg_data_batch, optim_wrapper=self.runner.optim_wrapper, is_trg=True, seg_weight=mixed_seg_weight)

        # masked_outputs, masked_features, masked_logits = self.mic(self.runner, trg_data_batch, trg_img, pseudo_label)


        self.cur_trg_data = trg_data_batch
        self.cur_trg_logits = trg_logits
        self.cur_trg_feats = trg_features
        self.cur_src_data = src_data_batch
        self.cur_src_logits = src_logits
        self.cur_src_feats = src_features

        src_outputs = add_prefix(src_outputs, 'src')
        trg_outputs = add_prefix(trg_outputs, 'trg')
        # masked_outputs = add_prefix(masked_outputs, 'masked')

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs={**src_outputs, **trg_outputs}
            )
        
        self._iter += 1


@LOOPS.register_module()
class UDADZValLoop(ValLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        self._runner = runner
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader


        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16
        self.val_loss: Dict[str, HistoryBuffer] = dict()

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        # clear val loss
        self.val_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.val_loss:
            loss_dict = _parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)

        outputs, self.val_loss = _update_losses(outputs, self.val_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class UDADZTestLoop(TestLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        self._runner = runner
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16
        self.test_loss: Dict[str, HistoryBuffer] = dict()

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # clear test loss
        self.test_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
