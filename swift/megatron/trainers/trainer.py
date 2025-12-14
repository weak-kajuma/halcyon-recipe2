# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import List, Optional

import torch
import torch.nn
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import get_args, get_timers
from torch.distributed.nn import all_reduce
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronTrainer(BaseMegatronTrainer):

    def seq_cls_loss_func(self, output_tensor, *, labels: torch.Tensor, packed_seq_params=None):
        args = self.args
        assert args.padding_free, 'Currently `task_type="seq_cls"` only supports padding_free.'
        assert args.context_parallel_size == 1, 'Currently `task_type="seq_cls"` does not support context parallelism.'
        last_token = packed_seq_params.cu_seqlens_q[1:packed_seq_params.num_samples + 1] - 1
        logits = output_tensor[0, last_token]
        num_labels = args.num_labels
        acc = None
        if args.problem_type == 'regression':
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif args.problem_type == 'single_label_classification':
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, num_labels)
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            acc = (logits.detach().argmax(dim=-1) == labels).float().mean()
        elif args.problem_type == 'multi_label_classification':
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        metric = {'loss': loss.detach().clone()}
        if acc is not None:
            metric['acc'] = acc
        metric = self._all_reduce_metric(metric)
        return loss, metric

    # Code borrowed from NVIDIA/Megatron-LM
    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  loss_scale: Optional[torch.Tensor] = None,
                  channels: Optional[List[str]] = None,
                  packed_seq_params=None):
        args = get_args()

        patch_size = getattr(args, 'patch_size', 1)
        if patch_size > 1:
            loss = self._compute_patch_loss(output_tensor, labels, patch_size)
        else:
            losses = output_tensor.float()
            loss_mask = labels != -100
            if args.enable_dft_loss:
                losses = losses * torch.exp(-losses.detach())
            if loss_scale is not None:
                losses = losses * loss_scale
            if args.enable_channel_loss and channels is not None:
                assert losses.shape[0] == 1, 'only support padding_free'
                mode = 'train' if self.unwrapped_models[0].training else 'eval'
                metrics = self.custom_metrics[mode]
                num_samples = packed_seq_params.num_samples
                cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
                for i in range(cu_seqlens.shape[0] - 1):
                    channel = channels[i]
                    slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                    metrics[f'loss_{channel}'].update(losses[0, slice_][loss_mask[0, slice_]])

            loss = torch.cat([torch.sum(losses * loss_mask).view(1), loss_mask.sum().view(1)])

        if args.context_parallel_size > 1 and not self.mcore_013:
            loss = all_reduce(loss, group=mpu.get_context_parallel_group())

        # Check individual rank losses are not NaN prior to DP all-reduce.
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message='found NaN in local forward loss calculation',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message='found Inf in local forward loss calculation',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
        # Check for spiky loss
        if args.check_for_spiky_loss:
            # define spiky loss as a loss that's 10x the max loss observed
            SPIKY_LOSS_FACTOR = 10
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context='loss',
                ),
                message='Spiky loss',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=False,
            )
        # Reduce loss for logging.
        reporting_loss = loss.detach().clone()
        lm_loss = loss[0]
        if not self.mcore_013:
            # fix megatron-lm bug
            # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
            lm_loss = lm_loss / mpu.get_context_parallel_world_size()
            reporting_loss = (reporting_loss[0], reporting_loss[1])
        else:
            lm_loss = lm_loss.clone()
        local_num_tokens = loss[1].detach().clone().to(torch.int)
        return (
            lm_loss,
            local_num_tokens,
            {
                'lm loss': reporting_loss
            },
        )

    def forward_step(self, data_iterator, model):
        timers = get_timers()

        # Get the batch.
        vp_stage = model.module.module.vp_stage
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()
        loss_scale = data.pop('loss_scale', None)
        channels = data.pop('channel', None)
        labels = data.get('labels')
        if self.args.task_type == 'seq_cls':
            data.pop('labels', None)
        with self.stimer:
            output_tensor = model(**data)
        packed_seq_params = data.get('packed_seq_params')
        if self.args.task_type == 'seq_cls':
            loss_func = partial(self.seq_cls_loss_func, labels=labels, packed_seq_params=packed_seq_params)
        else:
            loss_func = partial(
                self.loss_func,
                labels=labels,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params)
        return output_tensor, loss_func

    def _compute_patch_loss(self, logits: torch.Tensor, labels: torch.Tensor, patch_size: int):
        logits = logits.float()
        shift_logits = logits[:, :-1, :]
        if shift_logits.numel() == 0:
            zero = logits.new_zeros(1)
            return torch.stack([zero, zero])
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_labels = labels[..., patch_size:]
        if target_labels.shape[-1] % patch_size != 0:
            raise ValueError('Label length must align with patch_size for PLT.')
        log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        target_labels = target_labels.reshape(-1, patch_size)
        ignore_index = -100
        valid_mask = target_labels.ne(ignore_index)
        total_valid = valid_mask.sum()
        losses = []
        for offset in range(patch_size):
            if not valid_mask[:, offset].any():
                continue
            loss_offset = F.nll_loss(
                log_probs,
                target_labels[:, offset],
                ignore_index=ignore_index,
                reduction='sum',
            )
            denom = valid_mask[:, offset].sum()
            losses.append(loss_offset / denom)
        loss_val = torch.stack(losses).mean() if losses else log_probs.new_zeros(1)
        total_valid = total_valid.to(dtype=loss_val.dtype)
        return torch.stack([loss_val * total_valid, total_valid])
