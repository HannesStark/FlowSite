from collections import defaultdict
from typing import Any
import torch
from torch import optim, Tensor
import lightning.pytorch as pl
from utils.logging import lg


class GeneralModule(pl.LightningModule):
    def __init__(self, args, device, model):
        super().__init__()
        self.args = args
        self.model = model
        self.rare_logging_values = defaultdict(list)
        self.rare_logs_prepared = {}
        self.total_steps = 0
        self.total_val_steps = 0
        self.num_invalid_gradients = 0

    def make_logs(self, logs, prefix, batch, **kwargs):
        for key in logs:
            prog_bar = False if 'int' in key or 'time' in key or 'all_res' in key else True # dont log the interval losses to the progress bar
            self.log(f"{prefix}{key}", logs[key], batch_size=batch.num_graphs, prog_bar=prog_bar, **kwargs)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        if self.args.plateau_scheduler:
            raise NotImplementedError('This still needs to be integrated into the lightning training loop')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=self.args.scheduler_patience, min_lr=float(self.args.lr) / 100)
        else:
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.args.lr_start, end_factor=1.0, total_iters=self.args.warmup_dur)
            constant = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=self.args.constant_dur)
            decay = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=self.args.lr_end, total_iters=self.args.decay_dur)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, constant, decay], milestones=[self.args.warmup_dur, self.args.warmup_dur + self.args.constant_dur])
        return [optimizer], [scheduler]



    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            lg(f'WARNING: NaN or Inf gradients encountered after calling backward. Setting gradients to zero.')
            self.num_invalid_gradients += 1
            self.zero_grad()
    def on_before_optimizer_step(self, optimizer):
        if self.args.check_unused_params:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    lg(f'gradients were None for {name}')

        if self.args.check_nan_grads:
            had_nan_grads = False
            for name, p in self.model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    had_nan_grads = True
                    lg(f'gradients were nan for {name}')
            if had_nan_grads and self.args.except_on_nan_grads:
                raise Exception('There were nan gradients and except_on_nan_grads was set to True')
    def general_step_oom_wrapper(self, batch, batch_idx):
        try:
            return self.general_step(batch, batch_idx)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print('| WARNING: ran OOM error, skipping batch. Exception:', str(e))
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None
            else:
                raise e
    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        r"""Overrides the PyTorch Lightning backward step and adds the OOM check."""
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print('| WARNING: ran OOM error, skipping batch. Exception:', str(e))
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

