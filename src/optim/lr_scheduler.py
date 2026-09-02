"""
Custom Learning Rate Scheduler
MultiStepLR + Linear Warmup (only scheduler used in this project).
"""

import warnings
from torch.optim.lr_scheduler import _LRScheduler
from src.core import register


__all__ = ['MultiStepLRWarmup']


@register
class MultiStepLRWarmup(_LRScheduler):
    """
    MultiStepLR + Linear Warmup

    In warmup stage, lr grows linearly from 0 to base_lr.
    After warmup, decays at milestones with gamma factor.

    Args:
        optimizer: optimizer
        milestones: list of decay epochs (counted from total epochs, incl. warmup)
        warmup_epochs: number of warmup epochs
        gamma: decay factor
        last_epoch: last epoch index
    """

    def __init__(self, optimizer, milestones, warmup_epochs=5, gamma=0.1, last_epoch=-1):
        self.milestones = list(milestones)
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        super(MultiStepLRWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # Warmup: linear growth
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # MultiStep decay
        decay_factor = self.gamma ** sum(1 for m in self.milestones if self.last_epoch >= m)
        return [base_lr * decay_factor for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        decay_factor = self.gamma ** sum(1 for m in self.milestones if self.last_epoch >= m)
        return [base_lr * decay_factor for base_lr in self.base_lrs]
