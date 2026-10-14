from __future__ import annotations

import math
from typing import Any
import torch


class IdentityScheduler:
    """No-op scheduler for schedule-free, distance-adaptive (SafeProdigy), and fixed LR optimizers."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch: int | None = None) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"type": "identity"}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup: int = 5,
    *,
    warmup_epochs: int | None = None,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing learning rate scheduler with a minimum floor."""
    if warmup_epochs is not None:
        warmup = warmup_epochs

    def fn(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-3, (epoch + 1) / max(1, warmup))
        p = (epoch - warmup) / max(1, epochs - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


make_cosine_scheduler = make_scheduler

__all__ = [
    "IdentityScheduler",
    "make_cosine_scheduler",
    "make_scheduler",
]
