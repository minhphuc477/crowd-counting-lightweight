from __future__ import annotations

import math
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set deterministic seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing learning rate scheduler."""
    def fn(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-3, (epoch + 1) / max(1, warmup))
        p = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
