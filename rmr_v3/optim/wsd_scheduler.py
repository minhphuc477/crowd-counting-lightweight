"""Warmup-Stable-Decay (WSD) learning rate scheduler for RMR."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


def make_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 15,
    *,
    stable_ratio: float = 0.75,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Create Warmup-Stable-Decay (WSD) scheduler as validated by MiniCPM and DeepSeek.

    Phase 1: Warmup (0 -> warmup_epochs)
        Linear ascent to stabilize Adam moving averages and unrolled SIRT gradients.
    Phase 2: Stable (warmup_epochs -> stable_epochs)
        Holds peak learning rate flat at 1.0x base_lr for stable_ratio of total epochs.
        Maintains exploration kinetic energy without premature cosine momentum starvation.
    Phase 3: Decay (stable_epochs -> epochs)
        Rapid cosine annealing towards min_lr_ratio to settle firmly into flat minima.
    """
    total_epochs = max(1, int(epochs))
    w_epochs = max(1, min(int(warmup_epochs), total_epochs // 2))
    s_epochs = max(w_epochs, int(total_epochs * float(stable_ratio)))
    s_epochs = min(s_epochs, total_epochs - max(1, int(total_epochs * 0.05)))
    floor_ratio = float(min_lr_ratio)

    def wsd_lambda(epoch: int) -> float:
        # Phase 1: Linear Warmup
        if epoch < w_epochs:
            return max(1e-3, float(epoch + 1) / float(w_epochs))

        # Phase 2: Stable Plateau (Hold Flat at 100% Peak LR)
        if epoch < s_epochs:
            return 1.0

        # Phase 3: Decisive Cosine Annealing to Floor
        decay_steps = max(1, total_epochs - s_epochs)
        progress = min(1.0, float(epoch - s_epochs) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=wsd_lambda)
