"""Safe Learning Rate Finder with SIRT numerical stability safeguards for RMR."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SafeLRFinder:
    """Pre-training LR range test with numerical divergence guards for unrolled solvers.

    Sweeps learning rate exponentially from min_lr to max_lr over a fixed number of iterations.
    Applies exponential moving average (EMA) with bias correction to smooth loss curves.
    Enforces divergence guards to avoid corrupting model weights or triggering NaNs in SIRT.
    Automatically restores original model weights and optimizer state upon completion.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Any, torch.Tensor], dict[str, torch.Tensor] | torch.Tensor],
        device: torch.device,
        smooth_beta: float = 0.95,
        divergence_threshold: float = 4.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.smooth_beta = smooth_beta
        self.divergence_threshold = divergence_threshold

    def range_test(
        self,
        dataloader: DataLoader,
        min_lr: float = 1e-7,
        max_lr: float = 1e-1,
        num_iter: int = 80,
        max_safe_lr: float = 1e-3,
        grad_clip: float = 1.0,
    ) -> float:
        """Run range test on a snapshot of model weights and return optimal suggested LR."""
        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())

        orig_lrs = [group["lr"] for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group["lr"] = min_lr

        mult = (max_lr / min_lr) ** (1.0 / max(1, num_iter))
        current_lr = min_lr
        best_loss = float("inf")
        smoothed_loss = 0.0

        lrs: list[float] = []
        losses: list[float] = []

        self.model.train()
        data_iter = iter(dataloader)

        for step in range(num_iter):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if isinstance(batch, (list, tuple)):
                raw_img, raw_tgt = batch[0], batch[1]
            else:
                raw_img, raw_tgt = batch["image"], batch["target_y"]
            images = raw_img.to(self.device, non_blocking=True)
            targets = raw_tgt.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)
            loss_dict = self.loss_fn(outputs, targets)
            loss = loss_dict["total"] if isinstance(loss_dict, dict) else loss_dict

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

            if not math.isfinite(loss_val) or (step > 10 and smoothed_loss > self.divergence_threshold * best_loss):
                break

            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            if step == 0:
                smoothed_loss = loss_val
            else:
                smoothed_loss = self.smooth_beta * smoothed_loss + (1.0 - self.smooth_beta) * loss_val

            bias_corrected_loss = smoothed_loss / (1.0 - (self.smooth_beta ** (step + 1)))
            if bias_corrected_loss < best_loss:
                best_loss = bias_corrected_loss

            lrs.append(current_lr)
            losses.append(bias_corrected_loss)

            current_lr *= mult
            for group in self.optimizer.param_groups:
                group["lr"] = current_lr

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        for group, orig_lr in zip(self.optimizer.param_groups, orig_lrs):
            group["lr"] = orig_lr

        return self._find_optimal_lr(lrs, losses, max_safe_lr=max_safe_lr)

    @staticmethod
    def _find_optimal_lr(
        lrs: list[float],
        losses: list[float],
        max_safe_lr: float = 1e-3,
    ) -> float:
        """Find learning rate at point of steepest descent with moving window finite difference."""
        if len(losses) < 8:
            return min(1e-4, max_safe_lr)

        log_lrs = [math.log10(lr) for lr in lrs]
        best_steepest_lr = lrs[0]
        min_grad = float("inf")

        for i in range(2, len(losses) - 2):
            delta_log = log_lrs[i + 2] - log_lrs[i - 2]
            if abs(delta_log) < 1e-9:
                continue
            grad = (losses[i + 2] - losses[i - 2]) / delta_log
            if grad < min_grad:
                min_grad = grad
                best_steepest_lr = lrs[i]

        return float(min(best_steepest_lr, max_safe_lr))
