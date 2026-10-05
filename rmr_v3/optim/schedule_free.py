"""Meta FAIR Schedule-Free AdamW Optimizer.

Official implementation adapted from facebookresearch/schedule_free:
"The Road Less Scheduled: Defending Against Learning Rate Decay" (Defazio et al., ICLR 2024).
Eliminates the need for learning rate schedules (Cosine, Step, WSD) by combining
gradient evaluation at interpolation point y with Polyak-Ruppert iterate averaging on x.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.optim


class AdamWScheduleFree(torch.optim.Optimizer):
    r"""Schedule-Free AdamW optimizer.

    No learning rate schedule (cosine, linear, wsd) is needed with this optimizer.
    Maintains:
      - y_t: evaluation point where gradient is evaluated (used in train mode)
      - z_t: base update sequence
      - x_t: iterate-averaged optimal sequence (used in eval mode and for checkpointing)

    Requires .train() before training iterations and .eval() before evaluation and saving checkpoints.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        inner_momentum: float = 0.0,
        foreach: Optional[bool] = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if foreach is None:
            foreach = hasattr(torch, "_foreach_mul_")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            k=0,
            warmup_steps=warmup_steps,
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            inner_momentum=inner_momentum,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self) -> None:
        """Switch model parameters from evaluation sequence y to averaged sequence x."""
        for group in self.param_groups:
            train_mode = group.get("train_mode", False)
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state.get(p)
                    if state is not None and "z" in state:
                        # Set p to x: p.lerp_(z, weight=1 - 1/beta1)
                        p.data.lerp_(state["z"].to(p.device), weight=1.0 - 1.0 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Switch model parameters from averaged sequence x to evaluation sequence y."""
        for group in self.param_groups:
            train_mode = group.get("train_mode", False)
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state.get(p)
                    if state is not None and "z" in state:
                        # Set p to y: p.lerp_(z, weight=1 - beta1)
                        p.data.lerp_(state["z"].to(p.device), weight=1.0 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group.get("train_mode", False):
                # Auto-initialize into train mode if user forgot to call .train()
                self.train()
                break

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]
            inner_momentum = group["inner_momentum"]

            sched = (k + 1) / warmup_steps if (warmup_steps > 0 and k < warmup_steps) else 1.0

            bias_correction2 = 1.0 - beta2 ** (k + 1)
            bias_correction1 = 1.0 - inner_momentum ** (k + 1) if inner_momentum != 0 else 1.0
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            lr_max = max(lr, group["lr_max"])
            group["lr_max"] = lr_max

            weight = ((k + 1) ** r) * (lr_max ** weight_lr_power)
            weight_sum = group["weight_sum"] + weight
            group["weight_sum"] = weight_sum

            ckp1 = weight / weight_sum if weight_sum > 0 else 0.0
            active_p = [p for p in group["params"] if p.grad is not None]

            for p in active_p:
                if "z" not in self.state[p]:
                    self.state[p]["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if inner_momentum != 0:
                        self.state[p]["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            for p in active_p:
                y = p.data
                grad = p.grad.data
                state = self.state[p]
                z = state["z"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                if inner_momentum != 0:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(inner_momentum).add_(grad, alpha=1.0 - inner_momentum)
                    grad_normalized = exp_avg.div(bias_correction1).div_(denom)
                else:
                    grad_normalized = grad.div(denom)

                if decay != 0:
                    grad_normalized.add_(y, alpha=decay)

                # In-place Polyak-Ruppert interpolation update
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad_normalized, alpha=lr * (beta1 * (1.0 - ckp1) - 1.0))
                z.sub_(grad_normalized, alpha=lr)

            group["k"] = k + 1

        return loss
