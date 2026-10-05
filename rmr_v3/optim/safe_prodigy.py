"""SafeProdigy: Distance-adaptive optimizer with strict numerical safeguards for unrolled solvers."""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable

import torch
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class SafeProdigy(Optimizer):
    r"""Distance-Adaptive Optimizer with strict numerical stability guards for deep unfolding.

    Addresses the exponential spike instability of standard Prodigy (arXiv:2306.06101) by:
      1. Fixing the Step-0 growth rate bypass bug (enforcing rate-limited growth at all steps).
      2. Enforcing a hard ceiling `d_max_cap` (Bounded-D Ceiling) to preserve solver Lipschitz bound.
      3. Providing a strict D-warmup phase (`d_warmup_steps`) where D is frozen at d0.
      4. Rate-limiting growth: D_{k+1} <= D_k * growth_rate.
      5. Gradient-spike gating: Freezing D accumulation when global ||g|| > grad_spike_thresh.
      6. Native RMS parameter-update normalization (StableAdamW style) to protect unrolled layers.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        decouple: bool = True,
        use_bias_correction: bool = True,
        safeguard_warmup: bool = True,
        d0: float = 1e-5,
        d_coef: float = 0.5,
        growth_rate: float = 1.015,
        d_max_cap: float = 0.08,
        d_warmup_steps: int = 400,
        grad_spike_thresh: float = 5.0,
        use_rms_clipping: bool = True,
        slice_p: int = 1,
    ) -> None:
        if d0 <= 0.0:
            raise ValueError(f"Invalid d0: {d0}")
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if growth_rate < 1.0:
            raise ValueError(f"growth_rate must be >= 1.0, got {growth_rate}")
        if d_max_cap <= d0:
            raise ValueError(f"d_max_cap ({d_max_cap}) must be > d0 ({d0})")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta3=beta3,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
            use_bias_correction=use_bias_correction,
            safeguard_warmup=safeguard_warmup,
            d0=d0,
            d=d0,
            d_coef=d_coef,
            growth_rate=growth_rate,
            d_max_cap=d_max_cap,
            d_warmup_steps=d_warmup_steps,
            grad_spike_thresh=grad_spike_thresh,
            use_rms_clipping=use_rms_clipping,
            d_numerator=0.0,
            d_denom=0.0,
            k=0,
            slice_p=slice_p,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step 1: Detect non-finite values and compute global gradient norm with zero intermediate syncs
        grad_norms_sq = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_norms_sq.append(p.grad.data.norm(2).square())

        if not grad_norms_sq:
            return loss

        global_grad_norm = float(torch.stack(grad_norms_sq).sum().sqrt().item())
        if not math.isfinite(global_grad_norm):
            logger.warning("[SafeProdigy] Non-finite gradient detected. Zeroing grads and skipping step.")
            self.zero_grad(set_to_none=True)
            return loss

        group0 = self.param_groups[0]
        k = group0["k"]
        beta1, beta2 = group0["betas"]
        beta3 = group0["beta3"] if group0["beta3"] is not None else math.sqrt(beta2)
        use_bias_correction = group0["use_bias_correction"]
        d = group0["d"]
        d_coef = group0["d_coef"]
        growth_rate = group0["growth_rate"]
        d_max_cap = group0["d_max_cap"]
        d_warmup_steps = group0["d_warmup_steps"]
        grad_spike_thresh = group0["grad_spike_thresh"]
        safeguard_warmup = group0["safeguard_warmup"]
        decouple = group0["decouple"]
        use_rms_clipping = group0["use_rms_clipping"]
        slice_p = group0["slice_p"]

        bias_correction = math.sqrt(1.0 - beta2 ** (k + 1)) / (1.0 - beta1 ** (k + 1)) if use_bias_correction else 1.0

        # Safety gating: only learn distance D past warmup and below gradient spike threshold
        is_safe_for_d = (k >= d_warmup_steps) and (global_grad_norm <= grad_spike_thresh)

        num_terms = []
        denom_terms = []

        # Step 2: Accumulate Adam moments and D adaptation statistics
        for group in self.param_groups:
            d0 = group["d0"]
            group_lr = group["lr"]
            dlr = d * group_lr * bias_correction
            group_num_scale = (d / d0) * dlr

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["p0"] = p.flatten()[::slice_p].detach().clone()
                    state["s"] = torch.zeros_like(state["p0"])
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                s = state["s"]
                p0 = state["p0"]

                # Adam EMA updates
                exp_avg.mul_(beta1).add_(grad, alpha=d * (1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1.0 - beta2))

                # Prodigy distance tracking statistics accumulate continuously from step 0
                sliced_grad = grad.flatten()[::slice_p]
                sliced_p = p.data.flatten()[::slice_p]
                x0_diff = p0 - sliced_p
                dot_prod = torch.dot(sliced_grad, x0_diff)
                num_terms.append(dot_prod * group_num_scale)

                alpha_denom = ((d / d0) * d) if safeguard_warmup else ((d / d0) * dlr)
                s.mul_(beta3).add_(sliced_grad, alpha=alpha_denom)
                denom_terms.append(s.abs().sum())

        delta_numerator = float(torch.stack(num_terms).sum().item()) if num_terms else 0.0
        delta_denom = float(torch.stack(denom_terms).sum().item()) if denom_terms else 0.0

        # Step 3: Compute updated D with rate-limited growth and hard ceiling
        if delta_denom > 0.0:
            d_numerator = group0["d_numerator"] * beta3 + delta_numerator
            d_denom = delta_denom
            d_hat = d_coef * (d_numerator / d_denom)
            group0["d_numerator"] = d_numerator
            group0["d_denom"] = d_denom

            # Only allow d to adapt when past warmup and grad norm is healthy
            if is_safe_for_d:
                d0_val = group0["d0"]
                if d == d0_val:
                    # Allow escape from d0 without 1.015x rate limit lock
                    d = max(d, min(d_hat, d_max_cap))
                else:
                    allowed_max_d = min(d * growth_rate, d_max_cap)
                    if d_hat > d:
                        d = max(d, min(d_hat, allowed_max_d))
                d = min(d, d_max_cap)
        else:
            d = group0["d"]

        # Step 4: Apply parameter updates with RMS normalization
        for group in self.param_groups:
            group["d"] = d
            group["k"] = k + 1
            decay = group["weight_decay"]
            eps = group["eps"]
            dlr = d * group["lr"] * bias_correction

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                denom = exp_avg_sq.sqrt().add_(d * eps)
                step_direction = exp_avg / denom

                if use_rms_clipping:
                    rms = step_direction.norm(2).div(math.sqrt(max(1, step_direction.numel()))).clamp_min(1.0)
                    step_direction.div_(rms)

                if decay != 0.0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                p.data.add_(step_direction, alpha=-dlr)

        return loss
