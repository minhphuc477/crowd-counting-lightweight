from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_MAX_DISPERSION = 1e4


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-12)
    return y + math.log(-math.expm1(-y))


def sum_pool(
    x: torch.Tensor,
    k: Optional[int] = None,
    *,
    input_block_size: Optional[int] = None,
    output_stride: int = 4,
) -> torch.Tensor:
    """Exact non-overlapping sum pooling with both supported APIs."""
    if input_block_size is not None:
        if k is not None:
            raise ValueError("Specify either k or input_block_size, not both")
        if input_block_size % output_stride != 0:
            raise ValueError(
                f"input_block_size ({input_block_size}) must be divisible by "
                f"output_stride ({output_stride})"
            )
        k = input_block_size // output_stride
    if k is None or int(k) <= 0:
        raise ValueError("A positive pooling factor k is required")
    k = int(k)
    if x.shape[-2] % k != 0 or x.shape[-1] % k != 0:
        raise ValueError(
            f"Output map {tuple(x.shape[-2:])} must be divisible by pooling kernel {k}"
        )
    return F.avg_pool2d(x, kernel_size=k, stride=k) * float(k * k)


def negative_binomial_nll_mean_dispersion(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: float | torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """NB NLL with ``Var(Y)=mu+mu^2/r``, evaluated in float32."""
    y = target.to(device=mean.device, dtype=torch.float32)
    mu = mean.to(dtype=torch.float32).clamp_min(eps)
    r = torch.as_tensor(dispersion, device=mean.device, dtype=torch.float32)
    r = r.clamp(min=eps, max=_MAX_DISPERSION)
    if torch.any(y < 0):
        raise ValueError("Negative-Binomial targets must be non-negative")
    log_r_plus_mu = torch.log(r + mu)
    nll = -(
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - log_r_plus_mu)
        + y * (torch.log(mu) - log_r_plus_mu)
    )
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def nb_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    r: float | torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Backward-compatible elementwise NB NLL."""
    return negative_binomial_nll_mean_dispersion(y, mu, r, eps=eps, reduction="none")


def poisson_nll(y: torch.Tensor, mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Elementwise Poisson NLL including the constant ``log(y!)`` term."""
    y = y.float()
    mu = mu.float().clamp_min(eps)
    if torch.any(y < 0):
        raise ValueError("Poisson targets must be non-negative")
    return mu - y * torch.log(mu) + torch.lgamma(y + 1.0)


@torch.no_grad()
def estimate_nb_dispersion_method_of_moments(
    counts: torch.Tensor,
    poisson_like_dispersion: float = 1e4,
    min_dispersion: float = 1e-3,
) -> float:
    """Estimate ``r=mean^2/(variance-mean)`` with a finite Poisson limit."""
    x = counts.float().reshape(-1)
    if x.numel() == 0:
        raise ValueError("counts cannot be empty")
    mean = x.mean()
    var = x.var(unbiased=x.numel() > 1)
    if mean <= 0 or var <= mean:
        return float(poisson_like_dispersion)
    r = (mean * mean) / (var - mean)
    return float(min(max(float(r), min_dispersion), poisson_like_dispersion))


class HierarchicalNBLoss(nn.Module):
    """Existing multi-scale HPC NB/Poisson loss.

    NTPC uses :func:`negative_binomial_nll_mean_dispersion` directly; keeping
    this interface prevents the NTPC branch from breaking the older trainers.
    """

    def __init__(
        self,
        block_sizes: List[int],
        quantiles: Optional[Dict[int, Tuple[float, float]]] = None,
        use_stratified: bool = True,
        use_poisson: bool = False,
        max_dispersion: float = _MAX_DISPERSION,
        learn_dispersion: bool = False,
    ):
        super().__init__()
        if not block_sizes:
            raise ValueError("block_sizes cannot be empty")
        self.block_sizes = [int(b) for b in block_sizes]
        self.use_stratified = bool(use_stratified)
        self.use_poisson = bool(use_poisson)
        self.max_dispersion = float(max_dispersion)
        self.learn_dispersion = bool(learn_dispersion)
        self.quantiles: Dict[int, Tuple[float, float]] = {}
        for key, values in (quantiles or {}).items():
            q50, q90 = float(values[0]), float(values[1])
            if q50 > q90:
                raise ValueError(f"Invalid quantiles for block {key}: {q50} > {q90}")
            self.quantiles[int(key)] = (q50, q90)
        if not self.use_poisson:
            self.raw_dispersions = nn.ParameterDict({
                str(b): nn.Parameter(
                    torch.tensor(_inv_softplus(10.0), dtype=torch.float32),
                    requires_grad=self.learn_dispersion,
                )
                for b in self.block_sizes
            })

    def get_dispersion(self, block_size: int) -> torch.Tensor:
        if self.use_poisson:
            raise RuntimeError("Dispersion is undefined in Poisson mode")
        raw = self.raw_dispersions[str(block_size)]
        return (F.softplus(raw) + 1e-4).clamp_max(self.max_dispersion)

    def init_dispersion_from_stats(self, block_size: int, mean: float, var: float) -> None:
        if self.use_poisson:
            return
        mean, var = float(mean), float(var)
        if mean < 0 or var < 0 or not math.isfinite(mean) or not math.isfinite(var):
            raise ValueError(f"Invalid mean/var: mean={mean}, var={var}")
        r0 = mean * mean / (var - mean) if var - mean > 1e-8 else self.max_dispersion
        r0 = min(max(r0, 1e-3), min(self.max_dispersion, 100.0))
        with torch.no_grad():
            self.raw_dispersions[str(block_size)].fill_(_inv_softplus(max(r0 - 1e-4, 1e-8)))

    def forward(
        self,
        d_map: torch.Tensor,
        gt_block_counts: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        details: Dict[str, torch.Tensor] = {}
        scale_losses = []
        for block_size in self.block_sizes:
            if block_size not in gt_block_counts:
                raise KeyError(f"Missing GT block counts for scale {block_size}")
            pred = sum_pool(d_map, input_block_size=block_size, output_stride=4)
            if pred.ndim == 4 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            target = gt_block_counts[block_size]
            if target.ndim == 4 and target.shape[1] == 1:
                target = target.squeeze(1)
            target = target.to(device=pred.device, dtype=torch.float32)
            if target.shape != pred.shape:
                raise ValueError(
                    f"Scale {block_size}: GT shape {tuple(target.shape)} "
                    f"!= prediction shape {tuple(pred.shape)}"
                )
            if self.use_poisson:
                pointwise = poisson_nll(target, pred)
            else:
                dispersion = self.get_dispersion(block_size)
                pointwise = nb_nll(target, pred, dispersion)
                details[f"dispersion_{block_size}"] = dispersion.detach()
            if self.use_stratified and block_size in self.quantiles:
                q50, q90 = self.quantiles[block_size]
                masks = (
                    target == 0,
                    (target > 0) & (target <= q50),
                    (target > q50) & (target <= q90),
                    target > q90,
                )
                grouped = [pointwise[mask].mean() for mask in masks if mask.any()]
                scale_loss = torch.stack(grouped).mean() if grouped else pointwise.mean()
            else:
                scale_loss = pointwise.mean()
            if not torch.isfinite(scale_loss):
                raise FloatingPointError(f"Non-finite NB/Poisson loss at block scale {block_size}")
            scale_losses.append(scale_loss)
            details[f"hnb_scale_{block_size}"] = scale_loss.detach()
        total = torch.stack(scale_losses).mean() if scale_losses else d_map.new_zeros(())
        return total, details
