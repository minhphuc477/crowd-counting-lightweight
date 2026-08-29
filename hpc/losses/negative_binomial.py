from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_MAX_DISPERSION = 1e4


def sum_pool(
    x: torch.Tensor,
    k: Optional[int] = None,
    *,
    input_block_size: Optional[int] = None,
    output_stride: int = 4,
) -> torch.Tensor:
    """Exact non-overlapping sum pooling for count maps."""
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
