from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sum_pool(x: torch.Tensor, k: int) -> torch.Tensor:
    """Exact sum pooling with kernel factor k."""
    return F.avg_pool2d(x, kernel_size=k, stride=k) * (k * k)


def negative_binomial_nll_mean_dispersion(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: float | torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative Binomial NLL parameterization with mean mu and dispersion r.

    Always runs in float32.
    """
    y = target.to(device=mean.device, dtype=torch.float32)
    mu = mean.to(dtype=torch.float32).clamp_min(eps)

    r = torch.as_tensor(
        dispersion,
        device=mean.device,
        dtype=torch.float32,
    ).clamp_min(eps)

    log_prob = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - torch.log(r + mu))
        + y * (torch.log(mu) - torch.log(r + mu))
    )

    nll = -log_prob

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(reduction)


# Backward-compatible alias
nb_nll = negative_binomial_nll_mean_dispersion


@torch.no_grad()
def estimate_nb_dispersion_method_of_moments(
    counts: torch.Tensor,
    poisson_like_dispersion: float = 1e6,
    min_dispersion: float = 1e-3,
) -> float:
    """Method-of-moments estimator for NB dispersion r = mean^2 / (var - mean)."""
    x = counts.float()
    mean = x.mean()
    var = x.var(unbiased=True)

    if mean <= 0 or var <= mean:
        return poisson_like_dispersion

    r = (mean * mean) / (var - mean)
    return float(max(float(r), min_dispersion))


class HierarchicalNBLoss(nn.Module):
    """Hierarchical Negative Binomial loss wrapper for multi-scale block counts."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_means: dict[int, torch.Tensor],
        targets: dict[int, torch.Tensor],
        dispersions: dict[int, float],
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=next(iter(pred_means.values())).device)
        for k in pred_means:
            if k in targets and k in dispersions:
                total_loss = total_loss + negative_binomial_nll_mean_dispersion(
                    targets[k], pred_means[k], dispersions[k], eps=self.eps
                )
        return total_loss
