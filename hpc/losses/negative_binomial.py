from __future__ import annotations

import torch


def negative_binomial_nll_mean_dispersion(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: float | torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative Binomial NLL.

    Always runs in float32 (spec §23.4).  Returns float32 regardless of
    input dtype so that AMP autocast cannot truncate the lgamma values.
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
    return float(r.clamp_min(min_dispersion).item())
