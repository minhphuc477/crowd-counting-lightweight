from __future__ import annotations

import torch


_MAX_DISPERSION = 1e4


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

    if torch.any(r <= 0) or torch.any(r > _MAX_DISPERSION) or not torch.isfinite(r).all():
        raise ValueError(
            f"Negative-Binomial dispersion parameter r must be in (0, {_MAX_DISPERSION}], got {dispersion}"
        )
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
