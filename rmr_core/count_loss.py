from __future__ import annotations

import torch
import torch.nn.functional as F

_MAX_DISPERSION = 1e4


def negative_binomial_nll_mean_dispersion(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: float | torch.Tensor = 50.0,
    eps: float = 1e-8,
    reduction: str = "mean",
    check_bounds: bool = True,
) -> torch.Tensor:
    """Negative-Binomial NLL with Var(Y) = mu + mu^2 / r evaluated in float32."""
    y = target.to(device=mean.device, dtype=torch.float32)
    mu = mean.to(dtype=torch.float32).clamp_min(eps)
    r = torch.as_tensor(dispersion, device=mean.device, dtype=torch.float32)

    if check_bounds:
        if torch.any(r <= 0) or torch.any(r > _MAX_DISPERSION) or not torch.isfinite(r).all():
            raise ValueError(
                f"Negative-Binomial dispersion parameter r must be in (0, {_MAX_DISPERSION}], got {dispersion}"
            )
        if not torch.isfinite(y).all() or torch.any(y < 0):
            raise ValueError("Negative-Binomial targets must be finite non-negative numbers")
    else:
        y = y.clamp_min(0.0)
        r = r.clamp(1e-6, _MAX_DISPERSION)

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


def count_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "nb",
    dispersion: float = 50.0,
) -> torch.Tensor:
    """Total crop count loss using Negative Binomial NLL or log1p smooth L1."""
    if pred.numel() == 0 or target.numel() == 0:
        return (pred.sum() + target.sum()) * 0.0
    pn = pred.float().sum(dim=(-2, -1)).view(-1) if pred.ndim >= 2 else pred.float().view(-1)
    tn = target.float().sum(dim=(-2, -1)).view(-1) if target.ndim >= 2 else target.float().view(-1)
    if mode == "nb":
        return negative_binomial_nll_mean_dispersion(
            tn, pn, dispersion=dispersion, reduction="mean", check_bounds=False
        )
    elif mode == "log1p":
        return F.smooth_l1_loss(torch.log1p(pn), torch.log1p(tn), reduction="mean", beta=0.2)
    elif mode == "l1":
        return F.l1_loss(pn, tn, reduction="mean")
    elif mode in ("smooth_l1", "huber"):
        return F.smooth_l1_loss(pn, tn, reduction="mean", beta=1.0)
    elif mode == "anscombe":
        g_pn = 2.0 * torch.sqrt(torch.clamp_min(pn, 0.0) + 0.375)
        g_tn = 2.0 * torch.sqrt(torch.clamp_min(tn, 0.0) + 0.375)
        return F.smooth_l1_loss(g_pn, g_tn, reduction="mean", beta=0.2)
    raise ValueError(f"Unsupported count loss mode: {mode}")


global_count_loss = count_magnitude_loss

__all__ = [
    "count_magnitude_loss",
    "global_count_loss",
    "negative_binomial_nll_mean_dispersion",
]
