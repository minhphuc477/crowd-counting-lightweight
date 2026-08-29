from __future__ import annotations

import torch


def normalize_positive_mass(
    mass: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize positive mass to probability simplex.

    Clamps to ≥0 first, then adds eps smoothing so no entry is exactly 0.
    """
    mass = mass.clamp_min(0.0)
    k = mass.shape[dim]
    denom = mass.sum(dim=dim, keepdim=True)
    return (mass + eps) / (denom + eps * float(k))


def dirichlet_multinomial_nll(
    target_counts: torch.Tensor,
    probs: torch.Tensor,
    concentration: float,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Dirichlet-Multinomial NLL (spec §7.5).

    Always runs in float32 (spec §23.4).  Returns float32 regardless of
    input dtype so autocast cannot truncate lgamma values.

    Args:
        target_counts: integer target counts [..., K]
        probs: predicted probability simplex [..., K]  (need not sum to 1, will be renormalized)
        concentration: kappa (>0)
        valid_mask: boolean mask [...] selecting elements to include in loss
        eps: numerical stability floor
        reduction: "mean" | "sum" | "none"
    """
    if target_counts.shape != probs.shape:
        raise ValueError(
            f"target={target_counts.shape} probs={probs.shape}"
        )

    # Always float32 for lgamma stability
    y = target_counts.to(device=probs.device, dtype=torch.float32)
    p = probs.to(dtype=torch.float32).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)

    kappa = torch.as_tensor(
        concentration,
        device=probs.device,
        dtype=torch.float32,
    ).clamp_min(eps)

    alpha = (p * kappa).clamp_min(eps)
    alpha0 = alpha.sum(dim=-1)
    n = y.sum(dim=-1)

    # Multinomial coefficient (constant wrt params, gradient=0, but included for correct NLL value)
    log_coeff = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0).sum(dim=-1)
    )

    # Global (parent count) term
    log_global = (
        torch.lgamma(alpha0)
        - torch.lgamma(n + alpha0)
    )

    # Local (per-child allocation) term
    log_local = (
        torch.lgamma(y + alpha)
        - torch.lgamma(alpha)
    ).sum(dim=-1)

    nll = -(log_coeff + log_global + log_local)

    if valid_mask is not None:
        nll = nll[valid_mask.bool()]

    if nll.numel() == 0:
        # No valid elements: return a zero tensor that still participates in autograd
        return probs.to(torch.float32).sum() * 0.0

    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    return nll.mean()


def multinomial_nll(
    target_counts: torch.Tensor,
    probs: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Multinomial NLL (DM limit as kappa→∞).

    Always runs in float32 (spec §23.4).  Returns float32.
    """
    # Always float32
    y = target_counts.to(device=probs.device, dtype=torch.float32)
    p = probs.to(dtype=torch.float32).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)

    n = y.sum(dim=-1)

    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0).sum(dim=-1)
        + (y * torch.log(p)).sum(dim=-1)
    )

    nll = -log_prob

    if valid_mask is not None:
        nll = nll[valid_mask.bool()]

    if nll.numel() == 0:
        return probs.to(torch.float32).sum() * 0.0

    return nll.mean()
