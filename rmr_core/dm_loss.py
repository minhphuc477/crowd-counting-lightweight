"""Dirichlet-Multinomial spatial allocation losses and block summation operators.

Provides:
1. block_sum_2d: Fast non-overlapping 2-D block summation via reshape.
2. probs_from_positive_mass: Laplace-smoothed Dirichlet allocation probability distribution.
3. dm_nll_none: Exact Dirichlet-Multinomial negative log-likelihood.
4. flat_dm_block_loss: Arbitrary block-size Flat Dirichlet-Multinomial allocation loss.
5. multiscale_dm_loss: Multi-Scale Dirichlet-Multinomial allocation loss across granularities.
6. flat_dm16_loss: Canonical 16px Dirichlet-Multinomial loss.
"""
from __future__ import annotations

import torch


def block_sum_2d(x: torch.Tensor, k: int = 4, strict: bool = True) -> torch.Tensor:
    """Sum non-overlapping k x k blocks via reshape.

    If strict=True, raises ValueError when height or width is not divisible by k.
    If strict=False, trailing edge cells are trimmed.
    """
    had_channel = x.ndim == 4
    if not had_channel:
        x = x.unsqueeze(1)
    b, c, h, w = x.shape
    if h % k != 0 or w % k != 0:
        if strict:
            raise ValueError(
                f"Input spatial dimensions ({h}, {w}) must be divisible by block size k={k}"
            )
        h_trim = (h // k) * k
        w_trim = (w // k) * k
        x = x[:, :, :h_trim, :w_trim]
        h, w = h_trim, w_trim
    out = x.reshape(b, c, h // k, k, w // k, k).sum((3, 5))
    return out if had_channel else out.squeeze(1)


def probs_from_positive_mass(mass: torch.Tensor, tiny: float = 0.05) -> torch.Tensor:
    """Compute strictly positive probability distribution over blocks via Bayesian Laplace smoothing.

    Guarantees sum(pi) == 1.0, active gradients on all blocks without clamp_min cutoff,
    and bounded gradients (< 65504) that never overflow FP16 under AMP:
        pi_k = (mass_k + prior) / (sum(mass) + eps)
    """
    mass_f = mass.float()
    k_blocks = mass_f.shape[-1]
    eps = max(float(tiny), 0.01)
    prior = eps / float(max(k_blocks, 1))
    return (mass_f + prior) / (mass_f.sum(dim=-1, keepdim=True) + eps)


def dm_nll_none(
    y: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-8, check_bounds: bool = False
) -> torch.Tensor:
    """Dirichlet-Multinomial NLL; empty parents contribute exactly zero."""
    y = y.float()
    alpha = alpha.float().clamp_min(eps)
    if check_bounds and torch.any(y < 0):
        raise ValueError("Dirichlet-Multinomial targets must be non-negative")
    n = y.sum(dim=-1)
    alpha0 = alpha.sum(dim=-1)
    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0).sum(dim=-1)
        + torch.lgamma(alpha0)
        - torch.lgamma(n + alpha0)
        + (torch.lgamma(y + alpha) - torch.lgamma(alpha)).sum(dim=-1)
    )
    return torch.where(n == 0, torch.zeros_like(n), -log_prob)


def flat_dm_block_loss(
    pred_map: torch.Tensor,
    target_map: torch.Tensor,
    block_px: int = 16,
    kappa: float = 20.0,
    stride: int = 4,
    eps: float = 1e-8,
    normalize_by_count: bool = True,
    strict: bool = True,
    auto_scale_kappa: bool = True,
    norm_mode: str = "count",
    ref_count: float = 100.0,
) -> torch.Tensor:
    """Flat Dirichlet-Multinomial allocation loss on arbitrary block_px sizes."""
    if block_px % stride != 0:
        raise ValueError(f"block_px ({block_px}) must be divisible by stride ({stride})")

    k = block_px // stride
    h_pred, w_pred = pred_map.shape[-2:]
    h_tgt, w_tgt = target_map.shape[-2:]

    if h_pred % k != 0 or w_pred % k != 0:
        if strict:
            raise ValueError(
                f"FlatDM{block_px} requires grid dimensions divisible by block size k={k}: grid {pred_map.shape[-2:]}"
            )
    if h_tgt % k != 0 or w_tgt % k != 0:
        if strict:
            raise ValueError(
                f"FlatDM{block_px} requires grid dimensions divisible by block size k={k}: grid {target_map.shape[-2:]}"
            )

    if not strict and (h_pred < k or w_pred < k or h_tgt < k or w_tgt < k):
        return pred_map.new_tensor(0.0)

    if pred_map.numel() == 0 or target_map.numel() == 0:
        return (pred_map.float().sum() + target_map.float().sum()) * 0.0

    pred_block = block_sum_2d(pred_map.float(), k=k, strict=strict).flatten(1)
    target_block = block_sum_2d(target_map.float(), k=k, strict=strict).flatten(1)

    if pred_block.shape[-1] == 0:
        return pred_map.new_tensor(0.0)

    pi = probs_from_positive_mass(pred_block, tiny=eps)
    eff_kappa = float(kappa) * ((float(block_px) / 16.0) ** 2) if auto_scale_kappa else float(kappa)
    alpha = eff_kappa * pi

    per_image = dm_nll_none(target_block, alpha, eps=eps)
    if norm_mode == "head_balanced":
        count = target_block.sum(-1).clamp_min(1.0)
        per_image = per_image * (count / float(max(ref_count, 1.0)))
    elif norm_mode == "unnorm_ref":
        per_image = per_image / float(max(ref_count, 1.0))
    elif norm_mode == "none" or not normalize_by_count:
        pass
    else:
        count = target_block.sum(-1).clamp_min(1.0)
        per_image = per_image / count

    return per_image.mean()


def multiscale_dm_loss(
    pred_map: torch.Tensor,
    target_map: torch.Tensor,
    block_sizes_px: tuple[int, ...] = (16, 32, 64),
    weights: tuple[float, ...] | None = None,
    kappas: tuple[float, ...] | None = None,
    stride: int = 4,
    eps: float = 1e-8,
    normalize_by_count: bool = True,
    strict: bool = True,
    return_components: bool = False,
    norm_mode: str = "count",
    ref_count: float = 100.0,
) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Multi-Scale Dirichlet-Multinomial allocation loss across independent block granularities.

    Computes a weighted sum of Flat Dirichlet-Multinomial partition losses across multiple
    block scales (e.g. 16px, 32px, 64px), capturing local-to-regional count allocation
    without imposing a strict conditional tree-factored probability structure.
    """
    if not block_sizes_px:
        raise ValueError("block_sizes_px must not be empty")

    if weights is None:
        if len(block_sizes_px) == 3 and block_sizes_px == (16, 32, 64):
            weights = (0.50, 0.30, 0.20)
        else:
            weights = tuple(1.0 / len(block_sizes_px) for _ in block_sizes_px)
    if kappas is None:
        kappas = tuple(20.0 for _ in block_sizes_px)

    if not (len(block_sizes_px) == len(weights) == len(kappas)):
        raise ValueError(
            f"length mismatch: block_sizes_px={len(block_sizes_px)}, weights={len(weights)}, kappas={len(kappas)}"
        )

    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("sum(weights) must be > 0")

    terms = []
    components: dict[int, torch.Tensor] = {}
    for block_px, w, kappa in zip(block_sizes_px, weights, kappas):
        if w == 0:
            continue
        li = flat_dm_block_loss(
            pred_map,
            target_map,
            block_px=int(block_px),
            kappa=float(kappa),
            stride=stride,
            eps=eps,
            normalize_by_count=normalize_by_count,
            strict=strict,
            norm_mode=norm_mode,
            ref_count=ref_count,
        )
        components[int(block_px)] = li
        terms.append((float(w) / wsum) * li)

    total = torch.stack(terms).sum()
    if return_components:
        return total, components
    return total


# Backward compatibility alias
hierarchical_dm_loss = multiscale_dm_loss


def flat_dm16_loss(
    pred_map: torch.Tensor,
    target_map: torch.Tensor,
    kappa: float = 20.0,
    stride: int = 4,
    eps: float = 1e-8,
    normalize_by_count: bool = True,
    strict: bool = True,
    auto_scale_kappa: bool = True,
    norm_mode: str = "count",
    ref_count: float = 100.0,
) -> torch.Tensor:
    """Flat Dirichlet-Multinomial-16 allocation loss on 16px blocks (backward compatible)."""
    return flat_dm_block_loss(
        pred_map,
        target_map,
        block_px=16,
        kappa=kappa,
        stride=stride,
        eps=eps,
        normalize_by_count=normalize_by_count,
        strict=strict,
        auto_scale_kappa=auto_scale_kappa,
        norm_mode=norm_mode,
        ref_count=ref_count,
    )
