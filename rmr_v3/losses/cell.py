"""Cell-level spatial allocation losses for continuous Radon measure recovery.

Provides:
1. count_harmonized_cell_loss: Normalized foreground/background allocation with fractional count scaling.
2. count_invariant_cell_loss: Count-Invariant Two-Stream cell loss (CI-Cell v2) normalized by peak count.
3. mass_weighted_cell_loss: Density-weighted Smooth L1 emphasizing high-mass cluster cells.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mass_weighted_cell_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-3,
    alpha: float = 1.0,
    gamma: float = 1.0,
    stride: int = 4,
) -> torch.Tensor:
    """Mass-weighted cell allocation loss (RMR-v8 Stage 2 / RMR-v11).

    Weights each cell by its fractional mass p(x) = t(x) / sum(t), emphasizing
    crowded foreground cells over empty background cells while retaining L1 scale.
    """
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 2:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.ndim == 3:
        y = y.unsqueeze(1)

    work_dtype = y.dtype if y.dtype in (torch.float32, torch.float64) else torch.float32
    y_f = y.to(dtype=work_dtype)
    t_f = target.to(dtype=work_dtype)
    if y_f.numel() == 0 or t_f.numel() == 0:
        return (y_f.sum() + t_f.sum()) * 0.0

    total_mass = t_f.sum(dim=(-2, -1), keepdim=True)
    p = torch.where(total_mass > float(eps), t_f / total_mass.clamp_min(float(eps)), torch.zeros_like(t_f))

    if abs(float(gamma) - 1.0) > 1e-5:
        p_pow = p.pow(float(gamma))
        sum_p_pow = p_pow.sum(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
        p = p_pow / sum_p_pow

    hw = float(t_f.shape[-2] * t_f.shape[-1])
    raw_weight = 1.0 + float(alpha) * hw * p

    norm_factor = raw_weight.mean(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    weights = raw_weight / norm_factor

    per_pixel = F.smooth_l1_loss(y_f, t_f, beta=float(beta), reduction="none")
    return (weights * per_pixel).mean()


def count_invariant_cell_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 2.0,
    tau_head: float = 0.08,
    eps: float = 1e-4,
    stride: int = 4,
) -> torch.Tensor:
    """Count-Invariant Two-Stream Cell Loss v2 (CI-Cell v2).

    Corrected formulation for integer-domain density maps where each cell
    stores the number of people (0, 1, 2, 3, 4...), NOT a Gaussian amplitude.

    Weighting strategy:
        fg_fraction(x) = t(x) / max(per_image_max, 1)  in [0, 1]
        W(x) = 1 + (alpha - 1) * fg_fraction(x)

    Guarantees O(1) gradient per head across arbitrary density variations:
    - Weight is normalized by local peak count, not global sum N.
    - Cells in dense images receive equal relative emphasis to sparse images.
    - Background cells (t=0) retain W=1.0 for false-alarm suppression.
    """
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 2:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.ndim == 3:
        y = y.unsqueeze(1)

    work_dtype = y.dtype if y.dtype in (torch.float32, torch.float64) else torch.float32
    y_f = y.to(dtype=work_dtype)
    t_f = target.to(dtype=work_dtype)
    if y_f.numel() == 0 or t_f.numel() == 0:
        return (y_f.sum() + t_f.sum()) * 0.0

    t_peak = t_f.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    fg_fraction = (t_f / t_peak).clamp(0.0, 1.0)

    weight = 1.0 + (float(alpha) - 1.0) * fg_fraction
    per_pixel = F.smooth_l1_loss(y_f, t_f, beta=float(beta), reduction="none")
    return (weight * per_pixel).mean()


def count_harmonized_cell_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-4,
    gamma: float = 1.25,
    fg_ratio: float = 0.67,
    stride: int = 4,
    norm_power: float = 1.0,
    norm_ref: float = 100.0,
) -> torch.Tensor:
    """Count-Harmonized Cell Allocation Loss with optional fractional normalization.

    Harmonizes gradient magnitude between foreground heads and background empty cells:
    - Background cells receive fixed loss weight (1 - fg_ratio).
    - Foreground cells receive normalized mass weight: w_pos / denom.
    - Fractional count power (norm_power in [0, 1]) controls gradient starvation
      decay rate O(1 / N^norm_power) across high-count images.
    """
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 2:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.ndim == 3:
        y = y.unsqueeze(1)

    work_dtype = y.dtype if y.dtype in (torch.float32, torch.float64) else torch.float32
    y_f = y.to(dtype=work_dtype)
    t_f = target.to(dtype=work_dtype)
    if y_f.numel() == 0 or t_f.numel() == 0:
        return (y_f.sum() + t_f.sum()) * 0.0

    per_pixel = F.smooth_l1_loss(y_f, t_f, beta=float(beta), reduction="none")
    b_sz = y_f.shape[0]
    sample_losses = []
    for i in range(b_sz):
        tgt_i, per_i = t_f[i, 0], per_pixel[i, 0]
        pos_mask, neg_mask = tgt_i > 0, ~(tgt_i > 0)
        neg_loss = per_i[neg_mask].mean() if neg_mask.any() else per_i.new_tensor(0.0)
        if pos_mask.any():
            pos_t, pos_per = tgt_i[pos_mask], per_i[pos_mask]
            w_pos = pos_t.pow(float(gamma)) if abs(float(gamma) - 1.0) > 1e-5 else pos_t
            w_sum = w_pos.sum().clamp_min(float(eps))
            denom = (
                w_sum
                if abs(float(norm_power) - 1.0) <= 1e-5
                else (w_sum.pow(float(norm_power)) * (float(norm_ref) ** (1.0 - float(norm_power)))).clamp_min(float(eps))
            )
            pos_loss = (w_pos / denom * pos_per).sum()
            comb_loss = float(1.0 - fg_ratio) * neg_loss + float(fg_ratio) * pos_loss
            sample_losses.append(comb_loss if neg_mask.any() else pos_loss)
        else:
            sample_losses.append(neg_loss)

    return torch.stack(sample_losses).mean()


def compute_cell_loss(
    density_map: torch.Tensor,
    target_float: torch.Tensor,
    cfg: Any,
) -> torch.Tensor:
    """Dispatch cell allocation loss computation according to cfg.cell_loss_mode."""
    from rmr_core.losses import balanced_smooth_l1

    stride = int(getattr(cfg, "output_stride", 4))
    if cfg.cell_loss_mode in ("count_invariant", "ci_cell"):
        return count_invariant_cell_loss(
            density_map,
            target_float,
            beta=cfg.cell_beta,
            alpha=float(getattr(cfg, "cell_alpha", 2.0)),
            stride=stride,
        )
    if cfg.cell_loss_mode == "mass_weighted":
        return mass_weighted_cell_loss(
            density_map,
            target_float,
            beta=cfg.cell_beta,
            eps=cfg.cell_mass_weight_eps,
            alpha=float(cfg.cell_mass_weight_alpha),
            gamma=float(cfg.cell_mass_weight_gamma),
            stride=stride,
        )
    if cfg.cell_loss_mode == "count_harmonized":
        return count_harmonized_cell_loss(
            density_map,
            target_float,
            beta=cfg.cell_beta,
            eps=cfg.cell_mass_weight_eps,
            gamma=float(cfg.cell_mass_weight_gamma),
            fg_ratio=float(getattr(cfg, "cell_fg_ratio", 0.67)),
            stride=stride,
            norm_power=float(getattr(cfg, "cell_norm_power", 1.0)),
            norm_ref=float(getattr(cfg, "cell_norm_ref", 100.0)),
        )
    return balanced_smooth_l1(density_map, target_float, beta=cfg.cell_beta, stride=stride)

