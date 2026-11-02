"""Error-Bounded Geometric-Binning Loss (EB-GB Loss).

Solves the NLL gradient starvation crisis:
    dL/dmu ~ (mu - y) / (mu + mu^2 / r)
which severely penalizes dense crowd samples by up to 4,080x less than sparse background.

Under the EB-GB objective, continuous ground truth count y_R is mapped to a soft
convex combination between adjacent geometric bin centers:
    E_{Y_soft}[b] == y_R (Exact Expected Value Interpolation).

The resulting Cross-Entropy gradient is strictly bounded:
    ||nabla_z L_{CE}||_infty <= 1.0 (Zero starvation across [0, 450] persons).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rmr_core.operators import RegionSet


def compute_soft_bin_targets(
    target_count: torch.Tensor,
    bin_centers: torch.Tensor,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Project continuous ground truth regional counts onto geometric bin simplex.

    Args:
        target_count: [B, M] continuous non-negative crowd count per region.
        bin_centers: [K] strictly increasing bin lattice points (e.g. 16 or 18 bins).
        smoothing: Mean-preserving geometric label smoothing factor (0.0 to 0.1).

    Returns:
        target_probs: [B, M, K] soft categorical distribution satisfying:
                      sum_k target_probs[..., k] * bin_centers[k] == target_count.
    """
    b_sz, m_boxes = target_count.shape
    num_bins = bin_centers.shape[0]
    dtype = target_count.dtype
    device = target_count.device

    bins = bin_centers.to(device=device, dtype=dtype)
    y_clamped = target_count.clamp(min=float(bins[0]), max=float(bins[-1]))

    # Find right-hand bin index: bins[idx - 1] <= y < bins[idx]
    # bucketize returns index in [1, K-1] for y in [bins[0], bins[-1]]
    idx_right = torch.bucketize(y_clamped, bins, right=True).clamp(1, num_bins - 1)
    idx_left = idx_right - 1

    b_left = bins[idx_left]
    b_right = bins[idx_right]
    delta = (b_right - b_left).clamp_min(1e-6)

    # Linear barycentric weights
    w_right = (y_clamped - b_left) / delta
    w_left = 1.0 - w_right

    # Scatter weights into [B, M, K] simplex tensor
    out = torch.zeros((b_sz, m_boxes, num_bins), device=device, dtype=dtype)
    out.scatter_add_(dim=-1, index=idx_left.unsqueeze(-1), src=w_left.unsqueeze(-1))
    out.scatter_add_(dim=-1, index=idx_right.unsqueeze(-1), src=w_right.unsqueeze(-1))

    if smoothing > 0.0:
        valid = (idx_left > 0) & (idx_right < num_bins - 1)
        if valid.any():
            b_l_prev = bins[(idx_left - 1).clamp_min(0)]
            b_r_next = bins[(idx_right + 1).clamp_max(num_bins - 1)]
            d_l = (b_left - b_l_prev).clamp_min(1e-6)
            d_r = (b_r_next - b_right).clamp_min(1e-6)
            eps_l = torch.where(valid, (smoothing * w_left).clamp_max(0.20 * w_left), torch.zeros_like(w_left))
            eps_r = (eps_l * (d_l / d_r)).clamp_max(0.20 * w_right)
            eps_l = torch.where(valid, eps_r * (d_r / d_l), torch.zeros_like(eps_l))

            idx_l_prev = (idx_left - 1).clamp_min(0)
            idx_r_next = (idx_right + 1).clamp_max(num_bins - 1)
            out.scatter_add_(dim=-1, index=idx_l_prev.unsqueeze(-1), src=eps_l.unsqueeze(-1))
            out.scatter_add_(dim=-1, index=idx_left.unsqueeze(-1), src=(-eps_l).unsqueeze(-1))
            out.scatter_add_(dim=-1, index=idx_r_next.unsqueeze(-1), src=eps_r.unsqueeze(-1))
            out.scatter_add_(dim=-1, index=idx_right.unsqueeze(-1), src=(-eps_r).unsqueeze(-1))

    return out


def compute_ebgb_loss(
    bin_logits: torch.Tensor,
    target_count: torch.Tensor,
    bin_centers: torch.Tensor,
    hurdle_logit: torch.Tensor | None = None,
    lambda_hurdle: float = 1.0,
    smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Error-Bounded Geometric Binning Loss.

    Args:
        bin_logits: [B, M, K] unnormalized logits from UnsaturableRegionalEvidenceHead.
        target_count: [B, M] continuous ground truth counts per region.
        bin_centers: [K] geometric bin center locations.
        hurdle_logit: [B, 1, M] or [B, M] logit of crowd occupancy gate.
        lambda_hurdle: weight of hurdle occupancy binary cross entropy.
        smoothing: Mean-preserving geometric label smoothing factor.

    Returns:
        total_loss: scalar tensor.
        loss_dict: dictionary of individual loss components.
    """
    if target_count.ndim == 3:
        target_count = target_count.squeeze(1)

    # 1. Soft categorical cross entropy
    y_soft = compute_soft_bin_targets(target_count, bin_centers, smoothing=smoothing)
    log_probs = F.log_softmax(bin_logits, dim=-1)
    ce_loss = -(y_soft * log_probs).sum(dim=-1).mean()

    # 2. Hurdle binary cross entropy (if enabled)
    hurdle_loss = torch.zeros_like(ce_loss)
    if hurdle_logit is not None and lambda_hurdle > 0.0:
        h_logit = hurdle_logit.squeeze(1) if hurdle_logit.ndim == 3 else hurdle_logit
        t_occupancy = (target_count > 0.0).to(dtype=bin_logits.dtype)
        hurdle_loss = F.binary_cross_entropy_with_logits(h_logit, t_occupancy)

    total_loss = ce_loss + float(lambda_hurdle) * hurdle_loss

    loss_dict = {
        "ebgb_total": total_loss,
        "ebgb_ce": ce_loss,
        "ebgb_hurdle": hurdle_loss,
    }
    return total_loss, loss_dict


def scale_balanced_regional_ebgb_loss(
    bin_logits: torch.Tensor,
    target_region: torch.Tensor,
    bin_centers: torch.Tensor,
    regions: RegionSet,
    hurdle_logit: torch.Tensor | None = None,
    lambda_hurdle: float = 0.0,
    smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute scale-balanced Error-Bounded Geometric Binning Loss.

    Averages categorical Cross-Entropy across individual scale dictionaries
    (e.g. 32px, 64px, 128px, and full image if present), ensuring uniform multiscale regularisation and
    strictly bounded gradients ||nabla_z L||_infty <= 1.0.
    """
    if target_region.ndim == 1:
        target_region = target_region.unsqueeze(0)
    elif target_region.ndim == 3 and target_region.shape[1] == 1:
        target_region = target_region.squeeze(1)
    if bin_logits.ndim == 2:
        bin_logits = bin_logits.unsqueeze(0)
    if hurdle_logit is not None:
        if hurdle_logit.ndim == 1:
            hurdle_logit = hurdle_logit.unsqueeze(0)
        elif hurdle_logit.ndim == 3 and hurdle_logit.shape[1] == 1:
            hurdle_logit = hurdle_logit.squeeze(1)

    scale_losses: list[torch.Tensor] = []
    scale_ce: list[torch.Tensor] = []
    scale_hurdle: list[torch.Tensor] = []
    num_scales = int(getattr(regions, "num_scales", 3))

    start_sid = -1 if (regions.scale_id == -1).any() else 0
    for sid in range(start_sid, num_scales):
        mask = (regions.scale_id == sid)
        t_count = mask.float().sum()
        if t_count <= 0:
            continue

        z_s = bin_logits[:, mask, :]
        t_s = target_region[:, mask]
        h_s = hurdle_logit[:, mask] if hurdle_logit is not None else None

        loss_s, d_s = compute_ebgb_loss(
            bin_logits=z_s,
            target_count=t_s,
            bin_centers=bin_centers,
            hurdle_logit=h_s,
            lambda_hurdle=lambda_hurdle,
            smoothing=smoothing,
        )
        scale_losses.append(loss_s)
        scale_ce.append(d_s["ebgb_ce"])
        scale_hurdle.append(d_s["ebgb_hurdle"])

    if not scale_losses:
        return compute_ebgb_loss(
            bin_logits=bin_logits,
            target_count=target_region,
            bin_centers=bin_centers,
            hurdle_logit=hurdle_logit,
            lambda_hurdle=lambda_hurdle,
            smoothing=smoothing,
        )

    tot_loss = torch.stack(scale_losses).mean()
    tot_ce = torch.stack(scale_ce).mean()
    tot_hurdle = torch.stack(scale_hurdle).mean()
    return tot_loss, {
        "ebgb_total": tot_loss,
        "ebgb_ce": tot_ce,
        "ebgb_hurdle": tot_hurdle,
    }

