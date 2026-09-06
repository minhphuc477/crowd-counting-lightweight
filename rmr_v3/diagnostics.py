"""rmr_v3/diagnostics.py — Diagnostic utilities for RMR-v3 (Section 32).

Reliability mechanism gate:
    Pearson/Spearman rho of (rate_variance, abs_rate_error)
    Must be significantly positive before claiming reliability weighting works.

Additional diagnostics:
    - Solver energy monotonicity check (same as RMR-v2 audit)
    - Regional disagreement reduction (|AY_T - b| vs |AY_0 - b|)
    - Per-density-bin MAE breakdown (sparse / moderate / dense)
    - Weight distribution summary (mean, std, min, max of w_R)
"""
from __future__ import annotations

from typing import Optional
import torch
import numpy as np

from rmr_count.operators import RegionSet, regional_sum
from .losses import reliability_correlation_stats


# ---------------------------------------------------------------------------
# Solver energy monitoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def solver_energy(
    y: torch.Tensor,       # [B, 1, H, W]
    b_region: torch.Tensor,  # [B, 1, M]
    regions: RegionSet,
    w_region: Optional[torch.Tensor] = None,  # [B, 1, M] if weighted
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted least-squares energy: E(Y) = sum_R w_R * (A_R Y - b_R)^2 / (2 * |R|^2).

    For uniform weights (V3-A), this reduces to the unweighted SIRT energy.
    """
    ay = regional_sum(y, regions.boxes).unsqueeze(1).to(torch.float32)  # [B, 1, M]
    b_f32 = b_region.to(torch.float32)
    residual_sq = (ay - b_f32) ** 2

    if w_region is not None:
        w_f32 = w_region.to(torch.float32)
    else:
        w_f32 = torch.ones_like(residual_sq)

    area = regions.area.to(torch.float32).view(1, 1, -1).clamp_min(1.0)
    energy = (w_f32 * residual_sq / area.pow(2)).sum(dim=-1)  # [B, 1]
    return energy.squeeze(1)  # [B]


@torch.no_grad()
def check_energy_monotone(
    iterates: list[torch.Tensor],
    b_region: torch.Tensor,
    regions: RegionSet,
    w_region: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Check E(Y_0) > E(Y_1) > ... > E(Y_T) across solver iterates.

    Returns:
        monotone_frac: fraction of batch samples with strictly monotone energy
        energy_reduction: (E_0 - E_T) / E_0 relative reduction
    """
    energies = [
        solver_energy(y, b_region, regions, w_region).cpu()
        for y in iterates
    ]
    # Check monotone: E[t] > E[t+1] for all t
    T = len(energies)
    is_mono = torch.ones(energies[0].shape[0], dtype=torch.bool)
    for t in range(T - 1):
        is_mono &= energies[t] > energies[t + 1]

    e0 = energies[0].float()
    et = energies[-1].float()
    eps = 1e-8
    reduction = ((e0 - et) / e0.clamp_min(eps)).mean().item()

    return {
        "monotone_frac": is_mono.float().mean().item(),
        "energy_reduction": reduction,
        "n_iterates": T,
    }


# ---------------------------------------------------------------------------
# Regional disagreement reduction
# ---------------------------------------------------------------------------
@torch.no_grad()
def regional_disagreement_stats(
    iterates: list[torch.Tensor],  # [Y0, Y1, ..., Y_T]
    b_region: torch.Tensor,        # [B, 1, M]
    regions: RegionSet,
) -> dict[str, float]:
    """Compute |AY_t - b|_1 for each iterate; report reduction from Y0 to Y_T."""
    disags = []
    for y in iterates:
        ay = regional_sum(y, regions.boxes).unsqueeze(1).to(torch.float32)
        disag = (ay - b_region.to(torch.float32)).abs().sum(dim=-1).mean().item()
        disags.append(disag)

    return {
        "disag_y0": disags[0],
        "disag_yT": disags[-1],
        "disag_reduction": (disags[0] - disags[-1]) / max(disags[0], 1e-8),
        "disag_by_iterate": disags,
    }


# ---------------------------------------------------------------------------
# Reliability weight distribution
# ---------------------------------------------------------------------------
@torch.no_grad()
def weight_distribution_stats(
    w_region: torch.Tensor,   # [B, 1, M]
    regions: RegionSet,
) -> dict[str, float]:
    """Summary statistics of reliability weights, per scale and overall."""
    w = w_region.squeeze(1).float()  # [B, M]
    out = {
        "w_mean": w.mean().item(),
        "w_std": w.std().item(),
        "w_min": w.min().item(),
        "w_max": w.max().item(),
    }
    for sid in torch.unique(regions.scale_id).tolist():
        mask = regions.scale_id == int(sid)
        ws = w[:, mask]
        out[f"w_mean_scale{int(sid)}"] = ws.mean().item()
        out[f"w_std_scale{int(sid)}"] = ws.std().item()
    return out


# ---------------------------------------------------------------------------
# Per-density-bin MAE
# ---------------------------------------------------------------------------
@torch.no_grad()
def density_bin_mae(
    pred_y: torch.Tensor,    # [B, 1, H, W]
    target_y: torch.Tensor,  # [B, 1, H, W]
    bins: tuple[float, float] = (25.0, 200.0),
) -> dict[str, float]:
    """Compute MAE per density bin: sparse / moderate / dense.

    Bins defined by total GT count per image:
        sparse:   N < bins[0]
        moderate: bins[0] <= N < bins[1]
        dense:    N >= bins[1]
    """
    pred_count = pred_y.sum(dim=(-1, -2, -3)).float().cpu()  # [B]
    gt_count = target_y.sum(dim=(-1, -2, -3)).float().cpu()  # [B]
    ae = (pred_count - gt_count).abs()                        # [B]

    lo, hi = bins
    sparse_mask = gt_count < lo
    mod_mask = (gt_count >= lo) & (gt_count < hi)
    dense_mask = gt_count >= hi

    def _safe_mean(t: torch.Tensor, mask: torch.Tensor) -> float:
        return t[mask].mean().item() if mask.any() else float("nan")

    return {
        "mae_sparse":   _safe_mean(ae, sparse_mask),
        "mae_moderate": _safe_mean(ae, mod_mask),
        "mae_dense":    _safe_mean(ae, dense_mask),
        "n_sparse":     int(sparse_mask.sum().item()),
        "n_moderate":   int(mod_mask.sum().item()),
        "n_dense":      int(dense_mask.sum().item()),
    }


# ---------------------------------------------------------------------------
# Full diagnostic report
# ---------------------------------------------------------------------------
@torch.no_grad()
def full_diagnostic_report(
    outputs: dict,
    target_y: torch.Tensor,
    bins: tuple[float, float] = (25.0, 200.0),
) -> dict:
    """Collect all V3 diagnostics in a single call.

    Args:
        outputs: dict from RMRv3.forward()
        target_y: [B, 1, H, W] GT density map
    Returns:
        Nested dict with all diagnostic metrics.
    """
    regions: RegionSet = outputs["regions"]
    iterates: list = outputs["iterates"]
    b_region = outputs["b_region"]
    w_region = outputs["w_region"]
    y_final = outputs["y"]

    report = {}

    # Reliability correlation gate
    report["reliability"] = reliability_correlation_stats(
        w_region, b_region, target_y, regions
    )

    # Energy monotonicity
    report["energy"] = check_energy_monotone(iterates, b_region, regions, w_region)

    # Regional disagreement
    report["disagreement"] = regional_disagreement_stats(iterates, b_region, regions)

    # Weight distribution
    report["weights"] = weight_distribution_stats(w_region, regions)

    # Per-density MAE
    report["density_bins"] = density_bin_mae(y_final, target_y, bins)

    return report
