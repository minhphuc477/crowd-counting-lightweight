from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_barzilai_borwein_step(
    z_state: torch.Tensor,
    prev_y: torch.Tensor,
    field: torch.Tensor,
    prev_field: torch.Tensor,
    effective_omega: float,
    use_alternating_bb: bool = False,
    iter_idx: int = 1,
    bb_clamp_min: float = 0.2,
    bb_clamp_max: float = 2.0,
) -> torch.Tensor:
    """Compute adaptive spectral step size via Barzilai-Borwein (BB) formulation."""
    s_diff = (z_state - prev_y).float()
    r_diff = (field - prev_field).float()
    dot_sr = (s_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True)
    norm_r_sq = (r_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True) + 1e-6
    norm_s_sq = (s_diff * s_diff).sum(dim=(-3, -2, -1), keepdim=True) + 1e-6

    if use_alternating_bb and (iter_idx % 2 == 1):
        # BB-2: Inverse Rayleigh quotient alpha_2 = ||s||^2 / <s, r>
        omega_candidate = torch.where(
            dot_sr > 1e-7,
            norm_s_sq / dot_sr.clamp_min(1e-7),
            torch.full_like(dot_sr, effective_omega),
        ).detach()
    else:
        # BB-1: Standard Rayleigh quotient alpha_1 = <s, r> / ||r||^2
        omega_candidate = torch.where(
            dot_sr > 0.0,
            dot_sr / norm_r_sq,
            torch.full_like(dot_sr, effective_omega),
        ).detach()

    return torch.clamp(
        omega_candidate,
        min=float(bb_clamp_min) * effective_omega,
        max=float(bb_clamp_max) * effective_omega,
    )


def compute_adaptive_relaxation_factor(
    z_state: torch.Tensor,
    effective_relax_thresh: float,
    effective_relax_scale: float,
    adaptive_relax_sparse: float = 0.70,
    adaptive_relax_dense_boost: float = 0.50,
) -> torch.Tensor:
    """Compute density-adaptive over-relaxation multiplier (RMR-v20)."""
    z_smooth = F.avg_pool2d(
        z_state.float(), kernel_size=5, stride=1, padding=2, count_include_pad=False
    )
    scale = max(float(effective_relax_scale), 1e-6)
    gate_dense = torch.sigmoid((z_smooth - float(effective_relax_thresh)) / scale)
    boost = 1.0 - float(adaptive_relax_sparse) + float(adaptive_relax_dense_boost)
    return float(adaptive_relax_sparse) + boost * gate_dense
