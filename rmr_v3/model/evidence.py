from __future__ import annotations

from typing import Any

import torch

from rmr_core.operators import RegionSet
from ..regional_head import (
    apply_scale_consistency_gating,
    reliability_from_nb,
)
from .config import RMRv3Config


def extract_regional_evidence(
    cfg: RMRv3Config,
    region_head: Any,
    p4: torch.Tensor,
    p8: torch.Tensor,
    p16: torch.Tensor,
    regions: RegionSet,
    scale_weights: torch.Tensor | None,
    uniform_reliability: bool,
    grid_h: int,
) -> dict[str, Any]:
    """Extract probabilistic regional evidence (count, dispersion, reliability weights)."""
    regional = region_head((p4, p8, p16), regions)

    mu_count = regional["mu_count"]
    dispersion = regional["dispersion"]

    hurdle_pi_for_rel = None
    if cfg.hurdle_head and "hurdle_logit" in regional:
        hurdle_pi_for_rel = torch.sigmoid(regional["hurdle_logit"])

    reliability = reliability_from_nb(
        mu_count,
        dispersion,
        regions,
        mode=cfg.reliability_mode,
        hurdle_pi=hurdle_pi_for_rel,
        rate_std_floor=cfg.reliability_rate_std_floor,
        weight_min=cfg.reliability_weight_min,
        weight_max=cfg.reliability_weight_max,
        normalize_within_scale=cfg.normalize_reliability_within_scale,
        eps=cfg.eps,
    )

    weight = reliability["weight"]
    weight_solver = torch.ones_like(weight) if uniform_reliability else weight
    b_solver_raw = mu_count.detach() if cfg.detach_region_mean_in_solver else mu_count
    b_variance = reliability["count_variance"]

    if cfg.hurdle_head and "hurdle_logit" in regional:
        pi_r = torch.sigmoid(regional["hurdle_logit"].detach())
        # Principled Occupancy Gating:
        # Hurdle is an occupancy classifier P(Y > 0) meant to extinguish background phantom noise.
        # For occupied regions (mu >= 1.0), occupancy is physically certain (P = 1.0), so b_solver
        # retains 100% of mu_count without artificial fractional attenuation.
        # For borderline/empty regions (mu < 1.0), gate smoothly attenuates with pi_r.
        occ_gate = 1.0 - (1.0 - pi_r) * torch.clamp(1.0 - b_solver_raw, min=0.0, max=1.0)
        b_solver = occ_gate * b_solver_raw
        b_variance = occ_gate.square() * b_variance
    else:
        b_solver = b_solver_raw

    if cfg.detach_reliability_in_solver:
        weight_solver = weight_solver.detach()

    should_gate = (
        cfg.pre_solver_scale_gating
        or getattr(cfg, "density_scale_gating", False)
    ) and scale_weights is not None
    if should_gate:
        power = float(cfg.scale_gating_power) if cfg.pre_solver_scale_gating else 0.0
        weight_solver = apply_scale_consistency_gating(
            weight_solver,
            regions,
            scale_weights,
            power=power,
            eps=cfg.eps,
            perspective_horizon_gate=cfg.perspective_horizon_gate,
            horizon_cutoff=float(cfg.horizon_cutoff),
            region_sizes_px=cfg.region_sizes_px,
            grid_h=grid_h,
            regional_rate=regional["rate"],
            density_scale_gating=getattr(cfg, "density_scale_gating", False),
            density_scale_tau=float(getattr(cfg, "density_scale_tau", 0.15)),
        )

    return {
        "mu_count": mu_count,
        "dispersion": dispersion,
        "log_dispersion": regional["log_dispersion"],
        "rate": regional["rate"],
        "b_solver": b_solver,
        "b_variance": b_variance,
        "weight": weight,
        "weight_solver": weight_solver,
        "precision": reliability["precision"],
        "rate_variance": reliability["rate_variance"],
        "count_variance": reliability["count_variance"],
        "hurdle_logit": regional.get("hurdle_logit", None),
    }
