"""Spatial allocation losses (Bayesian, Dirichlet-Multinomial, FIDT, and Optimal Transport).

Provides:
1. compute_single_allocation: Computes the spatial allocation loss for a single density map (y or y0).
2. route_allocation_loss: Dispatches allocation supervision across y, y0, or symmetric dual.
"""
from __future__ import annotations

import torch

from rmr_core.losses import flat_dm_block_loss, multiscale_dm_loss
from .config import RMRv3LossConfig
from .fidt import canonical_fidt_loss
from .point_supervision import bayesian_loss, fidt_loss, sinkhorn_ot_loss


def compute_single_allocation(
    inp: torch.Tensor,
    target_float: torch.Tensor,
    points: list[torch.Tensor] | None,
    cfg: RMRv3LossConfig,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Compute spatial allocation loss for a single density map according to cfg."""
    comps: dict[int, torch.Tensor] = {}
    stride = int(getattr(cfg, "output_stride", 4))

    if cfg.allocation_loss_type in ("dual_bayesian_dm16", "bayesian"):
        if cfg.allocation_loss_type == "dual_bayesian_dm16":
            b_px = int(getattr(cfg, "dm_block_px", 16))
            if float(cfg.lambda_bayesian) > 0.0:
                s_max = float(getattr(cfg, "bayesian_sigma_max", 4.0))
                l_bay = bayesian_loss(
                    inp,
                    points,
                    sigma=cfg.bayesian_sigma,
                    background_ratio=cfg.bayesian_background_ratio,
                    stride=stride,
                    norm_mode=getattr(cfg, "bayesian_norm_mode", "canonical"),
                    canonical_background=getattr(cfg, "bayesian_canonical_bg", True),
                    adaptive_sigma=getattr(cfg, "bayesian_adaptive_sigma", False),
                    sigma_min=getattr(cfg, "bayesian_sigma_min", 2.0),
                    sigma_max=s_max,
                )
            else:
                l_bay = torch.zeros((), device=inp.device, dtype=inp.dtype)

            if float(cfg.lambda_flat_dm16) > 0.0:
                l_dm = flat_dm_block_loss(
                    inp,
                    target_float,
                    block_px=b_px,
                    kappa=cfg.kappa_flat16,
                    stride=stride,
                    normalize_by_count=cfg.normalize_flat_dm16,
                    strict=cfg.dm_strict,
                    auto_scale_kappa=bool(getattr(cfg, "auto_scale_kappa", True)),
                    norm_mode=getattr(cfg, "dm_norm_mode", "count"),
                    ref_count=float(getattr(cfg, "dm_ref_count", 100.0)),
                )
            else:
                l_dm = torch.zeros((), device=inp.device, dtype=inp.dtype)

            comps[b_px], comps[-999] = l_dm, l_bay
            loss_val = float(cfg.lambda_bayesian) * l_bay + float(cfg.lambda_flat_dm16) * l_dm
        else:
            s_max = float(getattr(cfg, "bayesian_sigma_max", 8.0))
            loss_val = bayesian_loss(
                inp,
                points,
                sigma=cfg.bayesian_sigma,
                background_ratio=cfg.bayesian_background_ratio,
                stride=stride,
                norm_mode=getattr(cfg, "bayesian_norm_mode", "canonical"),
                canonical_background=getattr(cfg, "bayesian_canonical_bg", True),
                adaptive_sigma=getattr(cfg, "bayesian_adaptive_sigma", False),
                sigma_min=getattr(cfg, "bayesian_sigma_min", 2.0),
                sigma_max=s_max,
            )
    elif cfg.allocation_loss_type == "fidt":
        if not getattr(cfg, "fidt_normalize_by_count", True):
            loss_val = canonical_fidt_loss(
                inp, points, stride=stride, loss_mode=getattr(cfg, "fidt_loss_type", "smooth_l1")
            )
        else:
            loss_val = fidt_loss(
                inp,
                points,
                k=getattr(cfg, "fidt_k", 6.0),
                stride=stride,
                loss_type=getattr(cfg, "fidt_loss_type", "smooth_l1"),
                normalize_by_count=True,
            )
    elif cfg.allocation_loss_type == "ot_sinkhorn":
        loss_val = sinkhorn_ot_loss(inp, points, reg=cfg.ot_reg, num_iters=cfg.ot_num_iters, stride=stride)
    elif cfg.use_multiscale_dm or cfg.use_hierarchical_dm:
        loss_val, comps = multiscale_dm_loss(
            inp,
            target_float,
            block_sizes_px=tuple(int(x) for x in cfg.dm_block_sizes_px),
            weights=tuple(float(x) for x in cfg.dm_weights),
            kappas=tuple(float(x) for x in cfg.dm_kappas),
            stride=stride,
            normalize_by_count=cfg.normalize_flat_dm16,
            strict=cfg.dm_strict,
            return_components=True,
            norm_mode=getattr(cfg, "dm_norm_mode", "count"),
            ref_count=float(getattr(cfg, "dm_ref_count", 100.0)),
        )
    else:
        b_px = int(getattr(cfg, "dm_block_px", 16))
        if float(cfg.lambda_flat_dm16) > 0.0:
            auto_k = bool(getattr(cfg, "auto_scale_kappa", True))
            loss_val = flat_dm_block_loss(
                inp,
                target_float,
                block_px=b_px,
                kappa=cfg.kappa_flat16,
                stride=stride,
                normalize_by_count=cfg.normalize_flat_dm16,
                strict=cfg.dm_strict,
                auto_scale_kappa=auto_k,
                norm_mode=getattr(cfg, "dm_norm_mode", "count"),
                ref_count=float(getattr(cfg, "dm_ref_count", 100.0)),
            )
        else:
            loss_val = torch.zeros((), device=inp.device, dtype=inp.dtype)
        comps[b_px] = loss_val

    return loss_val, comps


def route_allocation_loss(
    y: torch.Tensor,
    y0: torch.Tensor,
    target_float: torch.Tensor,
    points: list[torch.Tensor] | None,
    cfg: RMRv3LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute and route allocation loss (Bayesian, DM16, FIDT, OT) across y, y0, or dual."""
    comps: dict[int, torch.Tensor] = {}
    losses: dict[str, torch.Tensor] = {}
    if cfg.dm_target == "dual":
        loss_alloc_y, dm_comps_y = compute_single_allocation(y, target_float, points, cfg)
        loss_alloc_y0, dm_comps_y0 = compute_single_allocation(y0, target_float, points, cfg)
        loss_allocation = 0.5 * loss_alloc_y + 0.5 * loss_alloc_y0
        losses["allocation_y"], losses["allocation_y0"] = loss_alloc_y, loss_alloc_y0
        comps = {k: 0.5 * (dm_comps_y[k] + dm_comps_y0[k]) for k in dm_comps_y if k in dm_comps_y0}
    else:
        loss_allocation, comps = compute_single_allocation(
            y if cfg.dm_target == "y" else y0, target_float, points, cfg
        )

    losses["allocation"] = loss_allocation
    losses["flat_dm16"] = comps.get(16, loss_allocation)
    if -999 in comps:
        losses["bayesian"] = comps[-999]
    for bs, val in comps.items():
        if bs != -999:
            losses[f"dm_{bs}"] = val
    return loss_allocation, losses

