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
from .voronoi_poisson import scale_normalized_voronoi_poisson_loss
from .radon_ot import radon_sliced_wasserstein_loss


def compute_single_allocation(
    inp: torch.Tensor,
    target_float: torch.Tensor,
    points: list[torch.Tensor] | None,
    cfg: RMRv3LossConfig,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Compute spatial allocation loss for a single density map according to cfg."""
    comps: dict[int, torch.Tensor] = {}
    stride = int(getattr(cfg, "output_stride", 4))

    if cfg.allocation_loss_type in ("dual_bayesian_dm16", "bayesian", "dual_bayesian_radon_ot", "radon_ot"):
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
        elif cfg.allocation_loss_type == "dual_bayesian_radon_ot":
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

            rot_proj = int(getattr(cfg, "radon_ot_projections", 4))
            rot_ref = float(getattr(cfg, "radon_ot_norm_ref", 100.0))
            l_rot = radon_sliced_wasserstein_loss(
                inp, points, stride=stride, num_projections=rot_proj, norm_ref=rot_ref, scale_by_length=bool(getattr(cfg, "radon_ot_scale_by_length", False))
            )
            lam_rot = float(getattr(cfg, "lambda_radon_ot", 1.0))
            comps[-888], comps[-999] = l_rot, l_bay
            loss_val = float(cfg.lambda_bayesian) * l_bay + lam_rot * l_rot
        elif cfg.allocation_loss_type == "radon_ot":
            rot_proj = int(getattr(cfg, "radon_ot_projections", 4))
            rot_ref = float(getattr(cfg, "radon_ot_norm_ref", 100.0))
            loss_val = radon_sliced_wasserstein_loss(
                inp, points, stride=stride, num_projections=rot_proj, norm_ref=rot_ref, scale_by_length=bool(getattr(cfg, "radon_ot_scale_by_length", False))
            )
            comps[-888] = loss_val
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
    elif cfg.allocation_loss_type == "sn_voronoi_poisson":
        l_center = float(getattr(cfg, "sn_vp_lambda_center", 0.1))
        l_global = float(getattr(cfg, "sn_vp_lambda_global", 0.5))
        n_mode = str(getattr(cfg, "sn_vp_norm_mode", "canonical"))
        n_ref = float(getattr(cfg, "sn_vp_norm_ref", 100.0))
        c_size = int(getattr(cfg, "sn_vp_chunk_size", 4096))
        loss_val = scale_normalized_voronoi_poisson_loss(
            inp,
            points,
            stride=stride,
            lambda_center=l_center,
            lambda_global=l_global,
            norm_mode=n_mode,
            norm_ref=n_ref,
            chunk_size=c_size,
        )
        comps[-777] = loss_val
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
    if cfg.dm_target == "dual" and y is not y0:
        loss_alloc_y, dm_comps_y = compute_single_allocation(y, target_float, points, cfg)
        loss_alloc_y0, dm_comps_y0 = compute_single_allocation(y0, target_float, points, cfg)
        loss_allocation = 0.5 * loss_alloc_y + 0.5 * loss_alloc_y0
        losses["allocation_y"], losses["allocation_y0"] = loss_alloc_y, loss_alloc_y0
        comps = {k: 0.5 * (dm_comps_y[k] + dm_comps_y0[k]) for k in dm_comps_y if k in dm_comps_y0}
    else:
        loss_allocation, comps = compute_single_allocation(
            y if cfg.dm_target == "y" else y0, target_float, points, cfg
        )
        if cfg.dm_target == "dual":
            losses["allocation_y"], losses["allocation_y0"] = loss_allocation, loss_allocation

    losses["allocation"] = loss_allocation
    if cfg.allocation_loss_type == "bayesian":
        losses["bayesian"] = loss_allocation
        losses["flat_dm16"] = torch.zeros_like(loss_allocation)
        losses["radon_ot"] = torch.zeros_like(loss_allocation)
    elif cfg.allocation_loss_type == "sn_voronoi_poisson":
        losses["sn_voronoi_poisson"] = loss_allocation
        losses["bayesian"] = torch.zeros_like(loss_allocation)
        losses["radon_ot"] = torch.zeros_like(loss_allocation)
        losses["flat_dm16"] = torch.zeros_like(loss_allocation)
    elif cfg.allocation_loss_type == "radon_ot":
        losses["radon_ot"] = loss_allocation
        losses["flat_dm16"] = torch.zeros_like(loss_allocation)
        losses["bayesian"] = torch.zeros_like(loss_allocation)
    elif cfg.allocation_loss_type == "dual_bayesian_radon_ot":
        losses["flat_dm16"] = torch.zeros_like(loss_allocation)
        losses["bayesian"] = comps.get(-999, torch.zeros_like(loss_allocation))
        losses["radon_ot"] = comps.get(-888, torch.zeros_like(loss_allocation))
    else:
        b_px = int(getattr(cfg, "dm_block_px", 16))
        losses["flat_dm16"] = comps.get(b_px, loss_allocation)
        if -999 in comps:
            losses["bayesian"] = comps[-999]
        if -888 in comps:
            losses["radon_ot"] = comps[-888]
    for bs, val in comps.items():
        if bs not in (-999, -888):
            losses[f"dm_{bs}"] = val
    return loss_allocation, losses

