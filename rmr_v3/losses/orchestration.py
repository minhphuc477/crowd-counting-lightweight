"""Multi-task loss orchestration for Radon Measure Recovery (RMRv3/v35).

Coordinates hierarchical and multi-scale losses across:
- Macro & Global Count supervision (Negative-Binomial, Poisson, L1, Huber).
- Cell-level spatial supervision (Count-Invariant v2, Harmonized, Mass-Weighted, Balanced Smooth L1).
- Spatial Point Allocation (Dirichlet-Multinomial, Bayesian, FIDT, Optimal Transport).
- Probabilistic Regional Evidence (Scale-balanced Negative Binomial NLL, Truncated NB, Hurdle Focal BCE).
- Geometric and Background Priors (Curvature power, Top-K Hard Background, Scale-Density alignment).
- Dual-Lattice and Spectral Priors (Subpixel stride-2 push-forward, ChfL, Fourier spectral).
"""
from __future__ import annotations

from typing import Any
import torch

from rmr_core.operators import RegionSet, regional_sum
from rmr_core.losses import count_magnitude_loss
from rmr_core.spectral import characteristic_function_loss, count_preserving_spectral_loss

from .config import RMRv3LossConfig
from .router import TargetSupervisionRouter
from .dense_scaling import compute_elementwise_dense_scaling
from .allocation import route_allocation_loss
from .cell import compute_cell_loss
from .regional import (
    hurdle_focal_bce_loss,
    scale_balanced_regional_nb_nll,
    truncated_nb_nll_loss,
)
from .spatial_priors import (
    curvature_power_loss,
    foreground_gating_bce_loss,
    physical_scale_alignment_loss,
    topk_hard_background_loss,
)
from .dual_supervision import align_target_to_prediction, compute_dual_lattice_losses
from .chfl import canonical_chfl_loss
from rmr_v3.model.dual_lattice import push_forward_stride2_to_stride4

__all__ = ["compute_rmr_v3_losses", "TargetSupervisionRouter"]


def _compute_core_losses(
    target_float: torch.Tensor,
    target_region: torch.Tensor,
    y: torch.Tensor,
    y0: torch.Tensor,
    regions: RegionSet,
    mean_region: torch.Tensor,
    dispersion_region: torch.Tensor,
    cfg: RMRv3LossConfig,
    points: list[torch.Tensor] | None,
    router: TargetSupervisionRouter,
) -> dict[str, torch.Tensor]:
    """Compute primary count, cell, allocation, and regional Negative-Binomial losses."""
    losses: dict[str, torch.Tensor] = {}

    def _compute_count_loss(density_map: torch.Tensor) -> torch.Tensor:
        return count_magnitude_loss(
            density_map, target_float, mode=cfg.count_loss_mode, dispersion=cfg.count_nb_dispersion
        )

    def _compute_cell_loss_fn(density_map: torch.Tensor) -> torch.Tensor:
        return compute_cell_loss(density_map, target_float, cfg)

    cnt_m = cfg.dm_target if getattr(cfg, "count_target", "default") == "default" else cfg.count_target
    cell_m = cfg.dm_target if getattr(cfg, "cell_target", "default") == "default" else cfg.cell_target
    cnt_router = router if cnt_m == router.mode else TargetSupervisionRouter(cnt_m)
    cell_router = router if cell_m == router.mode else TargetSupervisionRouter(cell_m)

    # 1. Macro Count Loss
    if cfg.lambda_count > 0.0:
        loss_count, aux_count = cnt_router.dispatch(_compute_count_loss, y, y0)
        losses["count"] = loss_count
        if "y" in aux_count and "y0" in aux_count:
            losses["count_y"], losses["count_y0"] = aux_count["y"], aux_count["y0"]
    else:
        losses["count"] = torch.zeros((), device=y.device, dtype=y.dtype)

    # 2. Cell Allocation Loss
    if cfg.lambda_cell > 0.0 or getattr(cfg, "lambda_fine_cell", 0.0) > 0.0:
        loss_cell, aux_cell = cell_router.dispatch(_compute_cell_loss_fn, y, y0)
        losses["cell"] = loss_cell
        if "y" in aux_cell and "y0" in aux_cell:
            losses["cell_y"], losses["cell_y0"] = aux_cell["y"], aux_cell["y0"]
    else:
        losses["cell"] = torch.zeros((), device=y.device, dtype=y.dtype)

    # 3. Spatial Allocation Loss (Bayesian, DM16, FIDT, OT)
    loss_allocation, alloc_losses = route_allocation_loss(y, y0, target_float, points, cfg)
    losses.update(alloc_losses)

    # 4. Regional Negative-Binomial Evidence Loss
    if cfg.lambda_region_nb > 0.0:
        losses["region_nb"] = scale_balanced_regional_nb_nll(
            target_region,
            mean_region,
            dispersion_region,
            regions,
            mass_weight_alpha=float(getattr(cfg, "regional_mass_weight_alpha", 0.0)),
            regional_nb_beta=float(getattr(cfg, "regional_nb_beta", 0.0)),
            faithful_regional_nb=bool(getattr(cfg, "faithful_regional_nb", False)),
        )
    else:
        losses["region_nb"] = torch.zeros((), device=y.device, dtype=y.dtype)

    # Composite Total Core Loss
    if cfg.allocation_loss_type == "dual_bayesian_dm16":
        alloc_term = loss_allocation
    elif cfg.allocation_loss_type == "bayesian":
        lam_alloc = float(cfg.lambda_bayesian) if float(cfg.lambda_bayesian) > 0.0 else float(cfg.lambda_flat_dm16)
        alloc_term = lam_alloc * loss_allocation
    else:
        alloc_term = cfg.lambda_flat_dm16 * loss_allocation
    losses["total"] = (
        cfg.lambda_count * losses["count"]
        + alloc_term
        + cfg.lambda_cell * losses["cell"]
        + cfg.lambda_region_nb * losses["region_nb"]
    )

    # Optional L1 count supervision
    if getattr(cfg, "lambda_count_l1", 0.0) > 0.0:
        loss_l1, _ = cnt_router.dispatch(
            lambda dm: count_magnitude_loss(dm, target_float, mode="l1"), y, y0
        )
        losses["count_l1"] = loss_l1
        losses["total"] = losses["total"] + cfg.lambda_count_l1 * loss_l1
    else:
        losses["count_l1"] = torch.zeros((), device=y.device, dtype=y.dtype)

    return losses


def _compute_auxiliary_losses(
    losses: dict[str, torch.Tensor],
    outputs: dict[str, Any],
    target_float: torch.Tensor,
    target_region: torch.Tensor,
    mean_region: torch.Tensor,
    dispersion_region: torch.Tensor,
    y: torch.Tensor,
    y0: torch.Tensor,
    cfg: RMRv3LossConfig,
    zero_val: torch.Tensor,
    router: TargetSupervisionRouter,
    points: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute auxiliary geometric, prior, and dual-lattice regularizers."""
    stride = int(getattr(cfg, "output_stride", 4))

    # Curvature Power Loss (high-density variance stabilization)
    if cfg.lambda_curvature > 0.0:
        def _compute_curv(dmap: torch.Tensor) -> torch.Tensor:
            return curvature_power_loss(
                dmap,
                target_float,
                threshold=cfg.curvature_gate_threshold,
                kernel_size=cfg.curvature_gate_kernel,
                mode=cfg.curvature_gate_mode,
                smooth_scale=cfg.curvature_gate_scale,
                stride=stride,
            )
        loss_curv, _ = router.dispatch(_compute_curv, y, y0)
        losses["curvature"] = loss_curv
        losses["total"] = losses["total"] + cfg.lambda_curvature * loss_curv
    else:
        losses["curvature"] = zero_val

    # Top-K Hard Background Mining Loss
    if cfg.lambda_hard_bg > 0.0:
        def _compute_hard_bg(dmap: torch.Tensor) -> torch.Tensor:
            return topk_hard_background_loss(dmap, target_float, ratio=cfg.hard_bg_ratio, stride=stride)
        loss_hard_bg, _ = router.dispatch(_compute_hard_bg, y, y0)
        losses["hard_bg"] = loss_hard_bg
        losses["total"] = losses["total"] + cfg.lambda_hard_bg * loss_hard_bg
    else:
        losses["hard_bg"] = zero_val

    # Foreground Gating BCE Loss
    fg_logit = outputs.get("fg_logit", None)
    if fg_logit is not None and cfg.lambda_fg_gate > 0.0:
        fg_bce = foreground_gating_bce_loss(fg_logit, target_float)
        losses["fg_bce"] = fg_bce
        losses["total"] = losses["total"] + cfg.lambda_fg_gate * fg_bce
    else:
        losses["fg_bce"] = zero_val

    # Regional Hurdle BCE & Truncated NB Loss
    hurdle_logit = outputs.get("hurdle_logit", None)
    if hurdle_logit is not None:
        h_bce = (
            hurdle_focal_bce_loss(hurdle_logit.float(), target_region)
            if cfg.lambda_hurdle > 0.0
            else zero_val
        )
        t_nb = (
            truncated_nb_nll_loss(
                mean_region,
                dispersion_region,
                target_region,
                regional_nb_beta=float(getattr(cfg, "regional_nb_beta", 0.0)),
                faithful_regional_nb=bool(getattr(cfg, "faithful_regional_nb", False)),
            )
            if cfg.lambda_trunc_nb > 0.0
            else zero_val
        )
        losses["hurdle_bce"], losses["trunc_nb"] = h_bce, t_nb
        losses["total"] = losses["total"] + cfg.lambda_hurdle * h_bce + cfg.lambda_trunc_nb * t_nb
    else:
        losses["hurdle_bce"] = losses["trunc_nb"] = zero_val

    # Physical Scale Alignment Loss
    scale_weights = outputs.get("pi_scale", outputs.get("scale_weights", None))
    if scale_weights is not None and cfg.lambda_scale_align > 0.0:
        losses["scale_align"] = physical_scale_alignment_loss(
            scale_weights=scale_weights,
            target_y=target_float,
            tau_dense=cfg.scale_align_tau_dense,
            tau_sparse=cfg.scale_align_tau_sparse,
            kernel_size=cfg.scale_align_kernel,
            mask_background=cfg.scale_align_mask_bg,
            stride=stride,
        )
        losses["total"] = losses["total"] + cfg.lambda_scale_align * losses["scale_align"]
    else:
        losses["scale_align"] = zero_val

    # Dual-Lattice Carrier Supervision (Subpixel stride-2)
    y_carrier = outputs.get("y_carrier", None)
    is_dual_lattice = (
        y_carrier is not None
        and y_carrier.shape[-2:] != y.shape[-2:]
        and (cfg.lambda_carrier_cell > 0.0 or cfg.lambda_fine_cell > 0.0)
    )
    if is_dual_lattice:
        target_stride4 = push_forward_stride2_to_stride4(target_float)
        dual = compute_dual_lattice_losses(
            y_fine=y,
            y_carrier=y_carrier.float(),
            target_stride2=target_float,
            target_stride4=target_stride4,
            lambda_carrier_cell=cfg.lambda_carrier_cell,
            lambda_fine_cell=cfg.lambda_fine_cell,
            cell_loss_mode=cfg.cell_loss_mode,
            beta=cfg.cell_beta,
            eps=cfg.cell_mass_weight_eps,
            alpha=float(cfg.cell_mass_weight_alpha),
            gamma=float(cfg.cell_mass_weight_gamma),
        )
        losses["cell_carrier"], losses["cell_fine"] = dual["cell_carrier"], losses["cell"]
        eff_fine_delta = (cfg.lambda_fine_cell - cfg.lambda_cell) if cfg.lambda_fine_cell > 0.0 else 0.0
        losses["total"] = losses["total"] + cfg.lambda_carrier_cell * dual["cell_carrier"] + eff_fine_delta * losses["cell"]
    else:
        losses["cell_carrier"] = losses["cell_fine"] = zero_val

    # Count-Preserving Heavy-Tailed Spectral Loss
    if cfg.use_spectral_loss and cfg.lambda_spectral > 0.0:
        sp_kw = {
            "beta": cfg.spectral_beta,
            "lambda_count": cfg.lambda_spectral_dc,
            "lambda_spectral": 1.0,
            "omega_0": getattr(cfg, "spectral_omega_0", 0.05),
            "bandpass": getattr(cfg, "spectral_bandpass", False),
            "omega_low": getattr(cfg, "spectral_omega_low", 0.02),
            "omega_high": getattr(cfg, "spectral_omega_high", 0.35),
            "transform": getattr(cfg, "spectral_transform", "fft"),
        }
        if router.mode == "dual":
            l_y, c_y = count_preserving_spectral_loss(y, target_float, **sp_kw)
            l_y0, c_y0 = count_preserving_spectral_loss(y0, target_float, **sp_kw)
            loss_spec = 0.5 * (l_y + l_y0)
            loss_spec_dc = 0.5 * (c_y["spectral_dc"] + c_y0["spectral_dc"])
            loss_spec_ac = 0.5 * (c_y["spectral_ac"] + c_y0["spectral_ac"])
        else:
            loss_spec, comps = count_preserving_spectral_loss(
                y0 if router.mode == "y0" else y, target_float, **sp_kw
            )
            loss_spec_dc, loss_spec_ac = comps["spectral_dc"], comps["spectral_ac"]

        losses["spectral"], losses["spectral_dc"], losses["spectral_ac"] = loss_spec, loss_spec_dc, loss_spec_ac
        losses["total"] = losses["total"] + cfg.lambda_spectral * loss_spec
    else:
        losses["spectral"] = losses["spectral_dc"] = losses["spectral_ac"] = zero_val

    # Characteristic Function Loss (ChfL)
    if getattr(cfg, "use_chfl_loss", False) and getattr(cfg, "lambda_chfl", 0.0) > 0.0:
        def _compute_chfl(dmap: torch.Tensor) -> torch.Tensor:
            if getattr(cfg, "chfl_canonical", True):
                return canonical_chfl_loss(
                    dmap,
                    points,
                    chf_tik=getattr(cfg, "chfl_tik", 0.01),
                    chf_step=getattr(cfg, "chfl_step", 16),
                    bandwidth=getattr(cfg, "chfl_bandwidth", 8.0),
                    stride=stride,
                )
            return characteristic_function_loss(
                dmap,
                points,
                omega_max=getattr(cfg, "chfl_omega_max", 0.5),
                num_frequencies=getattr(cfg, "chfl_num_frequencies", 64),
                stride=stride,
            )
        loss_chfl, _ = router.dispatch(_compute_chfl, y, y0)
        losses["chfl"] = loss_chfl
        losses["total"] = losses["total"] + cfg.lambda_chfl * loss_chfl
    else:
        losses["chfl"] = zero_val

    return losses


def compute_rmr_v3_losses(
    outputs: dict[str, Any],
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig | None = None,
    points: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute composite multi-task loss for RMRv3/v35 model predictions."""
    if cfg is None:
        cfg = RMRv3LossConfig()

    if target_y.ndim == 2:
        target_y = target_y.unsqueeze(0).unsqueeze(0)
    elif target_y.ndim == 3:
        target_y = target_y.unsqueeze(1)

    y = outputs["y"].float()
    y0 = outputs["y0"].float()
    zero_val = torch.zeros((), device=y.device, dtype=y.dtype)

    if target_y.shape[0] == 0 or target_y.numel() == 0:
        empty_val = (y.sum() + y0.sum()) * 0.0
        keys = [
            "total", "count", "count_l1", "cell", "allocation", "flat_dm16", "region_nb", "curvature",
            "hard_bg", "fg_bce", "hurdle_bce", "trunc_nb", "scale_align", "cell_carrier", "cell_fine",
            "spectral", "spectral_dc", "spectral_ac", "chfl",
        ]
        return {k: empty_val for k in keys}

    if cfg.elementwise_dense_scaling or cfg.density_loss_scaling:
        return compute_elementwise_dense_scaling(
            outputs, target_y, cfg, compute_losses_fn=compute_rmr_v3_losses, points=points
        )

    target_float = align_target_to_prediction(
        target_y.float(), y, points=points, stride=int(getattr(cfg, "output_stride", 2 if y.shape[-2] > 150 else 4))
    )
    regions: RegionSet = outputs["regions"]
    mean_region = outputs["b_region"].float()
    dispersion_region = outputs["region_dispersion"].float()

    target_region = regional_sum(target_float, regions.boxes, out_dtype=torch.float32)

    router = TargetSupervisionRouter(cfg.dm_target)
    losses = _compute_core_losses(
        target_float=target_float,
        target_region=target_region,
        y=y,
        y0=y0,
        regions=regions,
        mean_region=mean_region,
        dispersion_region=dispersion_region,
        cfg=cfg,
        points=points,
        router=router,
    )
    return _compute_auxiliary_losses(
        losses=losses,
        outputs=outputs,
        target_float=target_float,
        target_region=target_region,
        mean_region=mean_region,
        dispersion_region=dispersion_region,
        y=y,
        y0=y0,
        cfg=cfg,
        zero_val=zero_val,
        router=router,
        points=points,
    )
