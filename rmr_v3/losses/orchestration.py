from __future__ import annotations

from dataclasses import replace
from typing import Any
import torch
import torch.nn.functional as F

from rmr_core.operators import RegionSet, regional_sum
from rmr_core.losses import (
    balanced_smooth_l1,
    count_magnitude_loss,
    flat_dm16_loss,
    multiscale_dm_loss,
)
from rmr_core.spectral import count_preserving_spectral_loss
from .config import RMRv3LossConfig
from .point_supervision import bayesian_loss, sinkhorn_ot_loss
from .auxiliary import (
    curvature_power_loss,
    hurdle_focal_bce_loss,
    mass_weighted_cell_loss,
    physical_scale_alignment_loss,
    scale_balanced_regional_nb_nll,
    topk_hard_background_loss,
    truncated_nb_nll_loss,
)
from .dual_supervision import compute_dual_lattice_losses
from rmr_v3.model.dual_lattice import push_forward_stride2_to_stride4


class TargetSupervisionRouter:
    """Routes target supervision across terminal measure y, initial carrier y0, or symmetric dual."""

    def __init__(self, target_mode: str = "y") -> None:
        if target_mode not in ("dual", "y0", "y"):
            raise ValueError(f"Unknown target supervision mode: '{target_mode}'. Expected 'dual', 'y0', or 'y'.")
        self.mode = target_mode

    def dispatch(
        self,
        fn: Any,
        y: torch.Tensor,
        y0: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.mode == "dual":
            l_y = fn(y, *args, **kwargs)
            l_y0 = fn(y0, *args, **kwargs)
            return 0.5 * l_y + 0.5 * l_y0, {"y": l_y, "y0": l_y0}
        elif self.mode == "y0":
            l_y0 = fn(y0, *args, **kwargs)
            return l_y0, {"y0": l_y0}
        else:
            l_y = fn(y, *args, **kwargs)
            return l_y, {"y": l_y}


def _compute_elementwise_dense_scaling(
    outputs: dict[str, Any],
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig,
    points: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Elementwise sample-level importance weighting without batch cross-talk leakage."""
    b_sz = target_y.shape[0]
    if b_sz == 0:
        return {}
    cfg_single = replace(cfg, elementwise_dense_scaling=False, density_loss_scaling=False)

    total_gt = target_y.float().sum(dim=(-2, -1)).view(-1)  # [B]
    dense_boost = float(cfg.dense_loss_alpha) * torch.clamp(
        (total_gt - float(cfg.dense_loss_thresh)) / float(cfg.dense_loss_norm),
        min=0.0,
        max=float(cfg.dense_loss_max_boost),
    )
    sample_weights = (1.0 + dense_boost).detach()  # [B]

    sample_losses = []
    for i in range(b_sz):
        out_i = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == b_sz:
                out_i[k] = v[i : i + 1]
            elif isinstance(v, list):
                out_i[k] = [
                    item[i : i + 1]
                    if isinstance(item, torch.Tensor) and item.ndim > 0 and item.shape[0] == b_sz
                    else item
                    for item in v
                ]
            else:
                out_i[k] = v
        tgt_i = target_y[i : i + 1]
        pts_i = [points[i]] if points is not None and i < len(points) else None
        l_i = compute_rmr_v3_losses(out_i, tgt_i, cfg_single, points=pts_i)
        sample_losses.append(l_i)

    aggregated = {}
    for k in sample_losses[0].keys():
        tensors = [sl[k] for sl in sample_losses]
        stacked = torch.stack(tensors)
        aggregated[k] = (sample_weights * stacked).mean() if k == "total" else stacked.mean()

    aggregated["dense_loss_scale"] = sample_weights.mean()
    return aggregated


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
    losses: dict[str, torch.Tensor] = {}

    def _compute_count_loss(density_map: torch.Tensor) -> torch.Tensor:
        return count_magnitude_loss(
            density_map, target_float, mode=cfg.count_loss_mode, dispersion=cfg.count_nb_dispersion,
        )

    def _compute_cell_loss(density_map: torch.Tensor) -> torch.Tensor:
        stride = int(getattr(cfg, "output_stride", 4))
        if cfg.cell_loss_mode == "mass_weighted":
            return mass_weighted_cell_loss(
                density_map, target_float,
                beta=cfg.cell_beta, eps=cfg.cell_mass_weight_eps,
                alpha=float(cfg.cell_mass_weight_alpha), gamma=float(cfg.cell_mass_weight_gamma),
                stride=stride,
            )
        return balanced_smooth_l1(density_map, target_float, beta=cfg.cell_beta, stride=stride)

    loss_count, aux_count = router.dispatch(_compute_count_loss, y, y0)
    losses["count"] = loss_count
    if "y" in aux_count and "y0" in aux_count:
        losses["count_y"] = aux_count["y"]
        losses["count_y0"] = aux_count["y0"]

    loss_cell, aux_cell = router.dispatch(_compute_cell_loss, y, y0)
    losses["cell"] = loss_cell
    if "y" in aux_cell and "y0" in aux_cell:
        losses["cell_y"] = aux_cell["y"]
        losses["cell_y0"] = aux_cell["y0"]

    def _compute_single_allocation(inp: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        comps: dict[int, torch.Tensor] = {}
        stride = int(getattr(cfg, "output_stride", 4))
        if cfg.allocation_loss_type == "bayesian":
            loss_val = bayesian_loss(
                inp, points, sigma=cfg.bayesian_sigma,
                background_ratio=cfg.bayesian_background_ratio, stride=stride,
            )
        elif cfg.allocation_loss_type == "ot_sinkhorn":
            loss_val = sinkhorn_ot_loss(inp, points, reg=cfg.ot_reg, num_iters=cfg.ot_num_iters, stride=stride)
        elif cfg.use_multiscale_dm or cfg.use_hierarchical_dm:
            loss_val, comps = multiscale_dm_loss(
                inp, target_float,
                block_sizes_px=tuple(int(x) for x in cfg.dm_block_sizes_px),
                weights=tuple(float(x) for x in cfg.dm_weights),
                kappas=tuple(float(x) for x in cfg.dm_kappas),
                stride=stride, normalize_by_count=cfg.normalize_flat_dm16,
                strict=cfg.dm_strict, return_components=True,
            )
        else:
            loss_val = flat_dm16_loss(
                inp, target_float, kappa=cfg.kappa_flat16, stride=stride,
                normalize_by_count=cfg.normalize_flat_dm16, strict=cfg.dm_strict,
            )
            comps[16] = loss_val
        return loss_val, comps

    dm_components: dict[int, torch.Tensor] = {}
    if cfg.dm_target == "dual":
        loss_alloc_y, dm_components = _compute_single_allocation(y)
        loss_alloc_y0, _ = _compute_single_allocation(y0)
        loss_allocation = 0.5 * loss_alloc_y + 0.5 * loss_alloc_y0
        losses["allocation_y"] = loss_alloc_y
        losses["allocation_y0"] = loss_alloc_y0
    elif cfg.dm_target == "y":
        loss_allocation, dm_components = _compute_single_allocation(y)
    else:  # "y0"
        loss_allocation, dm_components = _compute_single_allocation(y0)

    losses["allocation"] = loss_allocation
    losses["flat_dm16"] = loss_allocation
    for bs, val in dm_components.items():
        losses[f"dm_{bs}"] = val

    losses["region_nb"] = scale_balanced_regional_nb_nll(
        target_region, mean_region, dispersion_region, regions,
    )
    losses["total"] = (
        cfg.lambda_count * losses["count"]
        + cfg.lambda_flat_dm16 * loss_allocation
        + cfg.lambda_cell * losses["cell"]
        + cfg.lambda_region_nb * losses["region_nb"]
    )
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
) -> dict[str, torch.Tensor]:
    if cfg.lambda_curvature > 0.0:
        stride = int(getattr(cfg, "output_stride", 4))
        def _compute_curv(dmap: torch.Tensor) -> torch.Tensor:
            return curvature_power_loss(
                dmap, target_float, threshold=cfg.curvature_gate_threshold,
                kernel_size=cfg.curvature_gate_kernel, mode=cfg.curvature_gate_mode,
                smooth_scale=cfg.curvature_gate_scale, stride=stride,
            )
        loss_curv, _ = router.dispatch(_compute_curv, y, y0)
        losses["curvature"] = loss_curv
        losses["total"] = losses["total"] + cfg.lambda_curvature * loss_curv
    else:
        losses["curvature"] = zero_val

    if cfg.lambda_hard_bg > 0.0:
        stride = int(getattr(cfg, "output_stride", 4))
        def _compute_hard_bg(dmap: torch.Tensor) -> torch.Tensor:
            return topk_hard_background_loss(dmap, target_float, ratio=cfg.hard_bg_ratio, stride=stride)
        loss_hard_bg, _ = router.dispatch(_compute_hard_bg, y, y0)
        losses["hard_bg"] = loss_hard_bg
        losses["total"] = losses["total"] + cfg.lambda_hard_bg * loss_hard_bg
    else:
        losses["hard_bg"] = zero_val

    fg_logit = outputs.get("fg_logit", None)
    if fg_logit is not None and cfg.lambda_fg_gate > 0.0:
        t_bin = (target_float > 0.0).float()
        if t_bin.ndim == 3:
            t_bin = t_bin.unsqueeze(1)
        t_dilated = F.max_pool2d(t_bin, kernel_size=3, stride=1, padding=1)
        fg_pred = fg_logit.float()
        if fg_pred.shape[-2:] != t_dilated.shape[-2:]:
            fg_pred = F.interpolate(fg_pred, size=t_dilated.shape[-2:], mode="bilinear", align_corners=False)
        fg_bce = F.binary_cross_entropy_with_logits(fg_pred, t_dilated)
        losses["fg_bce"] = fg_bce
        losses["total"] = losses["total"] + cfg.lambda_fg_gate * fg_bce
    else:
        losses["fg_bce"] = zero_val

    hurdle_logit = outputs.get("hurdle_logit", None)
    if hurdle_logit is not None:
        if cfg.lambda_hurdle > 0.0:
            losses["hurdle_bce"] = hurdle_focal_bce_loss(hurdle_logit.float(), target_region)
            losses["total"] = losses["total"] + cfg.lambda_hurdle * losses["hurdle_bce"]
        else:
            losses["hurdle_bce"] = zero_val
        if cfg.lambda_trunc_nb > 0.0:
            losses["trunc_nb"] = truncated_nb_nll_loss(mean_region, dispersion_region, target_region)
            losses["total"] = losses["total"] + cfg.lambda_trunc_nb * losses["trunc_nb"]
        else:
            losses["trunc_nb"] = zero_val
    else:
        losses["hurdle_bce"] = zero_val
        losses["trunc_nb"] = zero_val

    scale_weights = outputs.get("pi_scale", outputs.get("scale_weights", None))
    if scale_weights is not None and cfg.lambda_scale_align > 0.0:
        stride = int(getattr(cfg, "output_stride", 4))
        losses["scale_align"] = physical_scale_alignment_loss(
            scale_weights=scale_weights, target_y=target_float,
            tau_dense=cfg.scale_align_tau_dense, tau_sparse=cfg.scale_align_tau_sparse,
            kernel_size=cfg.scale_align_kernel, mask_background=cfg.scale_align_mask_bg, stride=stride,
        )
        losses["total"] = losses["total"] + cfg.lambda_scale_align * losses["scale_align"]
    else:
        losses["scale_align"] = zero_val

    # Dual-Lattice Carrier Supervision (RMR-v30 H2/H3/H4: subpixel_stride2=True)
    # Auto-detected: y_carrier shape differs from y when stride-2 fine head is active.
    y_carrier = outputs.get("y_carrier", None)
    is_dual_lattice = (
        y_carrier is not None
        and y_carrier.shape[-2:] != y.shape[-2:]
        and (cfg.lambda_carrier_cell > 0.0 or cfg.lambda_fine_cell > 0.0)
    )
    if is_dual_lattice:
        # Mass-preserving push: 2x2 box sum → stride-4 carrier target
        target_stride4 = push_forward_stride2_to_stride4(target_float)
        dual = compute_dual_lattice_losses(
            y_fine=y, y_carrier=y_carrier.float(),
            target_stride2=target_float, target_stride4=target_stride4,
            lambda_carrier_cell=cfg.lambda_carrier_cell, lambda_fine_cell=cfg.lambda_fine_cell,
            cell_loss_mode=cfg.cell_loss_mode,
            beta=cfg.cell_beta, eps=cfg.cell_mass_weight_eps,
            alpha=float(cfg.cell_mass_weight_alpha), gamma=float(cfg.cell_mass_weight_gamma),
        )
        losses["cell_carrier"] = dual["cell_carrier"]
        losses["cell_fine"] = losses["cell"]
        # losses["total"] in core losses already includes cfg.lambda_cell * losses["cell"].
        # Add carrier cell loss, and adjust fine cell loss if lambda_fine_cell is explicitly specified:
        eff_fine_delta = (cfg.lambda_fine_cell - cfg.lambda_cell) if cfg.lambda_fine_cell > 0.0 else 0.0
        losses["total"] = losses["total"] + cfg.lambda_carrier_cell * dual["cell_carrier"] + eff_fine_delta * losses["cell"]
    else:
        losses["cell_carrier"] = zero_val
        losses["cell_fine"] = zero_val

    # Count-Preserving Heavy-Tailed Spectral Loss (Hypothesis H2)
    if cfg.use_spectral_loss and cfg.lambda_spectral > 0.0:
        if router.mode == "dual":
            l_y, c_y = count_preserving_spectral_loss(
                y, target_float, beta=cfg.spectral_beta,
                lambda_count=cfg.lambda_spectral_dc, lambda_spectral=1.0,
            )
            l_y0, c_y0 = count_preserving_spectral_loss(
                y0, target_float, beta=cfg.spectral_beta,
                lambda_count=cfg.lambda_spectral_dc, lambda_spectral=1.0,
            )
            loss_spec = 0.5 * l_y + 0.5 * l_y0
            loss_spec_dc = 0.5 * c_y["spectral_dc"] + 0.5 * c_y0["spectral_dc"]
            loss_spec_ac = 0.5 * c_y["spectral_ac"] + 0.5 * c_y0["spectral_ac"]
        elif router.mode == "y0":
            loss_spec, comps = count_preserving_spectral_loss(
                y0, target_float, beta=cfg.spectral_beta,
                lambda_count=cfg.lambda_spectral_dc, lambda_spectral=1.0,
            )
            loss_spec_dc = comps["spectral_dc"]
            loss_spec_ac = comps["spectral_ac"]
        else:
            loss_spec, comps = count_preserving_spectral_loss(
                y, target_float, beta=cfg.spectral_beta,
                lambda_count=cfg.lambda_spectral_dc, lambda_spectral=1.0,
            )
            loss_spec_dc = comps["spectral_dc"]
            loss_spec_ac = comps["spectral_ac"]

        losses["spectral"] = loss_spec
        losses["spectral_dc"] = loss_spec_dc
        losses["spectral_ac"] = loss_spec_ac
        losses["total"] = losses["total"] + cfg.lambda_spectral * loss_spec
    else:
        losses["spectral"] = zero_val
        losses["spectral_dc"] = zero_val
        losses["spectral_ac"] = zero_val

    return losses



def compute_rmr_v3_losses(
    outputs: dict[str, Any],
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig | None = None,
    points: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute composite multi-task loss for RMRv3 model predictions."""
    if cfg is None:
        cfg = RMRv3LossConfig()

    if target_y.ndim == 2:
        target_y = target_y.unsqueeze(0).unsqueeze(0)
    elif target_y.ndim == 3:
        target_y = target_y.unsqueeze(1)

    y = outputs["y"].float()
    y0 = outputs["y0"].float()
    zero_val = (y.sum() + y0.sum()) * 0.0

    if target_y.shape[0] == 0 or target_y.numel() == 0:
        return {
            "total": zero_val, "count": zero_val, "cell": zero_val,
            "allocation": zero_val, "flat_dm16": zero_val, "region_nb": zero_val,
            "curvature": zero_val, "hard_bg": zero_val, "fg_bce": zero_val,
            "hurdle_bce": zero_val, "trunc_nb": zero_val, "scale_align": zero_val,
            "cell_carrier": zero_val, "cell_fine": zero_val,
            "spectral": zero_val, "spectral_dc": zero_val, "spectral_ac": zero_val,
        }

    if cfg.elementwise_dense_scaling or cfg.density_loss_scaling:
        return _compute_elementwise_dense_scaling(outputs, target_y, cfg, points=points)

    target_float = target_y.float()
    regions: RegionSet = outputs["regions"]
    mean_region = outputs["b_region"].float()
    dispersion_region = outputs["region_dispersion"].float()

    target_region = regional_sum(target_float, regions.boxes, out_dtype=torch.float32)

    router = TargetSupervisionRouter(cfg.dm_target)
    losses = _compute_core_losses(
        target_float=target_float, target_region=target_region,
        y=y, y0=y0, regions=regions, mean_region=mean_region,
        dispersion_region=dispersion_region, cfg=cfg, points=points, router=router,
    )

    return _compute_auxiliary_losses(
        losses=losses, outputs=outputs, target_float=target_float,
        target_region=target_region, mean_region=mean_region,
        dispersion_region=dispersion_region, y=y, y0=y0,
        cfg=cfg, zero_val=zero_val, router=router,
    )
