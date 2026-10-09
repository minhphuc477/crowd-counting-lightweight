from __future__ import annotations

import torch
import torch.nn as nn

from rmr_core.heads import build_fine_head
from rmr_core.necks import (
    AdditiveFPNNeck,
    ASPPLiteFPNNeck,
    HDCLiteFPNNeck,
    RepWeightedFPNNeck,
)
from rmr_core.scale_routing import ScaleRoutingHead, FactorizedRoutingHead
from rmr_core.operators import RegionSet
from rmr_core.types import RMRModelOutput
from ..regional_head import ProbabilisticRegionalEvidenceHead
from .config import RMRv3Config, _softplus_inverse
from .perspective_geometry import (
    DynamicCameraAnglePredictor,
    DiAGScaleRoutingHead,
    DiAGFactorizedRoutingHead,
)
from .dual_lattice import (
    push_forward_stride2_to_stride4,
    scale_regions_to_stride2,
    SubpixelAllocationHead,
)


def build_neck_module(in_channels: tuple[int, ...], cfg: RMRv3Config) -> nn.Module:
    """Build multi-scale FPN feature fusion neck."""
    if cfg.neck_type == "rep_weighted":
        return RepWeightedFPNNeck(
            in_channels=in_channels,
            width=cfg.feature_width,
            context_dilations=cfg.context_dilations,
        )
    if cfg.neck_type == "aspp_lite":
        return ASPPLiteFPNNeck(
            in_channels=in_channels,
            width=cfg.feature_width,
            aspp_dilations=cfg.aspp_dilations,
            use_aspp_gap=cfg.use_aspp_gap,
        )
    if cfg.neck_type == "hdc_lite":
        return HDCLiteFPNNeck(
            in_channels=in_channels,
            width=cfg.feature_width,
            hdc_dilations=cfg.hdc_dilations,
            use_gap=cfg.use_aspp_gap,
        )
    if cfg.neck_type == "additive":
        return AdditiveFPNNeck(in_channels=in_channels, width=cfg.feature_width)
    raise ValueError(
        f"Unsupported neck_type: '{cfg.neck_type}'. Must be 'additive', 'aspp_lite', 'hdc_lite', or 'rep_weighted'."
    )


def build_scale_router(
    cfg: RMRv3Config,
) -> tuple[DynamicCameraAnglePredictor | None, nn.Module | None]:
    """Build dynamic scale routing and perspective head."""
    if cfg.use_diag:
        dcap = DynamicCameraAnglePredictor(
            cfg.feature_width,
            len(cfg.region_sizes_px),
            use_vertical_gradient=getattr(cfg, "use_vertical_gradient_dcap", False),
        )
        use_tilt = getattr(cfg, "use_dcap_tilt", True)
        if cfg.factorized_scale_routing:
            router: nn.Module | None = DiAGFactorizedRoutingHead(
                cfg.feature_width,
                cfg.num_marginal_scales,
                cfg.num_aspect_ratios,
                cfg.scale_router_temperature,
                use_tilt=use_tilt,
            )
        else:
            router = DiAGScaleRoutingHead(
                cfg.feature_width,
                len(cfg.region_sizes_px),
                cfg.scale_router_temperature,
                use_tilt=use_tilt,
            )
        return dcap, router

    if cfg.factorized_scale_routing:
        return None, FactorizedRoutingHead(
            cfg.feature_width, cfg.num_marginal_scales, cfg.num_aspect_ratios, cfg.scale_router_temperature
        )
    if cfg.dynamic_scale_routing:
        return None, ScaleRoutingHead(
            cfg.feature_width, len(cfg.region_sizes_px), cfg.scale_router_temperature
        )
    return None, None


def build_1x1_gate(enabled: bool, in_channels: int) -> nn.Conv2d | None:
    """Instantiate a calibrated 1x1 foreground/semantic gating convolution."""
    if not enabled:
        return None
    gate = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
    nn.init.normal_(gate.weight, std=0.01)
    nn.init.constant_(gate.bias, 2.0)
    return gate


def build_fine_carrier_head(cfg: RMRv3Config) -> nn.Module:
    """Construct fine carrier head from configuration."""
    init_bias = _softplus_inverse(cfg.init_m0)
    return build_fine_head(
        width=cfg.feature_width,
        scale_conditioned_fine_head=cfg.scale_conditioned_fine_head,
        num_scales=len(cfg.region_sizes_px),
        init_bias=init_bias,
        temp_softplus=cfg.temp_softplus,
        scale_conditioned_prior=cfg.scale_conditioned_prior,
        density_curvature=cfg.density_curvature,
        gated_density_curvature=cfg.gated_density_curvature,
        curvature_dense_threshold=cfg.curvature_dense_threshold,
        curvature_gate_beta=cfg.curvature_gate_beta,
        curvature_pool_kernel=cfg.curvature_pool_kernel,
        subpixel_stride2=cfg.subpixel_stride2,
        floor_tau=cfg.floor_tau,
        curvature_alpha_init=getattr(cfg, "curvature_alpha_init", -8.0),
        density_adaptive_scale=getattr(cfg, "density_adaptive_scale", False),
        density_scale_gamma=getattr(cfg, "density_scale_gamma", 0.0),
        density_scale_learnable=getattr(cfg, "density_scale_learnable", False),
        scale_prior_boost=getattr(cfg, "scale_prior_boost", 0.0),
    )


def build_regional_evidence_head(cfg: RMRv3Config) -> nn.Module:
    """Construct probabilistic regional evidence head from configuration."""
    return ProbabilisticRegionalEvidenceHead(
        feature_dim=cfg.feature_width,
        hidden=cfg.region_head_hidden,
        init_rate=cfg.init_m0,
        region_sizes_px=cfg.region_sizes_px,
        dispersion_init=cfg.dispersion_init,
        dispersion_min=cfg.dispersion_min,
        dispersion_max=cfg.dispersion_max,
        native_scale_pooling=cfg.native_scale_pooling,
        regional_feature_stats=cfg.regional_feature_stats,
        hurdle_head=cfg.hurdle_head,
        floor_tau=cfg.floor_tau,
    )


def package_subpixel_output(
    out: RMRModelOutput,
    p4: torch.Tensor,
    allocator: SubpixelAllocationHead | None,
    subpixel_stride2: bool,
    h_in: int,
    w_in: int,
    regions_solver: RegionSet,
) -> None:
    """Reconcile subpixel dual-lattice outputs and mass conservation."""
    if allocator is not None:
        target_h2, target_w2 = (h_in + 1) // 2, (w_in + 1) // 2
        out["y_carrier"] = out.y
        out["y0_carrier"] = out.y0
        out["regions_carrier"] = regions_solver
        y2_raw, y02_raw = allocator.forward_pair(p4, out.y, out.y0)
        y2_alloc = y2_raw[..., :target_h2, :target_w2]
        y02_alloc = y02_raw[..., :target_h2, :target_w2]
        m4_y = out.y.sum(dim=(-2, -1), keepdim=True)
        m2_y = y2_alloc.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        out["y"] = y2_alloc * (m4_y / m2_y)
        m4_y0 = out.y0.sum(dim=(-2, -1), keepdim=True)
        m2_y0 = y02_alloc.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        out["y0"] = y02_alloc * (m4_y0 / m2_y0)
        out["regions"] = scale_regions_to_stride2(regions_solver, target_h2, target_w2)
    elif subpixel_stride2:
        out["y_carrier"] = push_forward_stride2_to_stride4(out.y)
        out["y0_carrier"] = push_forward_stride2_to_stride4(out.y0)
    else:
        out["y_carrier"] = out.y
        out["y0_carrier"] = out.y0
