"""rmr_v3/model.py — RMR-v3: Reliability-Weighted Regional Measure Reconciliation.

Architecture contract (frozen):
  - Backbone: mobilenetv4_conv_small_050.e3000_r224_in1k, truncated at reduction 16
  - FPN: AdditiveFPNNeck, width=32
  - Fine head: FineMeasureHead (stride-4 positive measure Y0)
  - Regional evidence head: ScaleMatchedRegionalEvidenceHead (shared MLP, regions 32/64/128px)
  - Reliability head: NEW — additional Linear(48->1) on top of regional head hidden layer
  - Solver: T=2 nonneg projected SIRT with reliability-weighted residual, omega=1.0 (fixed)
  - Detach policy: detach_region_mean=True, detach_reliability=True (registered main variant)
  - All regional algebra in FP32 (prefix sums, rectangle inclusion-exclusion, adjoint, coverage)

Parameter budget: < 105k trainable. Expected ~101,763.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import backbone, FPN, fine-head from frozen RMR-v2 (do NOT modify rmr_count)
from rmr_count.model import (
    AdditiveFPNNeck,
    FineMeasureHead,
    MobileNetV4Backbone,
    count_parameters,
)
from rmr_count.operators import (
    RegionSet,
    build_multiscale_regions,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_adjoint,
    regional_sum,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FINE_HEAD_BIAS_INIT: float = math.log(math.expm1(0.015763))  # matches RMR-v2 M0 prior
_DISPERSION_INIT: float = 50.0
_DISPERSION_MIN: float = 0.5
_DISPERSION_MAX: float = 500.0
_RATE_STD_FLOOR: float = 0.01
_W_MIN: float = 0.25
_W_MAX: float = 4.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RMRv3Config:
    """Frozen specification for RMR-v3."""

    # Backbone
    backbone_name: str = "mobilenetv4_conv_small_050.e3000_r224_in1k"
    pretrained: bool = False          # set True to load ImageNet weights
    backbone_lr_scale: float = 0.1

    # FPN / head
    output_stride: int = 4
    feature_width: int = 32

    # Regional geometry (no full-image)
    region_sizes_px: tuple[int, ...] = (32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = False   # MUST remain False

    # Regional head
    regional_hidden: int = 48          # hidden dim of shared MLP

    # Probabilistic reliability
    dispersion_init: float = _DISPERSION_INIT
    dispersion_min: float = _DISPERSION_MIN
    dispersion_max: float = _DISPERSION_MAX
    rate_std_floor: float = _RATE_STD_FLOOR

    # Reliability weighting clamp
    w_min: float = _W_MIN
    w_max: float = _W_MAX

    # Scale-neutral normalization (normalize weights within each scale family)
    scale_neutral_normalize: bool = True

    # Solver
    iterations: int = 2
    sirt_omega: float = 1.0            # fixed, not learnable

    # Detach policy (registered main variant)
    detach_region_mean_in_solver: bool = True
    detach_reliability_in_solver: bool = True

    # V3-A control: uniform weights (ablation)
    uniform_reliability: bool = False  # False = V3-B (main); True = V3-A (control)

    # Count prior
    init_m0: float = 0.015763

    eps: float = 1e-6


# ---------------------------------------------------------------------------
# Reliability-augmented regional evidence head
# ---------------------------------------------------------------------------
class ReliabilityRegionalHead(nn.Module):
    """Regional head that outputs both rate (b_R) and reliability (w_R).

    Architecture:
        Shared MLP trunk (identical to RMR-v2 ScaleMatchedRegionalEvidenceHead):
            Linear(33, 48) -> SiLU -> Linear(48, 48) -> SiLU
        Two separate output heads (both Linear(48->1)):
            rate_head:        rho_R = softplus(raw_rate)  -> b_R = |R| * rho_R
            reliability_head: w_R via bounded sigmoid: w_min + (w_max - w_min) * sigmoid(raw_w)

    The reliability_head is the ONLY new parameter over RMR-v2 (49 params: 48 weights + 1 bias).

    Per-region variance estimate (probabilistic NB model):
        var_R = rho_R + rho_R^2 / r         [r = dispersion parameter]
        std_R = sqrt(var_R).clamp_min(rate_std_floor)
        -> Used for diagnostics (Section 32) but NOT as the weight.
          The learned weight w_R is the registered mechanism, not the analytic variance.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden: int = 48,
        region_sizes_px: Sequence[int] = (32, 64, 128),
        init_bias: float = _FINE_HEAD_BIAS_INIT,
        dispersion_init: float = _DISPERSION_INIT,
        dispersion_min: float = _DISPERSION_MIN,
        dispersion_max: float = _DISPERSION_MAX,
        rate_std_floor: float = _RATE_STD_FLOOR,
        w_min: float = _W_MIN,
        w_max: float = _W_MAX,
    ):
        super().__init__()
        self.region_sizes_px = tuple(int(s) for s in region_sizes_px)
        self.rate_std_floor = rate_std_floor
        self.w_min = w_min
        self.w_max = w_max

        # Shared trunk: Linear(33,48) -> SiLU -> Linear(48,48) -> SiLU
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )

        # Rate head: Linear(48,1)  [same as RMR-v2 final layer]
        self.rate_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.rate_head.weight, std=0.01)
        nn.init.constant_(self.rate_head.bias, init_bias)

        # Reliability head: Linear(48,1)  [new in RMR-v3, +49 params]
        self.reliability_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.reliability_head.weight)
        nn.init.zeros_(self.reliability_head.bias)   # sigmoid(0)=0.5 -> w ~ mid-range at init

        # Dispersion parameter (per-region scalar NB model, non-negative)
        log_r_init = math.log(float(dispersion_init))
        self.log_dispersion = nn.Parameter(torch.tensor(log_r_init))
        self._disp_min = float(dispersion_min)
        self._disp_max = float(dispersion_max)

    @property
    def dispersion(self) -> torch.Tensor:
        return self.log_dispersion.exp().clamp(self._disp_min, self._disp_max)

    def _extract_features(
        self,
        p4: torch.Tensor,
        p8: torch.Tensor,
        p16: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        """Extract region features aligned on P4 coordinate grid."""
        from rmr_count.operators import region_average_features

        b = p4.shape[0]
        device = p4.device
        dtype = p4.dtype
        size4 = p4.shape[-2:]

        # Upsample all pyramid levels to P4 resolution
        feat_p8_at_4 = (
            p8 if p8.shape[-2:] == size4
            else F.interpolate(p8, size=size4, mode="bilinear", align_corners=False)
        )
        feat_p16_at_4 = (
            p16 if p16.shape[-2:] == size4
            else F.interpolate(p16, size=size4, mode="bilinear", align_corners=False)
        )

        m_total = regions.boxes.shape[0]
        in_dim = self.trunk[0].in_features  # feature_dim + 1
        feat_all = torch.zeros((b, m_total, in_dim), device=device, dtype=dtype)

        for sid, size_px in enumerate(self.region_sizes_px):
            mask = regions.scale_id == sid
            if not mask.any():
                continue
            feat_s = p4 if size_px <= 48 else (feat_p8_at_4 if size_px <= 96 else feat_p16_at_4)
            boxes_s = regions.boxes[mask]
            u_s = region_average_features(feat_s, boxes_s)  # [B, M_s, C]
            m_s = u_s.shape[1]
            scale_feat = torch.full(
                (1, m_s, 1),
                math.log(float(size_px) / 32.0),
                device=device, dtype=dtype,
            ).expand(b, -1, -1)
            feat_all[:, mask] = torch.cat([u_s, scale_feat], dim=-1)

        return feat_all  # [B, M, 33]

    def forward(
        self,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        regions: RegionSet,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (b_region, w_region, rate).

        Returns:
            b_region: [B, 1, M]  regional count prediction
            w_region: [B, 1, M]  reliability weights in [w_min, w_max]
            rate:     [B, 1, M]  rate rho_R (count/stride-4 cell), for diagnostics
        """
        p4, p8, p16 = pyramid
        feat_all = self._extract_features(p4, p8, p16, regions)  # [B, M, 33]

        hidden = self.trunk(feat_all)          # [B, M, 48]
        raw_rate = self.rate_head(hidden)      # [B, M, 1]
        raw_w = self.reliability_head(hidden)  # [B, M, 1]

        rate = F.softplus(raw_rate).squeeze(-1)  # [B, M]  rho_R >= 0
        area = regions.area.to(dtype=rate.dtype).view(1, -1)  # [1, M]
        b_region = rate * area                   # [B, M]

        # Reliability weight: bounded sigmoid
        w_region = (
            self.w_min + (self.w_max - self.w_min) * torch.sigmoid(raw_w.squeeze(-1))
        )  # [B, M]

        return (
            b_region.unsqueeze(1),   # [B, 1, M]
            w_region.unsqueeze(1),   # [B, 1, M]
            rate.unsqueeze(1),       # [B, 1, M]
        )


# ---------------------------------------------------------------------------
# Scale-neutral normalization helper
# ---------------------------------------------------------------------------
def _scale_neutral_normalize(
    w: torch.Tensor,
    regions: RegionSet,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize reliability weights within each scale family to mean ~1.

    For each scale s: w_normalized_R = w_R / (mean_{R in scale s} w_R)

    Mathematical guarantee (Section 9):
        If all regions have the same residual density delta, then r = delta * 1
        regardless of w_R after normalization. This guarantees spatial uniformity
        is preserved under uniform residual, independently of learned weights.

    Args:
        w: [B, 1, M] raw reliability weights
        regions: RegionSet with scale_id [M]
    Returns:
        w_norm: [B, 1, M] normalized weights, mean ~1 per scale per image
    """
    w_norm = w.clone()
    for sid in torch.unique(regions.scale_id):
        mask = regions.scale_id == sid   # [M]
        if not mask.any():
            continue
        w_s = w[:, :, mask]              # [B, 1, M_s]
        mean_s = w_s.mean(dim=-1, keepdim=True).clamp_min(eps)  # [B, 1, 1]
        w_norm[:, :, mask] = w_s / mean_s
    return w_norm


# ---------------------------------------------------------------------------
# Weighted adjoint operator (FP32 enforced)
# ---------------------------------------------------------------------------
def _weighted_adjoint(
    residual: torch.Tensor,  # [B, 1, M]  r_R = (AY - b)_R
    weights: torch.Tensor,   # [B, 1, M]  w_R in [w_min, w_max]
    regions: RegionSet,
    H: int,
    W: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute reliability-weighted normalized adjoint field in FP32.

    Standard (unweighted) adjoint (RMR-v2 B5-P):
        r_cell = D_c^{-1} A^T D_a^{-1} (AY - b)

    Reliability-weighted adjoint (RMR-v3):
        r_cell = D_c(w)^{-1} A^T_w D_a^{-1} (AY - b)

    where:
        D_a: diagonal of region areas |R| (count -> rate conversion)
        A^T_w: weighted adjoint — each region R contributes its rate residual
               scaled by w_R before scatter to grid cells
        D_c(w): weighted coverage — normalization by sum of w_R for each cell

    Uses regional_adjoint for O(M+HW) vectorized scatter in FP32.

    Args:
        residual: [B, 1, M] regional count residual (AY - b)
        weights:  [B, 1, M] per-region reliability weights
        regions:  RegionSet
        H, W:     spatial dimensions (stride-4 grid)
    Returns:
        field: [B, 1, H, W] weighted adjoint field
    """
    boxes = regions.boxes  # [M, 4]
    area = regions.area.to(device=residual.device, dtype=torch.float32).clamp_min(1.0)
    # [M] -> [1, 1, M] for broadcasting with [B, 1, M]
    area_bcast = area.view(1, 1, -1)

    # Rate residual in FP32: [B, 1, M]
    res_f32 = residual.to(dtype=torch.float32)
    w_f32 = weights.to(dtype=torch.float32)
    rate_res = res_f32 / area_bcast  # [B, 1, M]

    # Weighted rate residual: w_R * (delta_R / |R|)
    weighted_rate_res = w_f32 * rate_res  # [B, 1, M]

    # Vectorized weighted scatter via regional_adjoint (O(M + HW) difference arrays)
    # Result: sum_R w_R * rate_res_R for each cell p in R  -> [B, 1, H, W]
    weighted_field = regional_adjoint(
        weighted_rate_res, boxes, H, W, out_dtype=torch.float32
    )

    # Weighted coverage: sum_R w_R for each cell p in R  -> [B, 1, H, W]
    coverage = regional_adjoint(
        w_f32, boxes, H, W, out_dtype=torch.float32
    )

    # Normalize: D_c(w)^{-1} weighted adjoint
    field_normalized = weighted_field / coverage.clamp_min(eps)  # [B, 1, H, W]
    return field_normalized


# ---------------------------------------------------------------------------
# Main RMR-v3 model
# ---------------------------------------------------------------------------
class RMRv3(nn.Module):
    r"""RMR-v3: Reliability-Weighted Regional Measure Reconciliation.

    Forward pass:
        1. Backbone (frozen pretrained) -> feature pyramid (P4, P8, P16)
        2. FineMeasureHead -> Y0 (stride-4 positive count-per-cell map)
        3. ReliabilityRegionalHead -> b_R, w_R (regional count + reliability)
        4. T=2 nonnegative projected SIRT with weighted adjoint:
               for t in range(T):
                   AY = regional_sum(Y)           [B, 1, M]
                   residual = AY - b_R            [B, 1, M]  (FP32)
                   r = weighted_adjoint(residual, w_R, ...)  [B, 1, H, W]
                   Y = max(0, Y - omega * r)      [B, 1, H, W]  (nonneg projection)
        5. Return Y (final), Y0 (observer), b_region, w_region, iterates

    Detach policy (registered main variant):
        detach_region_mean_in_solver=True:  b_R detached in AY-b computation
        detach_reliability_in_solver=True:  w_R detached in weighted adjoint
        -> Regional head trained via region_head loss (not solver gradient)
        -> Solver step has no gradient to regional parameters at inference time
    """

    def __init__(self, cfg: RMRv3Config = RMRv3Config()):
        super().__init__()
        if cfg.output_stride != 4:
            raise ValueError(f"RMRv3 requires output_stride=4, got {cfg.output_stride}")
        if cfg.include_full_image:
            raise ValueError("RMRv3: include_full_image must be False (tiling consistency)")
        self.cfg = cfg

        # Count prior init
        init_bias = math.log(math.expm1(float(cfg.init_m0)))

        # Submodules
        self.backbone = MobileNetV4Backbone(
            model_name=cfg.backbone_name,
            pretrained=cfg.pretrained,
        )
        self.neck = AdditiveFPNNeck(
            in_channels=self.backbone.out_channels,
            width=cfg.feature_width,
        )
        self.fine_head = FineMeasureHead(
            width=cfg.feature_width,
            init_bias=init_bias,
        )
        self.region_head = ReliabilityRegionalHead(
            feature_dim=cfg.feature_width,
            hidden=cfg.regional_hidden,
            region_sizes_px=cfg.region_sizes_px,
            init_bias=init_bias,
            dispersion_init=cfg.dispersion_init,
            dispersion_min=cfg.dispersion_min,
            dispersion_max=cfg.dispersion_max,
            rate_std_floor=cfg.rate_std_floor,
            w_min=cfg.w_min,
            w_max=cfg.w_max,
        )

    def _build_regions(self, H: int, W: int, device: torch.device) -> RegionSet:
        return build_multiscale_regions(
            H, W,
            output_stride=self.cfg.output_stride,
            region_sizes_px=self.cfg.region_sizes_px,
            overlap=self.cfg.region_overlap,
            include_full_image=False,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        regions: RegionSet | None = None,
    ) -> dict:
        cfg = self.cfg
        B, C, H_img, W_img = x.shape
        device = x.device
        orig_dtype = x.dtype

        # 1. Feature extraction
        c4, c8, c16 = self.backbone(x)       # (C4, C8, C16) tuple
        p4, p8, p16 = self.neck(c4, c8, c16) # (P4, P8, P16) tuple
        pyramid = (p4, p8, p16)

        H4, W4 = p4.shape[-2:]       # stride-4 grid

        # 2. Fine measure head
        y0 = self.fine_head(p4)      # [B, 1, H4, W4]

        # 3. Build regions if not provided
        if regions is None:
            regions = self._build_regions(H4, W4, device)

        # 4. Regional evidence + reliability
        b_region, w_region, rate = self.region_head(pyramid, regions)
        # b_region: [B, 1, M], w_region: [B, 1, M], rate: [B, 1, M]

        # 5. Scale-neutral normalization of weights
        if cfg.scale_neutral_normalize and not cfg.uniform_reliability:
            w_region = _scale_neutral_normalize(w_region, regions, eps=cfg.eps)

        # Uniform reliability (V3-A control): override weights to 1.0
        if cfg.uniform_reliability:
            w_region = torch.ones_like(w_region)

        # 6. Nonnegative projected SIRT with weighted adjoint
        y = y0
        iterates = [y0]

        for _t in range(cfg.iterations):
            # Regional sum in FP32 (prefix accumulation)
            ay = regional_sum(y, regions.boxes)  # [B, 1, M] — channel dim preserved

            # Detach policy
            b_for_solver = b_region.detach() if cfg.detach_region_mean_in_solver else b_region
            w_for_solver = w_region.detach() if cfg.detach_reliability_in_solver else w_region

            residual = ay.to(dtype=torch.float32) - b_for_solver.to(dtype=torch.float32)
            # [B, 1, M]

            # Weighted adjoint field (FP32, then cast back)
            r_field = _weighted_adjoint(
                residual,
                w_for_solver,
                regions,
                H4, W4,
                eps=cfg.eps,
            )  # [B, 1, H4, W4] FP32

            # Projected SIRT step: Y^{t+1} = max(0, Y^t - omega * r)
            y_f32 = y.to(dtype=torch.float32)
            y_new = (y_f32 - cfg.sirt_omega * r_field).clamp_min(0.0)
            y = y_new.to(dtype=orig_dtype)
            iterates.append(y)

        return {
            "y": y,                    # [B, 1, H4, W4] final refined measure
            "y0": y0,                  # [B, 1, H4, W4] initial fine measure
            "b_region": b_region,      # [B, 1, M] regional count prediction
            "w_region": w_region,      # [B, 1, M] reliability weights (after normalization)
            "rate": rate,              # [B, 1, M] rate rho_R (for diagnostics)
            "iterates": iterates,      # list of T+1 tensors
            "regions": regions,        # RegionSet
            "dispersion": self.region_head.dispersion,  # scalar
        }

    def parameter_groups(self) -> list[dict]:
        """Return parameter groups for optimizer with backbone LR scaling."""
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr_scale": self.cfg.backbone_lr_scale},
            {"params": other_params,    "lr_scale": 1.0},
        ]
