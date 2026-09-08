from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.backbones import MobileNetV4Backbone
from rmr_core.heads import FineMeasureHead
from rmr_core.necks import AdditiveFPNNeck
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    region_average_features,
    regional_adjoint,
    regional_sum,
)


def _softplus_inverse(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


@dataclass
class RMRv3Config:
    # Fine grid / carrier
    output_stride: int = 4
    feature_width: int = 32
    backbone_name: str = "mobilenetv4_conv_small_050.e3000_r224_in1k"
    pretrained: bool = True
    backbone_lr_scale: float = 0.1
    init_m0: float = 0.015763

    # Region dictionary
    region_sizes_px: tuple[int, ...] = (32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = False

    # Solver
    iterations: int = 2
    omega: float = 1.0
    residual_clip: float = 0.0
    eps: float = 1e-6

    # Negative-Binomial regional uncertainty
    dispersion_init: float = 50.0
    dispersion_min: float = 0.5
    dispersion_max: float = 500.0

    # Reliability
    reliability_mode: str = "nb_rate_variance"
    reliability_rate_std_floor: float = 0.01
    reliability_weight_min: float = 0.25
    reliability_weight_max: float = 4.0
    normalize_reliability_within_scale: bool = True

    # Registered clean-causal variant
    detach_region_mean_in_solver: bool = True
    detach_reliability_in_solver: bool = True

    # V4 candidate switches
    native_scale_pooling: bool = False
    regional_feature_stats: str = "mean"


def _map_boxes_between_grids(
    boxes: torch.Tensor,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    """Accurately project bounding boxes from source lattice to destination lattice."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw

    b = boxes.float()

    y1 = torch.floor(b[:, 0] * (dst_h / src_h))
    x1 = torch.floor(b[:, 1] * (dst_w / src_w))
    y2 = torch.ceil(b[:, 2] * (dst_h / src_h))
    x2 = torch.ceil(b[:, 3] * (dst_w / src_w))

    y1 = y1.clamp(0, max(dst_h - 1, 0))
    x1 = x1.clamp(0, max(dst_w - 1, 0))
    y2 = y2.clamp(1, dst_h)
    x2 = x2.clamp(1, dst_w)

    y2 = torch.maximum(y2, y1 + 1)
    x2 = torch.maximum(x2, x1 + 1)

    return torch.stack([y1, x1, y2, x2], dim=-1).long()


def region_mean_std_features(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Extract both spatial mean and standard deviation over bounding boxes."""
    mean = region_average_features(
        feature,
        boxes,
    ).float()

    mean_sq = region_average_features(
        feature.float().square(),
        boxes,
    )

    var = (
        mean_sq - mean.square()
    ).clamp_min(0.0)

    std = torch.sqrt(var + eps)

    return torch.cat(
        [mean, std],
        dim=-1,
    ).to(feature.dtype)


class ProbabilisticRegionalEvidenceHead(nn.Module):
    """Predict regional NB mean and dispersion.

    Mean:
        rate_R = softplus(mean_raw)
        mu_R   = area_R * rate_R

    Dispersion:
        log_r_R = bounded log-dispersion
        r_R     = exp(log_r_R)

    Reliability is not directly predicted.
    It is derived from the NB predictive variance.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden: int = 48,
        init_rate: float = 0.015763,
        region_sizes_px: tuple[int, ...] = (32, 64, 128),
        dispersion_init: float = 50.0,
        dispersion_min: float = 0.5,
        dispersion_max: float = 500.0,
        native_scale_pooling: bool = False,
        regional_feature_stats: str = "mean",
    ) -> None:
        super().__init__()

        if dispersion_min <= 0:
            raise ValueError("dispersion_min must be > 0")
        if dispersion_max <= dispersion_min:
            raise ValueError("dispersion_max must be > dispersion_min")
        if not (dispersion_min <= dispersion_init <= dispersion_max):
            raise ValueError("dispersion_init must lie inside [min,max]")
        if regional_feature_stats not in ("mean", "mean_std"):
            raise ValueError(f"regional_feature_stats must be 'mean' or 'mean_std', got '{regional_feature_stats}'")

        self.region_sizes_px = tuple(int(x) for x in region_sizes_px)
        self.dispersion_min = float(dispersion_min)
        self.dispersion_max = float(dispersion_max)
        self.native_scale_pooling = bool(native_scale_pooling)
        self.regional_feature_stats = str(regional_feature_stats)

        in_dim = feature_dim + 1 if self.regional_feature_stats == "mean" else 2 * feature_dim + 1

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )

        self.mean_head = nn.Linear(hidden, 1)
        self.log_dispersion_head = nn.Linear(hidden, 1)

        # Mean initialization: same empirical rate prior as fine head.
        nn.init.normal_(self.mean_head.weight, std=0.01)
        nn.init.constant_(
            self.mean_head.bias,
            _softplus_inverse(init_rate),
        )

        # Dispersion initialization.
        nn.init.zeros_(self.log_dispersion_head.weight)
        nn.init.constant_(
            self.log_dispersion_head.bias,
            math.log(float(dispersion_init)),
        )

    def _collect_region_features(
        self,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        regions: RegionSet,
    ) -> torch.Tensor:
        p4, p8, p16 = pyramid

        b = p4.shape[0]
        device = p4.device
        dtype = p4.dtype

        src_hw = p4.shape[-2:]

        m_total = int(regions.boxes.shape[0])
        feature_dim = int(p4.shape[1])
        out_dim = feature_dim + 1 if self.regional_feature_stats == "mean" else 2 * feature_dim + 1

        out = torch.zeros(
            (b, m_total, out_dim),
            device=device,
            dtype=dtype,
        )

        level_by_sid = {
            0: p4,
            1: p8,
            2: p16,
        }

        for sid, size_px in enumerate(self.region_sizes_px):
            mask = regions.scale_id == sid
            if not bool(mask.any()):
                continue

            feat = level_by_sid.get(sid, p16)
            boxes4 = regions.boxes[mask]

            if self.native_scale_pooling and feat.shape[-2:] != src_hw:
                boxes_level = _map_boxes_between_grids(
                    boxes4,
                    src_hw,
                    feat.shape[-2:],
                )
            else:
                if feat.shape[-2:] != src_hw:
                    feat = F.interpolate(
                        feat,
                        size=src_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                boxes_level = boxes4

            if self.regional_feature_stats == "mean_std":
                pooled = region_mean_std_features(
                    feat,
                    boxes_level,
                )
            else:
                pooled = region_average_features(
                    feat,
                    boxes_level,
                )

            ms = pooled.shape[1]

            log_scale = torch.full(
                (1, ms, 1),
                math.log(float(size_px) / 32.0),
                device=device,
                dtype=dtype,
            ).expand(b, -1, -1)

            out[:, mask] = torch.cat(
                [pooled, log_scale],
                dim=-1,
            )

        # Main method disables full-image regions.
        # Keep an explicit guard so a bad config cannot silently proceed.
        if bool((regions.scale_id == -1).any()):
            raise RuntimeError(
                "RMR-v3 registered method does not support full-image regions"
            )

        return out

    def forward(
        self,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        regions: RegionSet,
    ) -> dict[str, torch.Tensor]:
        x = self._collect_region_features(
            pyramid,
            regions,
        )

        h = self.trunk(x)

        mean_raw = self.mean_head(h).squeeze(-1)

        rate = F.softplus(mean_raw)

        area = regions.area.to(
            device=rate.device,
            dtype=rate.dtype,
        ).view(1, -1)

        mu_count = rate * area

        log_r = self.log_dispersion_head(h).squeeze(-1)
        log_r = log_r.clamp(
            min=math.log(self.dispersion_min),
            max=math.log(self.dispersion_max),
        )

        dispersion = torch.exp(log_r)

        return {
            "mu_count": mu_count.unsqueeze(1),       # [B,1,M]
            "rate": rate.unsqueeze(1),              # [B,1,M]
            "dispersion": dispersion.unsqueeze(1),  # [B,1,M]
            "log_dispersion": log_r.unsqueeze(1),   # [B,1,M]
        }


def reliability_from_nb(
    mu_count: torch.Tensor,
    dispersion: torch.Tensor,
    regions: RegionSet,
    *,
    rate_std_floor: float = 0.01,
    weight_min: float = 0.25,
    weight_max: float = 4.0,
    normalize_within_scale: bool = True,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Derive regional reliability from NB predictive rate variance.

    NB count variance:
        Var[N] = mu + mu^2 / r

    Rate variance:
        Var[N / area] = Var[N] / area^2

    Precision:
        q = 1 / (rate_var + floor^2)

    Main method normalizes q to mean 1 inside each scale family.
    """

    mu = mu_count.float().clamp_min(0.0)
    r = dispersion.float().clamp_min(eps)

    area = regions.area.float().view(1, 1, -1)
    area = area.clamp_min(1.0)

    count_var = mu + mu.square() / r

    floor_var = float(rate_std_floor) ** 2

    rate_var = count_var / area.square()
    rate_var = rate_var + floor_var

    precision = 1.0 / rate_var.clamp_min(eps)

    if normalize_within_scale:
        weight = torch.empty_like(precision)

        for sid in torch.unique(regions.scale_id):
            if int(sid.item()) < 0:
                continue

            mask = regions.scale_id == sid

            q = precision[..., mask]

            q_mean = q.mean(
                dim=-1,
                keepdim=True,
            ).clamp_min(eps)

            weight[..., mask] = q / q_mean
    else:
        weight = precision / precision.mean(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)

    weight = weight.clamp(
        min=float(weight_min),
        max=float(weight_max),
    )

    return {
        "weight": weight,
        "precision": precision,
        "rate_variance": rate_var,
        "count_variance": count_var,
    }


def weighted_coverage(
    weight: torch.Tensor,
    regions: RegionSet,
    height: int,
    width: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute D_{c,w} diagonal field = A^T w."""

    cov = regional_adjoint(
        weight.float(),
        regions.boxes,
        height,
        width,
        out_dtype=torch.float32,
    )

    return cov.clamp_min(float(eps))


def weighted_normalized_adjoint_field(
    y: torch.Tensor,
    b_region: torch.Tensor,
    weight: torch.Tensor,
    regions: RegionSet,
    *,
    weighted_cov: torch.Tensor | None = None,
    residual_clip: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute:

        r = D_cw^-1 A^T W D_a^-1 (A y - b)

    entirely in float32.
    """

    _, _, h, w = y.shape

    y32 = y.float()
    b32 = b_region.float()
    weight32 = weight.float()

    q = regional_sum(
        y32,
        regions.boxes,
        out_dtype=torch.float32,
    )

    delta = q - b32

    area = regions.area.float().view(1, 1, -1)
    rate_residual = delta / area.clamp_min(1.0)

    weighted_residual = weight32 * rate_residual

    back = regional_adjoint(
        weighted_residual,
        regions.boxes,
        h,
        w,
        out_dtype=torch.float32,
    )

    if weighted_cov is None:
        weighted_cov = weighted_coverage(
            weight32,
            regions,
            h,
            w,
            eps=eps,
        )

    field = back / weighted_cov.float().clamp_min(eps)

    if residual_clip > 0:
        field = field.clamp(
            -float(residual_clip),
            float(residual_clip),
        )

    return field


def weighted_regional_energy(
    y: torch.Tensor,
    b_region: torch.Tensor,
    weight: torch.Tensor,
    regions: RegionSet,
) -> torch.Tensor:
    """Per-sample weighted regional energy.

        E = 1/2 sum_R w_R * (Ay-b)^2 / area_R
    """

    q = regional_sum(
        y.float(),
        regions.boxes,
        out_dtype=torch.float32,
    )

    delta = q - b_region.float()

    area = regions.area.float().view(1, 1, -1)

    energy = 0.5 * (
        weight.float()
        * delta.square()
        / area.clamp_min(1.0)
    ).sum(dim=(-2, -1))

    return energy


class RMRv3(nn.Module):
    """Reliability-Weighted Regional Measure Reconciliation."""

    def __init__(
        self,
        cfg: RMRv3Config | None = None,
    ) -> None:
        super().__init__()

        if cfg is None:
            cfg = RMRv3Config()

        if cfg.output_stride != 4:
            raise ValueError(
                "RMR-v3 registered method requires output_stride=4"
            )

        if cfg.include_full_image:
            raise ValueError(
                "RMR-v3 registered method requires include_full_image=False"
            )

        if cfg.iterations < 1:
            raise ValueError("iterations must be >= 1")

        if cfg.omega <= 0:
            raise ValueError("omega must be > 0")

        if cfg.reliability_mode != "nb_rate_variance":
            raise ValueError(
                f"Unsupported reliability_mode: {cfg.reliability_mode}. Only 'nb_rate_variance' is supported."
            )

        if tuple(cfg.region_sizes_px) != (32, 64, 128):
            raise ValueError(
                f"RMR-v3 registered canonical method requires region_sizes_px=(32, 64, 128), got {cfg.region_sizes_px}"
            )

        if cfg.reliability_weight_min <= 0:
            raise ValueError(
                f"reliability_weight_min ({cfg.reliability_weight_min}) must be > 0"
            )

        if not (cfg.reliability_weight_min < cfg.reliability_weight_max):
            raise ValueError(
                f"reliability_weight_min ({cfg.reliability_weight_min}) must be < reliability_weight_max ({cfg.reliability_weight_max})"
            )

        if cfg.reliability_rate_std_floor <= 0:
            raise ValueError(
                f"reliability_rate_std_floor ({cfg.reliability_rate_std_floor}) must be > 0"
            )

        self.cfg = cfg

        self.encoder = MobileNetV4Backbone(
            model_name=cfg.backbone_name,
            pretrained=cfg.pretrained,
            target_reductions=(4, 8, 16),
        )

        self.fusion = AdditiveFPNNeck(
            in_channels=self.encoder.out_channels,
            width=cfg.feature_width,
        )

        init_bias = _softplus_inverse(cfg.init_m0)

        self.fine_head = FineMeasureHead(
            width=cfg.feature_width,
            init_bias=init_bias,
        )

        self.region_head = ProbabilisticRegionalEvidenceHead(
            feature_dim=cfg.feature_width,
            hidden=48,
            init_rate=cfg.init_m0,
            region_sizes_px=cfg.region_sizes_px,
            dispersion_init=cfg.dispersion_init,
            dispersion_min=cfg.dispersion_min,
            dispersion_max=cfg.dispersion_max,
            native_scale_pooling=cfg.native_scale_pooling,
            regional_feature_stats=cfg.regional_feature_stats,
        )

        self.solver_strength: float = 1.0

        self._region_cache: OrderedDict[
            tuple,
            RegionSet,
        ] = OrderedDict()

    def set_solver_strength(self, strength: float) -> None:
        self.solver_strength = float(min(max(strength, 0.0), 1.0))

    def _regions(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> RegionSet:
        key = (
            h,
            w,
            self.cfg.output_stride,
            self.cfg.region_sizes_px,
            self.cfg.region_overlap,
            device.type,
            device.index if device.type == "cuda" else None,
        )

        if key in self._region_cache:
            self._region_cache.move_to_end(key)
            return self._region_cache[key]

        if len(self._region_cache) >= 32:
            self._region_cache.popitem(last=False)

        region_set = build_multiscale_regions(
            height=h,
            width=w,
            output_stride=self.cfg.output_stride,
            region_sizes_px=self.cfg.region_sizes_px,
            overlap=self.cfg.region_overlap,
            include_full_image=False,
            device=device,
        )
        self._region_cache[key] = region_set
        return region_set

    def forward(
        self,
        x: torch.Tensor,
        *,
        uniform_reliability: bool = False,
        solver_strength: float | None = None,
    ) -> dict:
        c4, c8, c16 = self.encoder(x)

        p4, p8, p16 = self.fusion(
            c4,
            c8,
            c16,
        )

        z0 = self.fine_head(p4)
        y0 = F.softplus(z0)

        h, w = y0.shape[-2:]

        regions = self._regions(
            h,
            w,
            x.device,
        )

        regional = self.region_head(
            (p4, p8, p16),
            regions,
        )

        mu_count = regional["mu_count"]
        dispersion = regional["dispersion"]

        reliability = reliability_from_nb(
            mu_count,
            dispersion,
            regions,
            rate_std_floor=self.cfg.reliability_rate_std_floor,
            weight_min=self.cfg.reliability_weight_min,
            weight_max=self.cfg.reliability_weight_max,
            normalize_within_scale=(
                self.cfg.normalize_reliability_within_scale
            ),
            eps=self.cfg.eps,
        )

        weight = reliability["weight"]

        # V3-A control: probabilistic head, uniform solver.
        if uniform_reliability:
            weight_solver = torch.ones_like(weight)
        else:
            weight_solver = weight

        if self.cfg.detach_region_mean_in_solver:
            b_solver = mu_count.detach()
        else:
            b_solver = mu_count

        if self.cfg.detach_reliability_in_solver:
            weight_solver = weight_solver.detach()

        cov_w = weighted_coverage(
            weight_solver,
            regions,
            h,
            w,
            eps=self.cfg.eps,
        )

        strength = float(
            self.solver_strength
            if solver_strength is None
            else min(max(solver_strength, 0.0), 1.0)
        )
        effective_omega = float(self.cfg.omega) * strength

        y = y0

        iterates = [y0]
        residual_fields = []
        energy_trace = []

        for _ in range(self.cfg.iterations):
            energy_before = weighted_regional_energy(
                y,
                b_solver,
                weight_solver,
                regions,
            )

            field = weighted_normalized_adjoint_field(
                y,
                b_solver,
                weight_solver,
                regions,
                weighted_cov=cov_w,
                residual_clip=self.cfg.residual_clip,
                eps=self.cfg.eps,
            )

            y_next = torch.clamp_min(
                y.float()
                - effective_omega * field,
                0.0,
            ).to(y.dtype)

            energy_after = weighted_regional_energy(
                y_next,
                b_solver,
                weight_solver,
                regions,
            )

            energy_trace.append(
                {
                    "before": energy_before,
                    "after": energy_after,
                }
            )

            residual_fields.append(field)
            iterates.append(y_next)
            y = y_next

        return {
            "y": y,
            "y0": y0,
            "z0": z0,

            "regions": regions,

            "b_region": mu_count,
            "region_rate": regional["rate"],
            "region_dispersion": dispersion,
            "region_log_dispersion": regional["log_dispersion"],

            "region_weight": weight,
            "solver_region_weight": weight_solver,
            "region_precision": reliability["precision"],
            "region_rate_variance": reliability["rate_variance"],
            "region_count_variance": reliability["count_variance"],

            "iterates": iterates,
            "residual_fields": residual_fields,
            "energy_trace": energy_trace,

            "uniform_reliability": uniform_reliability,
            "solver_strength": strength,
        }
