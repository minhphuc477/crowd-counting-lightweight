from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.backbones import MobileNetV4Backbone
from rmr_core.heads import FineMeasureHead
from rmr_core.necks import AdditiveFPNNeck, ASPPLiteFPNNeck, RepWeightedFPNNeck
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    fractional_region_average_features,
    fractional_region_mean_std_features,
    region_average_features,
    region_mean_std_features,
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

    # Neck
    neck_type: str = "additive"  # "additive" | "aspp_lite" | "rep_weighted"
    context_dilations: tuple[int, ...] = (1, 2, 3)  # dilations for AdditiveFPNNeck / RepWeightedFPNNeck
    use_aspp_gap: bool = False  # if True AND neck_type=="aspp_lite", enable GAP branch in ASPP-lite
    aspp_dilations: tuple[int, ...] = (1, 3, 6)  # dilations for ASPPLiteFPNNeck branches

    # Region dictionary
    region_sizes_px: tuple[int, ...] = (32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = False

    # Solver
    enable_solver: bool = True
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

    # Region head MLP hidden dim (default 48 preserves all prior checkpoints)
    region_head_hidden: int = 48

    # ── RMR-v7 additions ──────────────────────────────────────────────────────
    # Hurdle NB head: predicts region occupancy probability π_R.
    # When enabled, the solver target for background regions is zeroed:
    #     b_solver_R = (1 - sigmoid(z_π_R)).detach() * mu_count_R.detach()
    # This eliminates systematic under-counting from ASPP wide receptive field.
    hurdle_head: bool = False

    # Total-variation Laplacian smoothing coefficient inside the SIRT loop.
    # Applied after each SIRT update: y ← y - tv_lambda * Δy  (Δ = Laplacian)
    # 0.0 disables smoothing (default, backward compatible).
    tv_lambda: float = 0.0

    # EMA weight tracking decay (0.0 = disabled, typical 0.999).
    # When > 0, train.py maintains an EMA shadow copy and saves it as best checkpoint.
    # Decouples best-epoch tracking from weight drift observed after epoch 445 in v6.
    ema_decay: float = 0.0

    # Learnable Temperature Softplus for FineMeasureHead.
    # When True, FineMeasureHead uses τ * softplus(z/τ) instead of softplus(z).
    # Fixes negative bias saturation at high density (Bias=-14.72 in v6 → near 0 target).
    temp_softplus: bool = False




class ProbabilisticRegionalEvidenceHead(nn.Module):
    """Predict regional NB mean and dispersion, and optionally occupancy (Hurdle).

    Mean:
        rate_R = softplus(mean_raw)
        mu_R   = area_R * rate_R

    Dispersion:
        log_r_R = bounded log-dispersion
        r_R     = exp(log_r_R)

    Hurdle (RMR-v7+, opt-in via hurdle_head=True):
        z_π_R  = hurdle_head_layer(h)   -- occupancy logit
        π_R    = sigmoid(z_π_R)         -- P(region is occupied)
        Used in solver: b_solver_R = π_R * mu_count_R  (background regions zeroed)

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
        hurdle_head: bool = False,
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
        self.has_hurdle_head = bool(hurdle_head)

        in_dim = feature_dim + 1 if self.regional_feature_stats == "mean" else 2 * feature_dim + 1

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )

        self.mean_head = nn.Linear(hidden, 1)
        self.log_dispersion_head = nn.Linear(hidden, 1)

        # Optional hurdle head: predicts occupancy logit z_π_R
        if self.has_hurdle_head:
            self.hurdle_head_layer = nn.Linear(hidden, 1)
            # Initialize near p=0.5 (logit=0) so early training is neutral
            nn.init.zeros_(self.hurdle_head_layer.weight)
            nn.init.zeros_(self.hurdle_head_layer.bias)

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

        for sid, size_px in enumerate(self.region_sizes_px):
            mask = regions.scale_id == sid
            if not bool(mask.any()):
                continue

            if size_px <= 32:
                feat = p4
                dst_stride = 4
            elif size_px <= 64:
                feat = p8
                dst_stride = 8
            else:
                feat = p16
                dst_stride = 16

            boxes4 = regions.boxes[mask]

            if self.native_scale_pooling and feat.shape[-2:] != src_hw:
                scale = 4.0 / float(dst_stride)
                float_boxes = boxes4.float() * scale
                if self.regional_feature_stats == "mean_std":
                    pooled = fractional_region_mean_std_features(
                        feat,
                        float_boxes,
                    )
                else:
                    pooled = fractional_region_average_features(
                        feat,
                        float_boxes,
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

        result: dict[str, torch.Tensor] = {
            "mu_count": mu_count.unsqueeze(1),      # [B,1,M]
            "rate": rate.unsqueeze(1),              # [B,1,M]
            "dispersion": dispersion.unsqueeze(1),  # [B,1,M]
            "log_dispersion": log_r.unsqueeze(1),  # [B,1,M]
        }

        # Hurdle head: occupancy logit z_π_R (RMR-v7+)
        if self.has_hurdle_head:
            z_pi = self.hurdle_head_layer(h).squeeze(-1)  # [B,M]
            result["hurdle_logit"] = z_pi.unsqueeze(1)    # [B,1,M]

        return result


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

        if tuple(cfg.region_sizes_px) not in {(32, 64, 128), (16, 32, 64, 128)}:
            raise ValueError(
                f"RMR-v3 requires region_sizes_px in [(32, 64, 128), (16, 32, 64, 128)], got {cfg.region_sizes_px}"
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

        if cfg.neck_type == "rep_weighted":
            self.fusion = RepWeightedFPNNeck(
                in_channels=self.encoder.out_channels,
                width=cfg.feature_width,
                context_dilations=cfg.context_dilations,
            )
        elif cfg.neck_type == "aspp_lite":
            self.fusion = ASPPLiteFPNNeck(
                in_channels=self.encoder.out_channels,
                width=cfg.feature_width,
                aspp_dilations=cfg.aspp_dilations,
            )
        elif cfg.neck_type == "additive":
            self.fusion = AdditiveFPNNeck(
                in_channels=self.encoder.out_channels,
                width=cfg.feature_width,
            )
        else:
            raise ValueError(
                f"Unsupported neck_type: '{cfg.neck_type}'. Must be 'additive', 'aspp_lite', or 'rep_weighted'."
            )

        init_bias = _softplus_inverse(cfg.init_m0)

        self.fine_head = FineMeasureHead(
            width=cfg.feature_width,
            init_bias=init_bias,
            temp_softplus=cfg.temp_softplus,
        )

        self.region_head = ProbabilisticRegionalEvidenceHead(
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
        if getattr(self.fine_head, "temp_softplus", False):
            y0 = z0
        else:
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

        # ── Hurdle head: modulate solver target by occupancy probability ──────
        # b_solver_raw is the raw regional NB mean (before hurdle masking).
        b_solver_raw = mu_count.detach() if self.cfg.detach_region_mean_in_solver else mu_count

        if self.cfg.hurdle_head and "hurdle_logit" in regional:
            # π_R = sigmoid(z_π_R): probability region is occupied.
            # b_solver = π_R * mu_count  — background regions approach 0 count target.
            pi_r = torch.sigmoid(regional["hurdle_logit"].detach())
            b_solver = pi_r * b_solver_raw
        else:
            b_solver = b_solver_raw

        if self.cfg.detach_reliability_in_solver:
            weight_solver = weight_solver.detach()

        # Collect hurdle logit for loss computation (not detached)
        hurdle_logit = regional.get("hurdle_logit", None)  # [B,1,M] or None

        # Bypass solver loop if solver is disabled (direct baseline mode)
        if not self.cfg.enable_solver:
            out = {
                "y": y0,
                "y0": y0,
                "z0": z0,
                "regions": regions,
                "b_region": mu_count,
                "b_solver": b_solver,
                "region_rate": regional["rate"],
                "region_dispersion": dispersion,
                "region_log_dispersion": regional["log_dispersion"],
                "region_weight": weight,
                "solver_region_weight": weight_solver,
                "region_precision": reliability["precision"],
                "region_rate_variance": reliability["rate_variance"],
                "region_count_variance": reliability["count_variance"],
                "iterates": [y0],
                "residual_fields": [],
                "energy_trace": [],
                "uniform_reliability": uniform_reliability,
                "solver_strength": 0.0,
            }
            if hurdle_logit is not None:
                out["hurdle_logit"] = hurdle_logit
            return out

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

        # Optional TV Laplacian kernel (pre-allocated on device, cached as buffer-like)
        tv_lambda = float(self.cfg.tv_lambda)
        if tv_lambda > 0.0:
            _laplace = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
                device=x.device,
            ).view(1, 1, 3, 3)

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
                y.float() - effective_omega * field,
                0.0,
            )

            # TV Laplacian smoothing: suppresses high-frequency noise accumulated
            # by SIRT backprojection in dense crowd regions.
            # Forward heat equation: y_{t+1} = y_t + \lambda \Delta y
            if tv_lambda > 0.0:
                lap = F.conv2d(y_next, _laplace, padding=1)
                y_next = torch.clamp_min(y_next + tv_lambda * lap, 0.0)

            y_next = y_next.to(y.dtype)

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

        out = {
            "y": y,
            "y0": y0,
            "z0": z0,

            "regions": regions,

            "b_region": mu_count,
            "b_solver": b_solver,
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

        if hurdle_logit is not None:
            out["hurdle_logit"] = hurdle_logit
        return out

    def switch_to_deploy(self) -> None:
        """Switch internal modules (e.g. RepWeightedFPNNeck) to fused deployment mode."""
        if hasattr(self.fusion, "switch_to_deploy"):
            self.fusion.switch_to_deploy()


