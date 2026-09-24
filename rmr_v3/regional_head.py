from __future__ import annotations

"""Probabilistic Regional Evidence Head and Reliability Modeling for RMR.

Predicts regional Negative-Binomial count distributions, dispersion parameters,
and occupancy probabilities (Hurdle model), while deriving physics-based precision
and reliability weights for the inverse solver.
"""

import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    _canonicalize_region_size,
    fractional_region_average_features,
    fractional_region_mean_std_features,
    partition_regions_by_scale,
    region_average_features,
    region_mean_std_features,
    regional_sum,
)


def _softplus_inverse(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


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
        floor_tau: float = 0.0,
    ) -> None:
        super().__init__()
        self.floor_tau = float(floor_tau)


        if dispersion_min <= 0:
            raise ValueError("dispersion_min must be > 0")
        if dispersion_max <= dispersion_min:
            raise ValueError("dispersion_max must be > dispersion_min")
        if not (dispersion_min <= dispersion_init <= dispersion_max):
            raise ValueError("dispersion_init must lie inside [min,max]")
        if regional_feature_stats not in ("mean", "mean_std"):
            raise ValueError(f"regional_feature_stats must be 'mean' or 'mean_std', got '{regional_feature_stats}'")

        self.region_sizes_px = tuple(
            _canonicalize_region_size(x) for x in region_sizes_px
        )
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
        # When floor_tau > 0, compensate bias so that the predicted rate AFTER floor subtraction
        effective_init_rate = float(init_rate) + 0.5 * self.floor_tau if self.floor_tau > 0.0 else float(init_rate)
        nn.init.normal_(self.mean_head.weight, std=0.01)
        nn.init.constant_(
            self.mean_head.bias,
            _softplus_inverse(effective_init_rate),
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

        scale_specs = regions.scale_sizes_px if regions.scale_sizes_px is not None else [
            _canonicalize_region_size(s) for s in self.region_sizes_px
        ]
        feature_parts: list[torch.Tensor] = []
        for sid, size_spec in enumerate(scale_specs):
            mask = regions.scale_id == sid
            boxes4 = regions.boxes[mask]
            if boxes4.shape[0] == 0:
                continue

            hy_px, wx_px = _canonicalize_region_size(size_spec)
            max_size_px = max(hy_px, wx_px)
            if max_size_px <= 32:
                feat_curr, dst_stride = p4, 4
            elif max_size_px <= 64:
                feat_curr, dst_stride = p8, 8
            else:
                feat_curr, dst_stride = p16, 16

            if self.native_scale_pooling and feat_curr.shape[-2:] != src_hw:
                scale = 4.0 / float(dst_stride)
                float_boxes = boxes4.float() * scale
                if self.regional_feature_stats == "mean_std":
                    pooled = fractional_region_mean_std_features(
                        feat_curr,
                        float_boxes,
                    )
                else:
                    pooled = fractional_region_average_features(
                        feat_curr,
                        float_boxes,
                    )
            else:
                if feat_curr.shape[-2:] != src_hw:
                    feat_curr = F.interpolate(
                        feat_curr,
                        size=src_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                boxes_level = boxes4

                if self.regional_feature_stats == "mean_std":
                    pooled = region_mean_std_features(
                        feat_curr,
                        boxes_level,
                    )
                else:
                    pooled = region_average_features(
                        feat_curr,
                        boxes_level,
                    )

            geom_scale_px = math.sqrt(float(hy_px) * float(wx_px))
            log_scale = torch.full_like(
                pooled[..., :1],
                fill_value=float(math.log(geom_scale_px / 32.0)),
            )

            feature_parts.append(torch.cat([pooled, log_scale], dim=-1))

        # Collect full-image regions (scale_id == -1) if present
        mask_full = (regions.scale_id == -1)
        if mask_full.any():
            boxes_full = regions.boxes[mask_full]
            feat_curr = p16
            if feat_curr.shape[-2:] != src_hw:
                feat_curr = F.interpolate(
                    feat_curr,
                    size=src_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            if self.regional_feature_stats == "mean_std":
                pooled_full = region_mean_std_features(feat_curr, boxes_full)
            else:
                pooled_full = region_average_features(feat_curr, boxes_full)
            geom_scale_px = math.sqrt(float(src_hw[0] * 4.0) * float(src_hw[1] * 4.0))
            log_scale_full = torch.full_like(
                pooled_full[..., :1],
                fill_value=float(math.log(max(geom_scale_px, 32.0) / 32.0)),
            )
            feature_parts.append(torch.cat([pooled_full, log_scale_full], dim=-1))

        if not feature_parts:
            return torch.zeros((b, 0, out_dim), device=device, dtype=dtype)

        return torch.cat(feature_parts, dim=1)

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
        if getattr(self, "floor_tau", 0.0) > 0.0:
            tau = float(self.floor_tau)
            rate = torch.where(rate > tau, rate - 0.5 * tau, rate.square() / (2.0 * tau))

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
    mode: str = "nb_rate_variance",
    hurdle_pi: torch.Tensor | None = None,
    rate_std_floor: float = 0.01,
    weight_min: float = 0.25,
    weight_max: float = 4.0,
    normalize_within_scale: bool = True,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Derive regional reliability from NB predictive rate variance, SNR, or Hurdle-hybrid.

    Modes:
    - 'nb_rate_variance': q = 1 / (Var[rate] + floor^2), classical inverse variance.
    - 'snr': q = mu^2 / Var[N] = r * mu / (r + mu), Signal-to-Noise Ratio weighting
             that prioritizes high-certainty crowd clumps over noisy empty background.
    - 'hybrid_hurdle': Blends rate variance on background (pi_R -> 0 => w -> 4.0)
                       with SNR on crowd clusters (pi_R -> 1 => w -> 4.0).
    """

    mu = mu_count.float().clamp_min(0.0)
    r = dispersion.float().clamp_min(eps)

    area = regions.area.float().view(1, 1, -1)
    area = area.clamp_min(1.0)

    count_var = mu + mu.square() / r

    floor_var = float(rate_std_floor) ** 2

    rate_var = count_var / area.square()
    rate_var = rate_var + floor_var

    def _normalize_precision(p: torch.Tensor) -> torch.Tensor:
        if normalize_within_scale:
            w = torch.zeros_like(p)
            num_scales = int(getattr(regions, "num_scales", 3))
            # Include scale_id=-1 (full image) if present, plus standard scales [0, num_scales-1]
            for sid in range(-1, num_scales):
                mask = (regions.scale_id == sid).to(dtype=p.dtype).view(1, 1, -1)
                q_sum = (p * mask).sum(dim=-1, keepdim=True)
                count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                q_mean = (q_sum / count).clamp_min(eps)
                w = w + (p / q_mean) * mask
        else:
            w = p / p.mean(dim=-1, keepdim=True).clamp_min(eps)
        return w.clamp(min=float(weight_min), max=float(weight_max))

    if mode == "hybrid_hurdle":
        prec_var = 1.0 / rate_var.clamp_min(eps)
        prec_snr = (r * mu) / (r + mu + float(eps))
        w_var = _normalize_precision(prec_var)
        w_snr = _normalize_precision(prec_snr)

        if hurdle_pi is not None:
            pi = hurdle_pi.float().clamp(0.0, 1.0)
            if pi.ndim == 2:
                pi = pi.unsqueeze(1)
        else:
            pi = torch.ones_like(prec_snr)

        weight = (1.0 - pi) * w_var + pi * w_snr
        precision = (1.0 - pi) * prec_var + pi * prec_snr
    elif mode == "snr":
        # Signal-to-Noise Ratio: mu^2 / Var[N] = r * mu / (r + mu + eps)
        # Scales naturally with crowd presence and certainty, prioritizing genuine clusters
        precision = (r * mu) / (r + mu + float(eps))
        weight = _normalize_precision(precision)
    else:
        precision = 1.0 / rate_var.clamp_min(eps)
        weight = _normalize_precision(precision)

    return {
        "weight": weight,
        "precision": precision,
        "rate_variance": rate_var,
        "count_variance": count_var,
    }


def apply_scale_consistency_gating(
    weight: torch.Tensor,
    regions: RegionSet,
    scale_weights: torch.Tensor,
    *,
    power: float = 1.0,
    eps: float = 1e-6,
    perspective_horizon_gate: bool = False,
    horizon_cutoff: float = 0.35,
    region_sizes_px: Sequence[Sequence[int] | int] | None = None,
    grid_h: int | None = None,
) -> torch.Tensor:
    """Pre-Solver Scale-Consistency Reliability Gating (RMR-v15/v18).

    Modulates regional reliability weight w_R by the average scale probability of region R:
        pi_bar_k(R) = (1 / |R|) * sum_{u in R} pi_k(u)
        w_R <- w_R * (pi_bar_k(R) + eps)^power

    When perspective_horizon_gate=True (RMR-v18):
        For any scale k representing an anisotropic vertical box (height > width),
        boxes near the horizon (y_center / H < horizon_cutoff) are smoothly suppressed:
        w_R <- w_R * sigmoid((y_center/H - horizon_cutoff) / 0.05)
        This physically guarantees zero false positive mass bleeding from vertical boxes at the horizon.
    """
    orig_ndim = weight.ndim
    if orig_ndim == 2:
        weight = weight.unsqueeze(1)

    b, k_scales = scale_weights.shape[:2]
    w_out = weight.clone()
    area = regions.area.to(device=weight.device)

    scale_partitions = partition_regions_by_scale(regions, k_scales, device=weight.device)
    for k, mask_k, boxes_k in scale_partitions:
        if mask_k is None or boxes_k is None:
            continue
        pi_k = scale_weights[:, k:k+1, :, :].float()
        sum_pi_k = regional_sum(pi_k, boxes_k)  # [B, 1, M_k]
        area_k = area[mask_k].float().view(1, 1, -1)
        mean_pi_k = (sum_pi_k / area_k.clamp_min(1.0)).clamp(0.0, 1.0)

        gate = (mean_pi_k + float(eps)).pow(float(power))

        if perspective_horizon_gate:
            is_vert = (boxes_k[:, 2] - boxes_k[:, 0]) > (boxes_k[:, 3] - boxes_k[:, 1])
            if is_vert.any() and grid_h is not None and grid_h > 0:
                y_center = 0.5 * (boxes_k[:, 0].float() + boxes_k[:, 2].float()) / float(grid_h)
                geo_gate = torch.sigmoid((y_center - float(horizon_cutoff)) / 0.05).view(1, 1, -1)
                gate = torch.where(is_vert.view(1, 1, -1), gate * geo_gate.to(device=gate.device, dtype=gate.dtype), gate)

        w_out[:, :, mask_k] = w_out[:, :, mask_k] * gate

    if orig_ndim == 2:
        return w_out.squeeze(1)
    return w_out

