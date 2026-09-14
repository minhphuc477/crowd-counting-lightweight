from __future__ import annotations

"""Probabilistic Regional Evidence Head and Reliability Modeling for RMR.

Predicts regional Negative-Binomial count distributions, dispersion parameters,
and occupancy probabilities (Hurdle model), while deriving physics-based precision
and reliability weights for the inverse solver.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    _canonicalize_region_size,
    fractional_region_average_features,
    fractional_region_mean_std_features,
    region_average_features,
    region_mean_std_features,
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

        feat_list = []

        for sid, size_spec in enumerate(self.region_sizes_px):
            mask = regions.scale_id == sid
            if not bool(mask.any()):
                continue

            hy_px, wx_px = _canonicalize_region_size(size_spec)
            max_size_px = max(hy_px, wx_px)

            if max_size_px <= 32:
                feat = p4
                dst_stride = 4
            elif max_size_px <= 64:
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

            geom_scale_px = math.sqrt(float(hy_px) * float(wx_px))
            log_scale = torch.full_like(
                pooled[..., :1],
                fill_value=float(math.log(geom_scale_px / 32.0)),
            )

            feat_list.append(
                torch.cat(
                    [pooled, log_scale],
                    dim=-1,
                )
            )

        out = torch.cat(feat_list, dim=1) if feat_list else torch.zeros((p4.shape[0], m_total, out_dim), device=device, dtype=dtype)

        # Main method disables full-image regions.
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
    mode: str = "nb_rate_variance",
    rate_std_floor: float = 0.01,
    weight_min: float = 0.25,
    weight_max: float = 4.0,
    normalize_within_scale: bool = True,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Derive regional reliability from NB predictive rate variance or SNR.

    Modes:
    - 'nb_rate_variance': q = 1 / (Var[rate] + floor^2), classical inverse variance.
    - 'snr': q = mu^2 / Var[N] = r * mu / (r + mu), Signal-to-Noise Ratio weighting
             that prioritizes high-certainty crowd clumps over noisy empty background.
    """

    mu = mu_count.float().clamp_min(0.0)
    r = dispersion.float().clamp_min(eps)

    area = regions.area.float().view(1, 1, -1)
    area = area.clamp_min(1.0)

    count_var = mu + mu.square() / r

    floor_var = float(rate_std_floor) ** 2

    rate_var = count_var / area.square()
    rate_var = rate_var + floor_var

    if mode == "snr":
        # Signal-to-Noise Ratio: mu^2 / Var[N] = r * mu / (r + mu + eps)
        # Scales naturally with crowd presence and certainty, prioritizing genuine clusters
        precision = (r * mu) / (r + mu + float(eps))
    else:
        precision = 1.0 / rate_var.clamp_min(eps)

    if normalize_within_scale:
        weight = torch.zeros_like(precision)

        for sid in torch.unique(regions.scale_id):
            mask = (regions.scale_id == sid).to(dtype=precision.dtype).view(1, 1, -1)
            q_sum = (precision * mask).sum(dim=-1, keepdim=True)
            count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            q_mean = (q_sum / count).clamp_min(eps)
            weight = weight + (precision / q_mean) * mask
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
