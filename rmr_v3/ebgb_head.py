"""Unsaturable Regional Evidence Head (EB-GB: Error-Bounded Geometric Binning).

Solves the Negative-Binomial / Poisson NLL gradient starvation crisis:
    dL/dmu ~ (mu - y) / (mu + mu^2 / r)
which severely penalizes dense crowd samples by up to 4,080x less than sparse background.

Instead of saturable direct regression, EB-GB predicts a categorical probability
distribution P_R in Delta^{K-1} over a 16-point geometric bin lattice:
    b = [0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300, 450]^T
under an ordinal Cross-Entropy objective with strictly bounded gradient:
    ||nabla_z L_{CE}||_infty <= 1.0 (Zero gradient starvation).
"""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    fractional_region_average_features,
    fractional_region_mean_std_features,
    fractional_region_sum_features,
    fractional_region_sum_std_features,
    region_average_features,
    region_mean_std_features,
    region_sum_features,
    region_sum_std_features,
)

DEFAULT_GEOMETRIC_BINS: tuple[float, ...] = (
    0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0,
    27.0, 40.0, 60.0, 90.0, 135.0, 200.0, 300.0, 450.0,
    750.0, 1200.0,
)


def _canonicalize_region_size(s: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(s, (tuple, list)):
        return (int(s[0]), int(s[1]))
    val = int(s)
    return (val, val)


class UnsaturableRegionalEvidenceHead(nn.Module):
    """Regional Evidence Head using Error-Bounded Geometric Binning (EB-GB)."""

    def __init__(
        self,
        region_sizes_px: Sequence[int | Sequence[int]] = (32, 64, 128),
        feature_dim: int = 32,
        hidden_dim: int = 42,
        regional_feature_stats: str = "mean",
        native_scale_pooling: bool = True,
        bin_centers: Sequence[float] = DEFAULT_GEOMETRIC_BINS,
        dispersion_min: float = 1.0,
        dispersion_max: float = 500.0,
        hurdle_head: bool = True,
    ) -> None:
        super().__init__()
        self.region_sizes_px = tuple(region_sizes_px)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.regional_feature_stats = str(regional_feature_stats).lower().strip()
        self.native_scale_pooling = bool(native_scale_pooling)
        self.dispersion_min = float(dispersion_min)
        self.dispersion_max = float(dispersion_max)
        self.has_hurdle_head = bool(hurdle_head)

        bins_t = torch.tensor(bin_centers, dtype=torch.float32)
        self.register_buffer("bin_centers", bins_t, persistent=False)
        self.num_bins = len(bin_centers)

        in_dim = self.feature_dim + 1 if self.regional_feature_stats in ("mean", "sum") else 2 * self.feature_dim + 1

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )

        # Head 1: Geometric Bin Logits (K classes)
        self.bin_head = nn.Linear(self.hidden_dim, self.num_bins)
        nn.init.normal_(self.bin_head.weight, std=0.01)
        nn.init.zeros_(self.bin_head.bias)

        # Head 2: Hurdle Occupancy Logit (optional binary empty/non-empty gate)
        if self.has_hurdle_head:
            self.hurdle_head = nn.Linear(self.hidden_dim, 1)
            nn.init.zeros_(self.hurdle_head.weight)
            nn.init.constant_(self.hurdle_head.bias, 0.0)
        else:
            self.hurdle_head = None

    def _collect_region_features(
        self,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        regions: RegionSet,
    ) -> torch.Tensor:
        p4, p8, p16 = pyramid
        b = p4.shape[0]
        src_hw = p4.shape[-2:]
        device, dtype = p4.device, p4.dtype
        out_dim = self.feature_dim + 1 if self.regional_feature_stats in ("mean", "sum") else 2 * self.feature_dim + 1

        scale_specs = regions.scale_sizes_px if regions.scale_sizes_px is not None else [
            _canonicalize_region_size(s) for s in self.region_sizes_px
        ]
        num_boxes = int(regions.boxes.shape[0])
        if num_boxes == 0:
            return torch.zeros((b, 0, out_dim), device=device, dtype=dtype)

        out = torch.zeros((b, num_boxes, out_dim), device=device, dtype=dtype)
        for sid, size_spec in enumerate(scale_specs):
            mask = (regions.scale_id == sid)
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
                src_stride = 4.0
                scale = src_stride / float(dst_stride)
                float_boxes = boxes4.float() * scale
                grid_scale = (float(dst_stride) / src_stride) ** 2
                if self.regional_feature_stats == "sum":
                    pooled = fractional_region_sum_features(feat_curr, float_boxes) * grid_scale
                elif self.regional_feature_stats == "sum_std":
                    pooled = fractional_region_sum_std_features(feat_curr, float_boxes)
                    sums = pooled[..., :self.feature_dim] * grid_scale
                    pooled = torch.cat([sums, pooled[..., self.feature_dim:]], dim=-1)
                elif self.regional_feature_stats == "mean_std":
                    pooled = fractional_region_mean_std_features(feat_curr, float_boxes)
                else:
                    pooled = fractional_region_average_features(feat_curr, float_boxes)
            else:
                if feat_curr.shape[-2:] != src_hw:
                    feat_curr = F.interpolate(feat_curr, size=src_hw, mode="bilinear", align_corners=False)
                boxes_level = boxes4
                if self.regional_feature_stats == "sum":
                    pooled = region_sum_features(feat_curr, boxes_level)
                elif self.regional_feature_stats == "sum_std":
                    pooled = region_sum_std_features(feat_curr, boxes_level)
                elif self.regional_feature_stats == "mean_std":
                    pooled = region_mean_std_features(feat_curr, boxes_level)
                else:
                    pooled = region_average_features(feat_curr, boxes_level)

            area_k = regions.area[mask].to(device=device, dtype=dtype).view(1, -1, 1).clamp_min(1.0)
            eff_scale_px = 4.0 * torch.sqrt(area_k)
            log_scale = torch.log((eff_scale_px / 32.0).clamp_min(1.0 / 32.0)).expand(b, -1, 1)
            out[:, mask, :] = torch.cat([pooled, log_scale], dim=-1)

        mask_full = (regions.scale_id == -1)
        if mask_full.any():
            boxes_full = regions.boxes[mask_full]
            feat_curr = p16
            if feat_curr.shape[-2:] != src_hw:
                feat_curr = F.interpolate(feat_curr, size=src_hw, mode="bilinear", align_corners=False)
            if self.regional_feature_stats == "sum":
                pooled_full = region_sum_features(feat_curr, boxes_full)
            elif self.regional_feature_stats == "sum_std":
                pooled_full = region_sum_std_features(feat_curr, boxes_full)
            elif self.regional_feature_stats == "mean_std":
                pooled_full = region_mean_std_features(feat_curr, boxes_full)
            else:
                pooled_full = region_average_features(feat_curr, boxes_full)
            area_full = regions.area[mask_full].to(device=device, dtype=dtype).view(1, -1, 1).clamp_min(1.0)
            eff_scale_full_px = 4.0 * torch.sqrt(area_full)
            log_scale_full = torch.log((eff_scale_full_px / 32.0).clamp_min(1.0)).expand(b, -1, 1)
            out[:, mask_full, :] = torch.cat([pooled_full, log_scale_full], dim=-1)

        return out

    def forward(
        self,
        pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        regions: RegionSet,
    ) -> dict[str, torch.Tensor]:
        x = self._collect_region_features(pyramid, regions)  # [B, M, in_dim]
        h = self.trunk(x)                                    # [B, M, hidden]

        z_bins = self.bin_head(h)                            # [B, M, 16]
        p_bins = F.softmax(z_bins, dim=-1)                   # [B, M, 16]

        bins = self.bin_centers.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
        mu_bin = (p_bins * bins).sum(dim=-1)                 # [B, M]

        # Analytical variance within categorical distribution
        within_var = (p_bins * (bins - mu_bin.unsqueeze(-1)).square()).sum(dim=-1)

        hurdle_logit = None
        if self.hurdle_head is not None:
            z_pi = self.hurdle_head(h).squeeze(-1)           # [B, M]
            pi_r = torch.sigmoid(z_pi)                       # [B, M]
            # Principled soft occupancy gate:
            # - For mu_bin >= 1.0: gate is identically 1.0 (zero crowd attenuation, 100% mass preservation).
            # - For mu_bin < 1.0: suppresses background noise via pi_r.
            occ_gate = 1.0 - (1.0 - pi_r) * torch.clamp(1.0 - mu_bin, min=0.0, max=1.0)
            c_r = occ_gate * mu_bin                          # [B, M]
            count_var = occ_gate.square() * within_var + 0.01
            hurdle_logit = z_pi.unsqueeze(1)
        else:
            c_r = mu_bin
            count_var = within_var + 0.01

        area = regions.area.to(device=h.device, dtype=h.dtype).view(1, -1).clamp_min(1.0)
        rate = c_r / area                                    # [B, M]

        # Effective Negative-Binomial equivalent dispersion
        excess_var = torch.relu(count_var - c_r) + 1e-4
        dispersion = (c_r.square() / excess_var).clamp(self.dispersion_min, self.dispersion_max)
        log_r = torch.log(dispersion)

        res = {
            "mu_count": c_r.unsqueeze(1),
            "rate": rate.unsqueeze(1),
            "dispersion": dispersion.unsqueeze(1),
            "log_dispersion": log_r.unsqueeze(1),
            "bin_logits": z_bins,
            "bin_probs": p_bins,
            "bin_centers": self.bin_centers,
            "count_variance": count_var.unsqueeze(1),
        }
        if hurdle_logit is not None:
            res["hurdle_logit"] = hurdle_logit
        return res
