from __future__ import annotations

"""Knowledge Distillation Module for RMR-v8 (Stage 3).

Transfers high-capacity crowd spatial representations from a heavy teacher
(e.g., DM-Count VGG19, 21.5M params) into our ultra-lightweight student (<105k params).

Design Principles:
1. Zero test-time parameter overhead: All distillation adapters and teacher weights
   are ephemeral and discarded after training.
2. Density-Distribution Divergence: Normalized spatial KL divergence transfers
   fine-grained head localization without suffering from absolute density scale mismatch.
3. Total Count Consistency: Smooth-L1 count matching between student and teacher predictions.
4. Ephemeral Feature Hints (optional): 1x1 conv adapter projecting teacher feature maps
   to student channel width (32) with cosine distance loss.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityMapKDLoss(nn.Module):
    """Normalized spatial KL divergence and count alignment for crowd distillation."""

    def __init__(
        self,
        lambda_spatial_kl: float = 1.0,
        lambda_count_kd: float = 0.5,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_spatial_kl = float(lambda_spatial_kl)
        self.lambda_count_kd = float(lambda_count_kd)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(
        self,
        y_student: torch.Tensor,
        y_teacher: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute distillation losses between student and teacher density maps.

        Args:
            y_student: [B, 1, H, W] student density prediction (float32).
            y_teacher: [B, 1, H, W] teacher density prediction (float32, detached).

        Returns:
            Dictionary with 'spatial_kl', 'count_kd', and 'total_kd'.
        """
        ys = y_student.float()
        yt = y_teacher.float().detach()

        # Resize teacher if spatial dimensions differ
        if ys.shape[-2:] != yt.shape[-2:]:
            yt = F.interpolate(yt, size=ys.shape[-2:], mode="bilinear", align_corners=False)
            # Re-normalize mass after interpolation
            yt = yt * (y_teacher.sum(dim=(-2, -1), keepdim=True) / yt.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps))

        # 1. Spatial distribution KL divergence
        # Softmax over spatial lattice with temperature
        flat_ys = (ys / self.temperature).flatten(start_dim=-2)  # [B, 1, H*W]
        flat_yt = (yt / self.temperature).flatten(start_dim=-2)

        log_p_student = F.log_softmax(flat_ys, dim=-1)
        p_teacher = F.softmax(flat_yt, dim=-1)

        kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (self.temperature ** 2)

        # 2. Total count alignment
        count_student = ys.sum(dim=(-2, -1))
        count_teacher = yt.sum(dim=(-2, -1))
        count_l1 = F.smooth_l1_loss(count_student, count_teacher, beta=1.0)

        total_kd = self.lambda_spatial_kl * kl + self.lambda_count_kd * count_l1

        return {
            "spatial_kl": kl,
            "count_kd": count_l1,
            "total_kd": total_kd,
        }


class FeatureHintKDLoss(nn.Module):
    """Ephemeral 1x1 projection adapter for feature hint distillation.

    Project teacher feature channels (e.g. 512) down to student width (32),
    measuring cosine similarity on normalized representations.
    """

    def __init__(self, in_channels_teacher: int, out_channels_student: int = 32) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels_teacher, out_channels_student, kernel_size=1, bias=False)

    def forward(self, feat_student: torch.Tensor, feat_teacher: torch.Tensor) -> torch.Tensor:
        fs = feat_student.float()
        ft = feat_teacher.float().detach()

        # Project teacher channels
        ft_proj = self.proj(ft)

        # Resize to match student spatial resolution
        if ft_proj.shape[-2:] != fs.shape[-2:]:
            ft_proj = F.interpolate(ft_proj, size=fs.shape[-2:], mode="bilinear", align_corners=False)

        # Cosine distance: 1 - cosine_similarity
        cos_sim = F.cosine_similarity(fs, ft_proj, dim=1)  # [B, H, W]
        return (1.0 - cos_sim).mean()
