from __future__ import annotations

import torch
import torch.nn.functional as F

from rmr_core.operators import RegionSet


def scale_balanced_region_rate_loss(
    pred_region: torch.Tensor,
    target_region: torch.Tensor,
    regions: RegionSet,
    beta: float = 0.1,
) -> torch.Tensor:
    """Scale-balanced loss on per-cell RATE, not raw count."""
    if pred_region.shape != target_region.shape:
        raise ValueError(f"shape mismatch: {pred_region.shape} vs {target_region.shape}")
    area = regions.area.to(dtype=pred_region.dtype).view(1, 1, -1).clamp_min(1.0)
    pred_rate = pred_region / area
    target_rate = target_region / area
    losses = []
    for sid in torch.unique(regions.scale_id):
        mask = regions.scale_id == sid
        if mask.any():
            losses.append(
                F.smooth_l1_loss(
                    pred_rate[..., mask],
                    target_rate[..., mask],
                    reduction="mean",
                    beta=beta,
                )
            )
    return torch.stack(losses).mean()


__all__ = ["scale_balanced_region_rate_loss"]
