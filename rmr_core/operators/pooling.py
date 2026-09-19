from __future__ import annotations

import torch

from .prefix_sums import fractional_box_sum, prefix2d, regional_sum


def region_average_features(features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Average pooled region features: [B,C,H,W] -> [B,M,C]."""
    sums = regional_sum(features, boxes, out_dtype=torch.float32)  # [B,C,M]
    area = ((boxes[:, 2] - boxes[:, 0]).abs() * (boxes[:, 3] - boxes[:, 1]).abs()).float()
    avg = sums / area.view(1, 1, -1).clamp_min(1.0)
    return avg.transpose(1, 2).contiguous().to(features.dtype)


def region_mean_std_features(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Extract both spatial mean and standard deviation over bounding boxes in FP32.

    Uses a shifted two-pass variance calculation (subtracting channel spatial mean)
    to eliminate floating-point catastrophic cancellation under unnormalized or shifted features.
    """
    f32 = feature.float()
    shift = f32.mean(dim=(-2, -1), keepdim=True)
    f32_centered = f32 - shift

    mean_centered = region_average_features(f32_centered, boxes)
    mean_sq_centered = region_average_features(f32_centered.square(), boxes)
    var = (mean_sq_centered - mean_centered.square()).clamp_min(0.0)
    std = torch.sqrt(var + eps)
    mean = mean_centered + shift.view(f32.shape[0], 1, f32.shape[1])
    return torch.cat([mean, std], dim=-1).to(feature.dtype)


def fractional_region_average_features(
    features: torch.Tensor,
    float_boxes: torch.Tensor,
) -> torch.Tensor:
    """Average pooled region features using exact continuous fractional overlap.

    features: [B, C, H, W]
    float_boxes: [M, 4] with (y1, x1, y2, x2) in continuous coordinates on the features grid.
    returns: [B, M, C] in features.dtype
    """
    if float_boxes.ndim != 2 or float_boxes.shape[-1] != 4:
        raise ValueError("float_boxes must have shape [M, 4]")

    h, w = features.shape[-2:]
    y1, x1, y2, x2 = float_boxes.float().unbind(dim=-1)
    y_min = torch.minimum(y1, y2).clamp(0.0, float(h))
    y_max = torch.maximum(y1, y2).clamp(0.0, float(h))
    x_min = torch.minimum(x1, x2).clamp(0.0, float(w))
    x_max = torch.maximum(x1, x2).clamp(0.0, float(w))

    clamped_boxes = torch.stack([y_min, x_min, y_max, x_max], dim=-1)
    pref = prefix2d(features, preserve_fp32=True)
    sums = fractional_box_sum(pref, clamped_boxes)  # [B, C, M] in fp32

    area = ((y_max - y_min) * (x_max - x_min)).clamp_min(1e-6)  # [M]
    avg = sums / area.view(1, 1, -1)
    return avg.transpose(1, 2).contiguous().to(features.dtype)


def fractional_region_mean_std_features(
    feature: torch.Tensor,
    float_boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Extract both spatial mean and standard deviation over continuous boxes in FP32.

    Uses a shifted two-pass variance calculation (subtracting channel spatial mean)
    to eliminate floating-point catastrophic cancellation under unnormalized or shifted features.
    """
    f32 = feature.float()
    shift = f32.mean(dim=(-2, -1), keepdim=True)
    f32_centered = f32 - shift

    mean_centered = fractional_region_average_features(f32_centered, float_boxes)
    mean_sq_centered = fractional_region_average_features(f32_centered.square(), float_boxes)
    var = (mean_sq_centered - mean_centered.square()).clamp_min(0.0)
    std = torch.sqrt(var + eps)
    mean = mean_centered + shift.view(f32.shape[0], 1, f32.shape[1])
    return torch.cat([mean, std], dim=-1).to(feature.dtype)
