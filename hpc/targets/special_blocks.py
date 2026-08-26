"""Special block mask builder (SR48 §10.5).

Builds fixed-shape ``gt_large_mask16`` and ``gt_true_border_mask16`` tensors
from per-point flags produced by the safe-crop transform.

These are **training metadata only** — not deployment outputs.
d_nn is a scale proxy, not a true head-size measurement.
"""

import torch
import numpy as np


def build_special_block_masks(
    crop_points: np.ndarray,
    point_large_flags: np.ndarray,
    point_true_border_flags: np.ndarray,
    crop_h: int,
    crop_w: int,
    block_size: int = 16,
) -> dict:
    """Build fixed block mask tensors for large and true-border annotated points.

    Uses the same ``floor(x / B), floor(y / B)`` indexing as exact count targets.

    Args:
        crop_points: (M, 2) float32 array of (x, y) point centres in crop space.
        point_large_flags: (M,) bool array — True for large/isolated proxy points.
        point_true_border_flags: (M,) bool array — True for true-image-border proxy points.
        crop_h, crop_w: spatial dimensions of the crop in input pixels.
        block_size: input-pixel block size (default 16).

    Returns:
        dict with keys:
            ``gt_large_mask16``:       FloatTensor (crop_h // block_size, crop_w // block_size)
            ``gt_true_border_mask16``: FloatTensor (crop_h // block_size, crop_w // block_size)
            ``gt_special_mask16``:     FloatTensor — elementwise max of the above two
    """
    if crop_h % block_size != 0 or crop_w % block_size != 0:
        raise ValueError(
            f"crop ({crop_h}, {crop_w}) must be divisible by block_size={block_size}"
        )
    h_blk = crop_h // block_size
    w_blk = crop_w // block_size

    large_mask = np.zeros((h_blk, w_blk), dtype=np.float32)
    border_mask = np.zeros((h_blk, w_blk), dtype=np.float32)

    pts = np.asarray(crop_points, dtype=np.float32).reshape(-1, 2)
    lf = np.asarray(point_large_flags, dtype=bool)
    bf = np.asarray(point_true_border_flags, dtype=bool)

    if len(pts) != len(lf) or len(pts) != len(bf):
        raise ValueError(
            f"crop_points ({len(pts)}), large_flags ({len(lf)}), "
            f"border_flags ({len(bf)}) must have equal length"
        )

    for i in range(len(pts)):
        x, y = pts[i, 0], pts[i, 1]
        # Clamp to valid block indices (same as block_counts.py floor indexing)
        bx = int(x / block_size)
        by = int(y / block_size)
        bx = max(0, min(bx, w_blk - 1))
        by = max(0, min(by, h_blk - 1))
        if lf[i]:
            large_mask[by, bx] = 1.0
        if bf[i]:
            border_mask[by, bx] = 1.0

    special_mask = np.maximum(large_mask, border_mask)

    return {
        "gt_large_mask16": torch.from_numpy(large_mask),
        "gt_true_border_mask16": torch.from_numpy(border_mask),
        "gt_special_mask16": torch.from_numpy(special_mask),
    }
