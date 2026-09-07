"""Backward-compatibility shim re-exporting from rmr_core.data."""
from __future__ import annotations

import torch

from rmr_core.data import (
    CrowdManifestDataset,
    collate_eval,
    collate_train,
    compute_manifest_density,
    normalize_image,
    rasterize_points,
    train_transform,
)


def _pad_to_crop(
    image: torch.Tensor,
    points: torch.Tensor,
    crop_h: int,
    crop_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deprecated legacy padding helper retained for backward-compatibility tests."""
    _, h, w = image.shape
    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)
    if pad_h or pad_w:
        mean = image.new_tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        new_h = h + pad_h
        new_w = w + pad_w
        canvas = mean.expand(3, new_h, new_w).clone()
        canvas[:, :h, :w] = image
        image = canvas
    return image, points


__all__ = [
    "CrowdManifestDataset",
    "_pad_to_crop",
    "collate_eval",
    "collate_train",
    "compute_manifest_density",
    "normalize_image",
    "rasterize_points",
    "train_transform",
]
