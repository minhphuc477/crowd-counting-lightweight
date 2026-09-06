"""Backward-compatibility shim re-exporting from rmr_core.data."""
from __future__ import annotations

from rmr_core.data import (
    CrowdManifestDataset,
    _pad_to_crop,
    collate_eval,
    collate_train,
    compute_manifest_density,
    normalize_image,
    rasterize_points,
    train_transform,
)

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
