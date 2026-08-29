import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .transforms import ScaleAwareSafeGeometricTransforms, PhotometricTransforms
from ..targets.block_counts import build_hierarchical_block_counts
from ..targets.allocation_target import build_block_constrained_allocation_target
from ..targets.special_blocks import build_special_block_masks
from ..targets.routing_target import build_routing_target


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BaseCrowdDataset(Dataset):
    """Base dataset with a stable schema; use :func:`custom_collate_fn`.

    ``gt_points`` has variable length, so PyTorch's default collator is not a
    valid batching strategy for crowd samples.

    Train sample schema (all keys always present):
        image                (3, H, W)  — ImageNet-normalized clean crop
        image_degraded       (3, H, W)  — photometrically degraded view (or clone if not used)
        has_degraded         ()  bool   — True if degraded view was actually generated
        gt_blocks            dict[B → (H/B, W/B)] — exact integer block counts
        gt_z_alloc           (H/4, W/4) — block-constrained soft allocation target
        gt_count             ()  float  — total annotated count in crop
        gt_large_mask16      (H/16, W/16) float — proxy large/isolated point blocks
        gt_true_border_mask16 (H/16, W/16) float — proxy true-border point blocks
        gt_special_mask16    (H/16, W/16) float — union of large + border masks
        has_gt               ()  bool
        img_path             str

    Note: gt_large_mask16 / gt_true_border_mask16 use d_nn as a scale *proxy*,
    not true head-size labels. This is documented in the proposal §10.7.
    """

    def __init__(
        self,
        image_paths: List[str],
        points_list: List[np.ndarray],
        crop_size: int = 448,
        hnb_blocks: List[int] = (8, 16, 32, 64),
        allocation_block: int = 16,
        is_train: bool = True,
        scale_range: Tuple[float, float] = (0.75, 2.0),
        flip_prob: float = 0.5,
        second_view_prob: float = 0.30,
        photometric_cfg: Optional[Dict[str, Any]] = None,
        has_ground_truth: Optional[List[bool]] = None,
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
        # Safe-crop / scale-aware crop parameters (passed through to transform)
        max_crop_attempts: int = 20,
        crop_guard_nn_factor: float = 0.20,
        crop_guard_min_px: float = 8.0,
        crop_guard_max_px: float = 48.0,
        random_crop_prob: float = 0.75,
        large_center_crop_prob: float = 0.15,
        border_center_crop_prob: float = 0.10,
        large_nn_threshold_px: float = 48.0,
        true_border_threshold_px: float = 32.0,
        crop_sampling: str = "safe_mixture",
        ntpc_only: bool = False,
    ):
        if len(image_paths) != len(points_list):
            raise ValueError("image_paths and points_list must have the same length")
        if not 0.0 <= second_view_prob <= 1.0:
            raise ValueError("second_view_prob must be in [0, 1]")
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")

        self.image_mean = list(image_mean) if image_mean is not None else list(IMAGENET_MEAN)
        self.image_std = list(image_std) if image_std is not None else list(IMAGENET_STD)

        self.image_paths = list(image_paths)
        self.points_list = [np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in points_list]
        self.crop_size = int(crop_size)
        self.hnb_blocks = [int(b) for b in hnb_blocks]
        self.allocation_block = int(allocation_block)
        self.is_train = bool(is_train)
        self.second_view_prob = float(second_view_prob) if self.is_train else 0.0
        self.ntpc_only = bool(ntpc_only)
        self.has_ground_truth = (
            list(has_ground_truth)
            if has_ground_truth is not None
            else [True] * len(self.image_paths)
        )

        if len(self.has_ground_truth) != len(self.image_paths):
            raise ValueError("has_ground_truth must match dataset length")
        if self.is_train and not all(self.has_ground_truth):
            raise ValueError("Training dataset contains samples without ground truth")

        for b in self.hnb_blocks + [self.allocation_block]:
            if self.crop_size % b != 0:
                raise ValueError(
                    f"crop_size={self.crop_size} must be divisible by block size {b}"
                )
        if self.allocation_block % 4 != 0:
            raise ValueError("allocation_block must be divisible by output stride 4")

        self.geom_transform = ScaleAwareSafeGeometricTransforms(
            crop_size=self.crop_size,
            scale_range=scale_range,
            flip_prob=flip_prob,
            max_crop_attempts=max_crop_attempts,
            crop_guard_nn_factor=crop_guard_nn_factor,
            crop_guard_min_px=crop_guard_min_px,
            crop_guard_max_px=crop_guard_max_px,
            random_crop_prob=random_crop_prob,
            large_center_crop_prob=large_center_crop_prob,
            border_center_crop_prob=border_center_crop_prob,
            large_nn_threshold_px=large_nn_threshold_px,
            true_border_threshold_px=true_border_threshold_px,
            crop_sampling=crop_sampling,
            compute_point_metadata=not self.ntpc_only,
        )
        self.photo_transform = PhotometricTransforms(**(photometric_cfg or {}))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as im:
            image = im.convert("RGB")
        points = self.points_list[idx].copy()
        has_gt = bool(self.has_ground_truth[idx])

        if self.is_train:
            # --- Geometric transform (returns metadata dict) ---
            geom_out = self.geom_transform(image, points)
            image_crop = geom_out["image"]
            crop_pts = geom_out["points"]
            large_flags = geom_out["point_large_flags"]
            border_flags = geom_out["point_true_border_flags"]

            img_tensor = TF.to_tensor(image_crop)
            img_clean = TF.normalize(img_tensor, mean=self.image_mean, std=self.image_std)

            # --- Photometric second view ---
            use_degraded = random.random() < self.second_view_prob
            if use_degraded:
                deg_pil = self.photo_transform(image_crop.copy())
                deg_tensor = TF.to_tensor(deg_pil)
                img_degraded = TF.normalize(deg_tensor, mean=self.image_mean, std=self.image_std)
            else:
                # Placeholder; criterion uses has_degraded to mask it out.
                img_degraded = img_clean.clone()

            # --- Count and density targets ---
            gt_blocks = build_hierarchical_block_counts(
                crop_pts, self.crop_size, self.crop_size, self.hnb_blocks
            )
            gt_count = torch.tensor(float(len(crop_pts)), dtype=torch.float32)
            if self.ntpc_only:
                return {
                    "image": img_clean,
                    "gt_blocks": gt_blocks,
                    "gt_count": gt_count,
                    "gt_points": torch.from_numpy(crop_pts).float(),
                    "has_gt": torch.tensor(True, dtype=torch.bool),
                    "img_path": img_path,
                }
            gt_z_alloc = build_block_constrained_allocation_target(
                crop_pts,
                self.crop_size,
                self.crop_size,
                block_size=self.allocation_block,
                output_stride=4,
            )

            # --- Special block masks (training metadata only) ---
            special_masks = build_special_block_masks(
                crop_points=crop_pts,
                point_large_flags=large_flags,
                point_true_border_flags=border_flags,
                crop_h=self.crop_size,
                crop_w=self.crop_size,
                block_size=self.allocation_block,
            )

            # --- Soft scale routing targets (SSER geometry supervision) ---
            crop_dnn = geom_out["point_dnn"]
            routing_targets = build_routing_target(
                crop_points=crop_pts,
                d_nn=crop_dnn,
                crop_h=self.crop_size,
                crop_w=self.crop_size,
                route_stride=8,  # backbone /8 resolution for routing
            )

            return {
                "image": img_clean,
                "image_degraded": img_degraded,
                "has_degraded": torch.tensor(use_degraded, dtype=torch.bool),
                "gt_blocks": gt_blocks,
                "gt_z_alloc": gt_z_alloc,
                "gt_count": gt_count,
                "gt_large_mask16": special_masks["gt_large_mask16"],
                "gt_true_border_mask16": special_masks["gt_true_border_mask16"],
                "gt_special_mask16": special_masks["gt_special_mask16"],
                "gt_route_q": routing_targets["gt_route_q"],         # (4, H/8, W/8)
                "gt_route_mask": routing_targets["gt_route_mask"],   # (H/8, W/8) bool
                "gt_points": torch.from_numpy(crop_pts).float(),     # (M, 2) crop-space points
                "has_gt": torch.tensor(True, dtype=torch.bool),
                "img_path": img_path,
            }

        # --- Validation / test path ---
        img_tensor = TF.to_tensor(image)
        img_norm = TF.normalize(img_tensor, mean=self.image_mean, std=self.image_std)
        gt_count = torch.tensor(
            float(len(points)) if has_gt else float("nan"), dtype=torch.float32
        )
        return {
            "image": img_norm,
            "gt_count": gt_count,
            "gt_points": points,
            "has_gt": torch.tensor(has_gt, dtype=torch.bool),
            "img_path": img_path,
        }


def custom_collate_fn(batch: List[dict]) -> dict:
    """Collate function supporting the full HPC training schema."""
    images = torch.stack([s["image"] for s in batch])

    # Hierarchical block counts: dict[int -> Tensor]
    scales = list(batch[0]["gt_blocks"].keys())
    gt_blocks = {b: torch.stack([s["gt_blocks"][b] for s in batch]) for b in scales}

    gt_count = torch.stack([s["gt_count"] for s in batch])

    res = {
        "image": images,
        "gt_blocks": gt_blocks,
        "gt_count": gt_count,
        "img_path": [s["img_path"] for s in batch],
    }

    if "gt_z_alloc" in batch[0]:
        res["gt_z_alloc"] = torch.stack([s["gt_z_alloc"] for s in batch])

    for key in ("gt_large_mask16", "gt_true_border_mask16", "gt_special_mask16"):
        if key in batch[0]:
            res[key] = torch.stack([s[key] for s in batch])

    if "gt_route_q" in batch[0]:
        res["gt_route_q"] = torch.stack([s["gt_route_q"] for s in batch])
        res["gt_route_mask"] = torch.stack([s["gt_route_mask"] for s in batch])

    if "image_degraded" in batch[0]:
        res["image_degraded"] = torch.stack([s["image_degraded"] for s in batch])
        res["has_degraded"] = torch.stack([s["has_degraded"] for s in batch])

    if "gt_points" in batch[0]:
        res["gt_points"] = [s["gt_points"] for s in batch]

    return res
