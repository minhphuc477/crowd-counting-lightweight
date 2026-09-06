from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .point_counts import build_exact_count_pyramid
from .transforms import NTPCGeometricTransform, in_closed_pixel_support


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_point_annotations(
    points: Any,
    source: str,
    coordinate_base: int = 1,
    image_shape: Optional[Tuple[int, int]] = None,
    tol: float = 1e-3,
    bounds_policy: str = "error",
) -> np.ndarray:
    """Validate/convert point coordinates with an explicit source-bounds policy.

    ``allow`` exists for the original ShanghaiTech release: its official MAT files
    contain source outliers and widely used PET/DM-Count loaders preserve the raw
    annotations, then filter them only when forming a training crop.
    """
    if bounds_policy not in {"error", "allow", "clip"}:
        raise ValueError(
            f"Unsupported bounds_policy={bounds_policy!r}; must be 'error', 'allow', or 'clip'"
        )
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Invalid point array in {source}: shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite point coordinate in {source}")

    arr = arr.astype(np.float32, copy=True)
    if coordinate_base == 1:
        # Standard MATLAB 1-based coordinates [1, W] x [1, H] -> 0-based [0, W-1] x [0, H-1]
        arr -= 1.0
    elif coordinate_base != 0:
        raise ValueError(f"Unsupported coordinate_base={coordinate_base}; must be 0 or 1")

    if image_shape is not None:
        w, h = image_shape
        bad = (
            (arr[:, 0] < -tol)
            | (arr[:, 0] > float(w - 1) + tol)
            | (arr[:, 1] < -tol)
            | (arr[:, 1] > float(h - 1) + tol)
        )
        if bad.any() and bounds_policy == "error":
            bad_pts = arr[bad][:10].tolist()
            raise ValueError(
                f"Out-of-bounds annotation in {source} (image_size={image_shape}): "
                f"found {bad.sum()} points out of bounds, samples: {bad_pts}"
            )
        if bounds_policy == "clip":
            arr[:, 0] = np.clip(arr[:, 0], 0.0, float(w - 1.0))
            arr[:, 1] = np.clip(arr[:, 1], 0.0, float(h - 1.0))

    return arr


class BaseCrowdDataset(Dataset):
    """Base dataset for NTPC crowd counting.

    Exposes recursive exact integer count pyramids {4, 8, 16, 32, 64}.
    Use :func:`ntpc_collate_fn` for training batching.
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        points_list: Sequence[np.ndarray],
        crop_size: int = 256,
        is_train: bool = True,
        scale_range: Tuple[float, float] = (0.7, 1.3),
        flip_prob: float = 0.5,
        has_ground_truth: Optional[Sequence[bool]] = None,
        image_mean: Optional[Sequence[float]] = None,
        image_std: Optional[Sequence[float]] = None,
    ):
        if len(image_paths) != len(points_list):
            raise ValueError("image_paths and points_list must have the same length")
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if crop_size % 64 != 0:
            raise ValueError(
                f"NTPC requires crop_size divisible by 64 (for stride-4 -> stride-64 pyramid), "
                f"got crop_size={crop_size}"
            )

        self.image_paths = list(image_paths)
        self.points_list = [np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in points_list]
        self.crop_size = int(crop_size)
        self.is_train = bool(is_train)
        self.image_mean = list(image_mean) if image_mean is not None else list(IMAGENET_MEAN)
        self.image_std = list(image_std) if image_std is not None else list(IMAGENET_STD)

        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must each have exactly 3 elements")
        if not np.isfinite(self.image_mean).all() or not np.isfinite(self.image_std).all():
            raise ValueError("image_mean and image_std must contain finite values")
        if np.any(np.asarray(self.image_std) <= 0):
            raise ValueError("image_std elements must be strictly positive")

        self.has_ground_truth = (
            list(has_ground_truth)
            if has_ground_truth is not None
            else [True] * len(self.image_paths)
        )

        if len(self.has_ground_truth) != len(self.image_paths):
            raise ValueError("has_ground_truth must match dataset length")
        if self.is_train and not all(self.has_ground_truth):
            raise ValueError("Training dataset contains samples without ground truth")

        self.geom_transform = NTPCGeometricTransform(
            crop_size=self.crop_size,
            scale_range=scale_range,
            flip_prob=flip_prob if self.is_train else 0.0,
        )

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
            image_crop, crop_pts = self.geom_transform(image, points)

            # Assert geometry bounds in closed continuous pixel-center support [-0.5, crop_size - 0.5]
            if len(crop_pts) > 0:
                if not in_closed_pixel_support(crop_pts, self.crop_size, self.crop_size).all():
                    raise RuntimeError(
                        f"Geometric transform produced out-of-bounds points for crop_size={self.crop_size}"
                    )

            img_tensor = TF.to_tensor(image_crop)
            img_clean = TF.normalize(img_tensor, mean=self.image_mean, std=self.image_std)

            # Recursive exact integer count pyramid: Y4 -> Y8 -> Y16 -> Y32 -> Y64
            tree = build_exact_count_pyramid(
                [torch.from_numpy(crop_pts).float()],
                height=self.crop_size,
                width=self.crop_size,
                block_sizes=(4, 8, 16, 32, 64),
            )
            gt_count = tree["N"][0].float()
            gt_blocks = {b: tree[b][0].float() for b in (4, 8, 16, 32, 64)}

            # Invariant checks
            if int(gt_count.item()) != len(crop_pts):
                raise RuntimeError(
                    f"Point/tree count mismatch: {len(crop_pts)} points vs {gt_count.item()} tree count"
                )
            if gt_count.ndim != 0:
                raise RuntimeError(f"gt_count must be scalar, got shape {gt_count.shape}")

            for b in (4, 8, 16, 32, 64):
                if gt_blocks[b].ndim != 2:
                    raise RuntimeError(f"gt_blocks[{b}] must be 2D, got shape {gt_blocks[b].shape}")
                if not torch.equal(gt_blocks[b], gt_blocks[b].round()):
                    raise RuntimeError(f"gt_blocks[{b}] contains non-integer values")
                if (gt_blocks[b] < 0).any():
                    raise RuntimeError(f"gt_blocks[{b}] contains negative values")
                if not torch.allclose(gt_blocks[b].sum(), gt_count, atol=1e-4):
                    raise RuntimeError(
                        f"gt_blocks[{b}] count {gt_blocks[b].sum().item()} != gt_count {gt_count.item()}"
                    )

            return {
                "image": img_clean,
                "gt_blocks": gt_blocks,
                "gt_count": gt_count,
                "gt_points": torch.from_numpy(crop_pts).float(),
                "has_gt": torch.tensor(True, dtype=torch.bool),
                "img_path": img_path,
            }

        # Validation / test path (full image, uncropped)
        img_np = np.array(image, dtype=np.uint8, copy=True)
        image.close()
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float().div(255.0)
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


def ntpc_collate_fn(batch: List[dict]) -> dict:
    """Collate function supporting fixed-size cropped training batches."""
    images = torch.stack([s["image"] for s in batch])
    gt_count = torch.stack([s["gt_count"] for s in batch])

    res = {
        "image": images,
        "gt_count": gt_count,
        "img_path": [s["img_path"] for s in batch],
    }

    if "gt_blocks" in batch[0]:
        scales = list(batch[0]["gt_blocks"].keys())
        res["gt_blocks"] = {b: torch.stack([s["gt_blocks"][b] for s in batch]) for b in scales}

    if "gt_points" in batch[0]:
        res["gt_points"] = [s["gt_points"] for s in batch]

    if "has_gt" in batch[0]:
        res["has_gt"] = torch.stack([s["has_gt"] for s in batch])

    return res


custom_collate_fn = ntpc_collate_fn
