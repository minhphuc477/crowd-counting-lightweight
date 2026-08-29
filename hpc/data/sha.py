from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import scipy.io as sio

from .common import BaseCrowdDataset


def _validate_points(
    points,
    source: str,
    coordinate_base: int = 1,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Validate and convert coordinates to 0-based pixel-center coordinates [0, W-1] x [0, H-1]."""
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
        # Guarantee points lie within image bounds [0, W-1] x [0, H-1]
        arr[:, 0] = np.clip(arr[:, 0], 0.0, float(w - 1.0))
        arr[:, 1] = np.clip(arr[:, 1], 0.0, float(h - 1.0))

    return arr


def load_sha_mat_points(
    mat_path: str,
    coordinate_base: int = 1,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Strict ShanghaiTech annotation loader with explicit coordinate base conversion."""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"ShanghaiTech annotation not found: {mat_path}")
    try:
        mat = sio.loadmat(mat_path)
        if "image_info" in mat:
            points = mat["image_info"][0, 0][0, 0][0]
        elif "annPoints" in mat:
            points = mat["annPoints"]
        elif "points" in mat:
            points = mat["points"]
        else:
            candidates = [
                v
                for k, v in mat.items()
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2
            ]
            if len(candidates) != 1:
                raise KeyError(f"Could not uniquely find point array in {mat_path}")
            points = candidates[0]
        return _validate_points(points, mat_path, coordinate_base=coordinate_base, image_shape=image_shape)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse ShanghaiTech annotation {mat_path}: {exc}") from exc


class ShanghaiTechDataset(BaseCrowdDataset):
    """ShanghaiTech Part A and Part B dataset loader for NTPC."""

    def __init__(
        self,
        root: str,
        part: str = "part_A",
        split: str = "train_data",
        crop_size: int = 256,
        is_train: bool = True,
        scale_range: Tuple[float, float] = (0.7, 1.3),
        flip_prob: float = 0.5,
        image_mean: Optional[Sequence[float]] = None,
        image_std: Optional[Sequence[float]] = None,
        coordinate_base: int = 1,
        # Accepted as keyword arguments for backwards-compatibility callers but ignored:
        **kwargs,
    ):
        candidates = [
            os.path.join(root, part, split),
            os.path.join(root, f"{part}_final", split),
            os.path.join(root, split),
        ]
        data_dir = next((p for p in candidates if os.path.isdir(p)), None)
        if data_dir is None:
            raise FileNotFoundError(f"ShanghaiTech split directory not found; tried: {candidates}")

        img_dir = os.path.join(data_dir, "images")
        gt_candidates = [
            os.path.join(data_dir, "ground-truth"),
            os.path.join(data_dir, "ground_truth"),
            os.path.join(data_dir, "ground_truths"),
        ]
        gt_dir = next((p for p in gt_candidates if os.path.isdir(p)), None)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"ShanghaiTech image directory not found: {img_dir}")
        if gt_dir is None:
            raise FileNotFoundError(f"ShanghaiTech GT directory not found; tried: {gt_candidates}")

        image_paths: List[str] = []
        points_list: List[np.ndarray] = []
        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, img_name)
            stem = os.path.splitext(img_name)[0]
            gt_paths = [os.path.join(gt_dir, f"GT_{stem}.mat"), os.path.join(gt_dir, f"{stem}.mat")]
            mat_path = next((p for p in gt_paths if os.path.exists(p)), None)
            if mat_path is None:
                raise FileNotFoundError(f"Missing annotation for {img_path}; tried {gt_paths}")

            with Image.open(img_path) as im:
                img_shape = im.size  # (W, H)

            pts = load_sha_mat_points(mat_path, coordinate_base=coordinate_base, image_shape=img_shape)
            image_paths.append(img_path)
            points_list.append(pts)

        if not image_paths:
            raise RuntimeError(f"No images found in {img_dir}")

        super().__init__(
            image_paths=image_paths,
            points_list=points_list,
            crop_size=crop_size,
            is_train=is_train,
            scale_range=scale_range,
            flip_prob=flip_prob,
            image_mean=image_mean,
            image_std=image_std,
        )
