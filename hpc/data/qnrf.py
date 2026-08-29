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
    tol: float = 1e-3,
) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Invalid point array in {source}: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite point coordinate in {source}")

    arr = arr.astype(np.float32, copy=True)
    if coordinate_base == 1:
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
        if bad.any():
            bad_pts = arr[bad][:10].tolist()
            raise ValueError(
                f"Out-of-bounds annotation in {source} (image_size={image_shape}): "
                f"found {bad.sum()} points out of bounds, samples: {bad_pts}"
            )
        arr[:, 0] = np.clip(arr[:, 0], 0.0, float(w - 1.0))
        arr[:, 1] = np.clip(arr[:, 1], 0.0, float(h - 1.0))

    return arr


def load_qnrf_mat_points(
    mat_path: str,
    coordinate_base: int = 1,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"UCF-QNRF annotation not found: {mat_path}")
    try:
        mat = sio.loadmat(mat_path)
        if "annPoints" in mat:
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
        raise RuntimeError(f"Failed to parse UCF-QNRF annotation {mat_path}: {exc}") from exc


class UCFQNRFDataset(BaseCrowdDataset):
    """UCF-QNRF dataset loader for NTPC."""

    def __init__(
        self,
        root: str,
        split: str = "Train",
        crop_size: int = 256,
        is_train: bool = True,
        scale_range: Tuple[float, float] = (0.7, 1.3),
        flip_prob: float = 0.5,
        image_mean: Optional[Sequence[float]] = None,
        image_std: Optional[Sequence[float]] = None,
        coordinate_base: int = 1,
        **kwargs,
    ):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"UCF-QNRF split directory not found: {split_dir}")

        image_paths: List[str] = []
        points_list: List[np.ndarray] = []
        for img_name in sorted(os.listdir(split_dir)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")) or "_ann" in img_name:
                continue
            img_path = os.path.join(split_dir, img_name)
            stem = os.path.splitext(img_name)[0]
            mat_path = os.path.join(split_dir, f"{stem}_ann.mat")
            if not os.path.exists(mat_path):
                raise FileNotFoundError(f"Missing UCF-QNRF annotation for {img_path}: {mat_path}")

            with Image.open(img_path) as im:
                img_shape = im.size

            image_paths.append(img_path)
            points_list.append(load_qnrf_mat_points(mat_path, coordinate_base=coordinate_base, image_shape=img_shape))

        if not image_paths:
            raise RuntimeError(f"No UCF-QNRF images found in {split_dir}")

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
