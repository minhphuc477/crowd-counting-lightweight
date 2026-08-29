import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio

from .common import BaseCrowdDataset


def _validate_points(points, source: str) -> np.ndarray:
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
    return arr.astype(np.float32, copy=False)


def load_qnrf_mat_points(mat_path: str) -> np.ndarray:
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
                v for k, v in mat.items()
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2
            ]
            if len(candidates) != 1:
                raise KeyError(f"Could not uniquely find point array in {mat_path}")
            points = candidates[0]
        return _validate_points(points, mat_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse UCF-QNRF annotation {mat_path}: {exc}") from exc


class UCFQNRFDataset(BaseCrowdDataset):
    def __init__(
        self,
        root: str,
        split: str = "Train",
        crop_size: int = 672,
        hnb_blocks: List[int] = (16, 32, 96),
        allocation_block: int = 16,
        is_train: bool = True,
        scale_range: Tuple[float, float] = (0.75, 2.0),
        flip_prob: float = 0.5,
        second_view_prob: float = 0.30,
        photometric_cfg: Optional[Dict[str, Any]] = None,
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
        crop_sampling: str = "safe_mixture",
        ntpc_only: bool = False,
    ):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"UCF-QNRF split directory not found: {split_dir}")

        image_paths, points_list = [], []
        for img_name in sorted(os.listdir(split_dir)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")) or "_ann" in img_name:
                continue
            img_path = os.path.join(split_dir, img_name)
            stem = os.path.splitext(img_name)[0]
            mat_path = os.path.join(split_dir, f"{stem}_ann.mat")
            if not os.path.exists(mat_path):
                raise FileNotFoundError(f"Missing UCF-QNRF annotation for {img_path}: {mat_path}")
            image_paths.append(img_path)
            points_list.append(load_qnrf_mat_points(mat_path))

        if not image_paths:
            raise RuntimeError(f"No UCF-QNRF images found in {split_dir}")

        super().__init__(
            image_paths=image_paths,
            points_list=points_list,
            crop_size=crop_size,
            hnb_blocks=hnb_blocks,
            allocation_block=allocation_block,
            is_train=is_train,
            scale_range=scale_range,
            flip_prob=flip_prob,
            second_view_prob=second_view_prob,
            photometric_cfg=photometric_cfg,
            image_mean=image_mean,
            image_std=image_std,
            crop_sampling=crop_sampling,
            ntpc_only=ntpc_only,
        )
