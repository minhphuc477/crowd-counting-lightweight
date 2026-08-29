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
        raise ValueError(f"Invalid point array in {source}: shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite point coordinate in {source}")
    return arr.astype(np.float32, copy=False)


def load_sha_mat_points(mat_path: str) -> np.ndarray:
    """Strict ShanghaiTech annotation loader; parse failures are never converted to negatives."""
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
                v for k, v in mat.items()
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2
            ]
            if len(candidates) != 1:
                raise KeyError(f"Could not uniquely find point array in {mat_path}")
            points = candidates[0]
        return _validate_points(points, mat_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse ShanghaiTech annotation {mat_path}: {exc}") from exc


class ShanghaiTechDataset(BaseCrowdDataset):
    def __init__(
        self,
        root: str,
        part: str = "part_A",
        split: str = "train_data",
        crop_size: int = 448,
        hnb_blocks: List[int] = (8, 16, 32, 64),
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

        image_paths, points_list = [], []
        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(img_dir, img_name)
            stem = os.path.splitext(img_name)[0]
            gt_paths = [os.path.join(gt_dir, f"GT_{stem}.mat"), os.path.join(gt_dir, f"{stem}.mat")]
            mat_path = next((p for p in gt_paths if os.path.exists(p)), None)
            if mat_path is None:
                raise FileNotFoundError(f"Missing annotation for {img_path}; tried {gt_paths}")
            image_paths.append(img_path)
            points_list.append(load_sha_mat_points(mat_path))

        if not image_paths:
            raise RuntimeError(f"No images found in {img_dir}")

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
