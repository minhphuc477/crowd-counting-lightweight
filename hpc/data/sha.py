from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import scipy.io as sio

from .common import BaseCrowdDataset, validate_point_annotations

_validate_points = validate_point_annotations


def load_sha_mat_points(
    mat_path: str,
    coordinate_base: int = 0,
    image_shape: Optional[Tuple[int, int]] = None,
    bounds_policy: str = "allow",
) -> np.ndarray:
    """Load original ShanghaiTech points using the official-code raw-coordinate convention."""
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
        return _validate_points(
            points,
            mat_path,
            coordinate_base=coordinate_base,
            image_shape=image_shape,
            bounds_policy=bounds_policy,
        )
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
        coordinate_base: int = 0,
        annotation_bounds_policy: str = "allow",
    ):
        if split in {"val_data", "val", "val_split", "train_split"}:
            candidates = [
                os.path.join(root, part, "train_data"),
                os.path.join(root, f"{part}_final", "train_data"),
                os.path.join(root, "train_data"),
            ]
        else:
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

            pts = load_sha_mat_points(
                mat_path,
                coordinate_base=coordinate_base,
                image_shape=img_shape,
                bounds_policy=annotation_bounds_policy,
            )
            image_paths.append(img_path)
            points_list.append(pts)

        if not image_paths:
            raise RuntimeError(f"No images found in {img_dir}")

        if split in {"val_data", "val", "val_split", "train_split"}:
            perm = np.random.RandomState(42).permutation(len(image_paths))
            n_val = max(1, int(len(image_paths) * 0.10))
            if split in {"val_data", "val", "val_split"}:
                sel_indices = sorted(perm[-n_val:])
            else:
                sel_indices = sorted(perm[:-n_val])
            image_paths = [image_paths[i] for i in sel_indices]
            points_list = [points_list[i] for i in sel_indices]

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
