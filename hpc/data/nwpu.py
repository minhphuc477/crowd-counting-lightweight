from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import scipy.io as sio

from .common import BaseCrowdDataset, validate_point_annotations


def resolve_nwpu_split_file(ds_cfg: dict, eval_split: str) -> Optional[str]:
    """Resolve split file key for NWPU without cross-split fallbacks."""
    split_file_keys = {
        "train": "train_split_file",
        "val": "val_split_file",
        "test": "test_split_file",
    }
    if eval_split not in split_file_keys:
        raise ValueError(
            f"Unsupported NWPU split '{eval_split}'; must be one of {list(split_file_keys.keys())}"
        )
    return ds_cfg.get(split_file_keys[eval_split])


def _validate_points(
    points,
    source: str,
    coordinate_base: int = 0,
    image_shape: Optional[Tuple[int, int]] = None,
    tol: float = 1e-3,
) -> np.ndarray:
    arr = np.asarray(points)
    if arr.dtype == object and arr.shape == ():
        arr = arr.item()
    if isinstance(arr, dict):
        arr = arr.get("points", arr.get("annPoints", None))
        if arr is None:
            raise KeyError(f"No points/annPoints in {source}")
    return validate_point_annotations(
        arr, source=source, coordinate_base=coordinate_base, image_shape=image_shape, tol=tol
    )


def load_nwpu_points(
    ann_path: str,
    coordinate_base: int = 0,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Strict NWPU loader."""
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"NWPU annotation not found: {ann_path}")
    ext = os.path.splitext(ann_path)[1].lower()
    try:
        if ext == ".mat":
            mat = sio.loadmat(ann_path)
            if "annPoints" in mat:
                pts = mat["annPoints"]
            elif "points" in mat:
                pts = mat["points"]
            else:
                candidates = [
                    v
                    for k, v in mat.items()
                    if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2
                ]
                if len(candidates) != 1:
                    raise KeyError("Could not uniquely locate point array")
                pts = candidates[0]
        elif ext == ".npy":
            data = np.load(ann_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
                data = data.item()
            pts = data
        elif ext == ".json":
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pts = data.get("points", data.get("annPoints", data)) if isinstance(data, dict) else data
        elif ext == ".txt":
            if os.path.getsize(ann_path) == 0:
                pts = np.empty((0, 2), dtype=np.float32)
            else:
                pts = np.loadtxt(ann_path, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported NWPU annotation extension: {ext}")
        return _validate_points(pts, ann_path, coordinate_base=coordinate_base, image_shape=image_shape)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse NWPU annotation {ann_path}: {exc}") from exc


def _find_annotation(stem: str, gt_dirs: List[str]) -> Optional[str]:
    for d in gt_dirs:
        for ext in (".mat", ".npy", ".json", ".txt"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    return None


class NWPUDataset(BaseCrowdDataset):
    """NWPU-Crowd dataset loader for NTPC."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 256,
        is_train: bool = True,
        scale_range: Tuple[float, float] = (0.7, 1.3),
        flip_prob: float = 0.5,
        split_file: Optional[str] = None,
        image_mean: Optional[Sequence[float]] = None,
        image_std: Optional[Sequence[float]] = None,
        coordinate_base: int = 0,
    ):
        img_dir = os.path.join(root, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"NWPU image directory not found: {img_dir}")

        gt_dirs = [
            p
            for p in [
                os.path.join(root, "mats"),
                os.path.join(root, "ground_truth"),
                os.path.join(root, "jsons"),
                root,
            ]
            if os.path.isdir(p)
        ]

        if split_file is None:
            candidate = os.path.join(root, f"{split}.txt")
            if os.path.exists(candidate):
                split_file = candidate

        allow_missing_gt = split.lower() == "test" and not is_train
        image_paths: List[str] = []
        points_list: List[np.ndarray] = []
        has_gt_list: List[bool] = []

        if split_file is not None:
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"NWPU split file not found: {split_file}")
            with open(split_file, "r", encoding="utf-8") as f:
                ids = [line.split()[0] for line in f if line.strip()]
            image_names = [x if x.lower().endswith((".jpg", ".png", ".jpeg")) else f"{x}.jpg" for x in ids]
        else:
            if split.lower() not in {"all", "test"}:
                raise FileNotFoundError(
                    f"No split file for split='{split}'. Provide split_file explicitly to avoid split leakage."
                )
            image_names = sorted(
                n for n in os.listdir(img_dir) if n.lower().endswith((".jpg", ".png", ".jpeg"))
            )

        for img_name in image_names:
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"NWPU image listed but missing: {img_path}")
            stem = os.path.splitext(img_name)[0]
            ann_path = _find_annotation(stem, gt_dirs)

            with Image.open(img_path) as im:
                img_shape = im.size

            if ann_path is None:
                if not allow_missing_gt:
                    raise FileNotFoundError(f"Missing NWPU annotation for {img_path}")
                pts = np.empty((0, 2), dtype=np.float32)
                has_gt = False
            else:
                pts = load_nwpu_points(ann_path, coordinate_base=coordinate_base, image_shape=img_shape)
                has_gt = True
            image_paths.append(img_path)
            points_list.append(pts)
            has_gt_list.append(has_gt)

        if not image_paths:
            raise RuntimeError("NWPU dataset resolved to zero images")

        super().__init__(
            image_paths=image_paths,
            points_list=points_list,
            crop_size=crop_size,
            is_train=is_train,
            scale_range=scale_range,
            flip_prob=flip_prob,
            has_ground_truth=has_gt_list,
            image_mean=image_mean,
            image_std=image_std,
        )
