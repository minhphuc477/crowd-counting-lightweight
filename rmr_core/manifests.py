"""Dataset manifest resolution, validation, and empirical prior density computation.

Provides:
1. resolve_manifest_path: Resolves manifest file across local directories and data roots.
2. compute_manifest_density: Computes empirical mean cell density m0 for calibrated prior initialization.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from PIL import Image
import torch


def resolve_manifest_path(
    manifest_val: str | Path, data_root: str | Path | None = None
) -> Path | None:
    """Resolve manifest file on local filesystem.

    Tries in order:
    1. Direct path (manifest_val)
    2. Relative to data_root (if provided): data_root / manifest_val
    3. data_root / manifest_val.name
    4. Fallback to repo 'data/' directory (handles cross-environment simulation)
    """
    p = Path(manifest_val)
    if p.is_file():
        return p
    if data_root is not None:
        p_data = Path(data_root) / p
        if p_data.is_file():
            return p_data
        p_name = Path(data_root) / p.name
        if p_name.is_file():
            return p_name
    p_repo_data = Path("data") / p.name
    if p_repo_data.is_file():
        return p_repo_data
    return None


def compute_manifest_density(
    manifest: str | Path | None,
    output_stride: int = 4,
    default_m0: float = 0.015763,
    data_root: str | Path | None = None,
) -> float:
    """Empirically compute the mean cell density m0 across training images in a manifest.

    m0 = total_valid_points / total_stride4_cells.
    Points outside [0, w) x [0, h) are filtered identically to rasterize_points.
    """
    manifest_path_obj = Path(manifest) if manifest is not None else None
    if manifest_path_obj is not None and manifest_path_obj.name in (
        "sha_a_train.jsonl",
        "sha_a_val.jsonl",
    ):
        raise ValueError(
            f"Ad-hoc split manifest '{manifest}' has been deleted and is strictly forbidden "
            f"under the Zero Ad-hoc Split Policy! Use 'data/sha_a_train_all.jsonl' (300 samples)."
        )

    if manifest is None:
        return default_m0

    resolved_manifest = resolve_manifest_path(manifest, data_root=data_root)
    manifest_path = resolved_manifest if resolved_manifest is not None else Path(manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest '{manifest_path}' does not exist. Cannot compute manifest density."
        )

    root = Path(data_root) if data_root is not None else manifest_path.parent
    total_pts = 0
    total_cells = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pts_list = item.get("points", [])
            img_path = Path(item["image"])
            if not img_path.is_absolute():
                img_path = root / img_path
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image '{img_path}' referenced at line {line_num} in manifest '{manifest_path}' does not exist."
                )
            with Image.open(img_path) as img:
                w, h = img.size
            cells = math.ceil(h / output_stride) * math.ceil(w / output_stride)
            if pts_list:
                pts_arr = torch.tensor(pts_list, dtype=torch.float32).reshape(-1, 2)
                valid = (
                    (pts_arr[:, 0] >= 0)
                    & (pts_arr[:, 0] < w)
                    & (pts_arr[:, 1] >= 0)
                    & (pts_arr[:, 1] < h)
                )
                n_pts = int(valid.sum().item())
            else:
                n_pts = 0
            total_pts += n_pts
            total_cells += cells

    if total_cells <= 0:
        raise ValueError(f"Manifest '{manifest_path}' contains no valid images or cells.")

    return float(total_pts / total_cells)
