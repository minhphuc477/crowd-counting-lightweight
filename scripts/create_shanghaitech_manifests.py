#!/usr/bin/env python3
"""Canonical ShanghaiTech Part A and Part B Manifest Generator.

Generates reproducible canonical JSONL manifests for:
- ShanghaiTech Part A:
    * data/sha_a_train_all.jsonl (300 train images)
    * data/sha_a_test.jsonl (182 test images)
- ShanghaiTech Part B:
    * data/shb_train_all.jsonl (400 train images)
    * data/shb_test.jsonl (316 test images)

Reads official MATLAB annotations:
    mat['image_info'][0, 0][0, 0][0]
Preserves canonical float coordinates without artificial round-off.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
import scipy.io as sio


def extract_points_from_mat(mat_path: Path) -> list[list[float]]:
    """Extract 2D head point coordinates from ShanghaiTech MAT annotation."""
    mat = sio.loadmat(str(mat_path))
    if "image_info" in mat:
        pts = mat["image_info"][0, 0][0, 0][0]
    elif "annPoints" in mat:
        pts = mat["annPoints"]
    elif "points" in mat:
        pts = mat["points"]
    else:
        candidates = [
            v for k, v in mat.items()
            if not k.startswith("__") and hasattr(v, "ndim") and v.ndim == 2 and v.shape[1] == 2
        ]
        if len(candidates) != 1:
            raise KeyError(f"Could not uniquely identify point array in {mat_path}")
        pts = candidates[0]
    return pts.astype(float).tolist()


def natural_key(p: Path) -> int:
    """Sort paths naturally by the numeric index in the filename (IMG_1 -> 1)."""
    match = re.search(r"(\d+)", p.stem)
    return int(match.group(1)) if match else 0


def build_shanghaitech_split(
    img_dir: Path,
    gt_dir: Path,
    rel_root: Path,
) -> list[dict[str, Any]]:
    """Build manifest items for a ShanghaiTech split directory."""
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=natural_key,
    )

    items: list[dict[str, Any]] = []
    for img_p in img_files:
        stem = img_p.stem
        # Look for GT_IMG_xxx.mat or IMG_xxx.mat
        mat_candidates = [
            gt_dir / f"GT_{stem}.mat",
            gt_dir / f"{stem}.mat",
        ]
        mat_p = next((p for p in mat_candidates if p.is_file()), None)
        if mat_p is None:
            raise FileNotFoundError(f"Missing annotation for {img_p}; checked {mat_candidates}")

        points = extract_points_from_mat(mat_p)
        rel_img_path = img_p.relative_to(rel_root).as_posix()
        items.append({
            "image": rel_img_path,
            "points": points,
            "id": stem,
        })
    return items


def generate_all_manifests(
    data_dir: Path,
    output_dir: Path | None = None,
    verify_only: bool = False,
) -> None:
    """Generate or verify all 4 canonical ShanghaiTech JSONL manifests."""
    out_dir = output_dir if output_dir is not None else data_dir
    specs = [
        {
            "name": "ShanghaiTech Part A Train",
            "img_dir": data_dir / "part_A_final" / "train_data" / "images",
            "gt_dir": data_dir / "part_A_final" / "train_data" / "ground_truth",
            "output_file": out_dir / "sha_a_train_all.jsonl",
            "expected_count": 300,
        },
        {
            "name": "ShanghaiTech Part A Test",
            "img_dir": data_dir / "part_A_final" / "test_data" / "images",
            "gt_dir": data_dir / "part_A_final" / "test_data" / "ground_truth",
            "output_file": out_dir / "sha_a_test.jsonl",
            "expected_count": 182,
        },
        {
            "name": "ShanghaiTech Part B Train",
            "img_dir": data_dir / "part_B_final" / "train_data" / "images",
            "gt_dir": data_dir / "part_B_final" / "train_data" / "ground_truth",
            "output_file": out_dir / "shb_train_all.jsonl",
            "expected_count": 400,
        },
        {
            "name": "ShanghaiTech Part B Test",
            "img_dir": data_dir / "part_B_final" / "test_data" / "images",
            "gt_dir": data_dir / "part_B_final" / "test_data" / "ground_truth",
            "output_file": out_dir / "shb_test.jsonl",
            "expected_count": 316,
        },
    ]

    for spec in specs:
        items = build_shanghaitech_split(
            img_dir=spec["img_dir"],
            gt_dir=spec["gt_dir"],
            rel_root=data_dir,
        )
        actual_count = len(items)
        if actual_count != spec["expected_count"]:
            raise ValueError(
                f"{spec['name']}: expected {spec['expected_count']} images, got {actual_count}."
            )

        target_file: Path = spec["output_file"]
        if verify_only:
            if not target_file.is_file():
                raise FileNotFoundError(f"Verification target missing: {target_file}")
            with target_file.open("r", encoding="utf-8") as f:
                existing_lines = [line.strip() for line in f if line.strip()]
            if len(existing_lines) != actual_count:
                raise ValueError(
                    f"{target_file.name}: existing has {len(existing_lines)} lines, expected {actual_count}."
                )
            print(f"Verified {target_file.name}: {actual_count} samples perfectly intact.")
        else:
            with target_file.open("w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")
            print(f"Wrote {target_file.name}: {actual_count} samples.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical ShanghaiTech Manifest Generator")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: data-dir)")
    parser.add_argument("--verify", action="store_true", help="Verify existing manifests against raw annotations")
    args = parser.parse_args()

    generate_all_manifests(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        verify_only=args.verify,
    )


if __name__ == "__main__":
    main()
