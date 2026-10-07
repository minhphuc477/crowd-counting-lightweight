#!/usr/bin/env python3
"""Canonical NWPU-Crowd Manifest Generator & Forensics Profiler.

Generates data/nwpu_train.jsonl, data/nwpu_val.jsonl, data/nwpu_test.jsonl
from official annotations in F:/PET/data/downloads/NWPU-test, and computes
exact empirical forensic parameters (m_0, negative sample count, 1-NN distances).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_split_txt(path: Path) -> List[Tuple[str, int, int]]:
    """Parse NWPU split file format: image_id scene_type luminance_code."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_id = parts[0]
            scene = int(parts[1]) if len(parts) > 1 else -1
            luminance = int(parts[2]) if len(parts) > 2 else -1
            records.append((img_id, scene, luminance))
    return records


def compute_knn_sample(points: np.ndarray, max_points: int = 1500) -> np.ndarray:
    """Sample pairwise 1-NN Euclidean distances without OOM."""
    if len(points) <= 1:
        return np.empty(0, dtype=np.float32)
    pts = points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return dists.min(axis=-1)


def generate_manifests(
    source_dir: Path,
    out_dir: Path,
    loc_file: Path | None = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsons_zip = source_dir / "jsons.zip"
    if not jsons_zip.exists():
        raise FileNotFoundError(f"Missing {jsons_zip}")

    train_records = parse_split_txt(source_dir / "train.txt")
    val_records = parse_split_txt(source_dir / "val.txt")
    test_records = parse_split_txt(source_dir / "test.txt")

    print(f"Loaded NWPU split lists: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")

    forensics: Dict[str, Any] = {
        "dataset": "NWPU-Crowd",
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_test": len(test_records),
        "train_counts": [],
        "val_counts": [],
        "zero_count_samples": 0,
        "negative_scenes": 0,
        "all_nn_distances": [],
    }

    with zipfile.ZipFile(jsons_zip, "r") as zf:
        for split_name, records in [("train", train_records), ("val", val_records)]:
            manifest_path = out_dir / f"nwpu_{split_name}.jsonl"
            lines_written = 0
            with open(manifest_path, "w", encoding="utf-8") as out_f:
                for img_id, scene, luminance in records:
                    json_name = f"{img_id}.json"
                    data = json.loads(zf.read(json_name))
                    raw_pts = data.get("points", [])
                    pts = [[float(p[0]), float(p[1])] for p in raw_pts]
                    boxes = data.get("boxes", [])
                    count = len(pts)

                    if split_name == "train":
                        forensics["train_counts"].append(count)
                    else:
                        forensics["val_counts"].append(count)

                    if count == 0:
                        forensics["zero_count_samples"] += 1
                    if scene == 0:
                        forensics["negative_scenes"] += 1

                    if 1 < count < 2500 and (lines_written % 15 == 0):
                        pts_arr = np.array(pts, dtype=np.float32)
                        nn_d = compute_knn_sample(pts_arr)
                        if len(nn_d) > 0:
                            forensics["all_nn_distances"].extend(nn_d.tolist())

                    entry = {
                        "id": img_id,
                        "image": f"data/nwpu/images/{img_id}.jpg",
                        "points": pts,
                        "boxes": boxes,
                        "scene_type": scene,
                        "luminance": luminance,
                        "count": count,
                    }
                    out_f.write(json.dumps(entry) + "\n")
                    lines_written += 1
            print(f"  Wrote {manifest_path} ({lines_written} lines)")

    test_manifest_path = out_dir / "nwpu_test.jsonl"
    with open(test_manifest_path, "w", encoding="utf-8") as out_f:
        for img_id, scene, luminance in test_records:
            entry = {
                "id": img_id,
                "image": f"data/nwpu/images/{img_id}.jpg",
                "scene_type": scene,
                "luminance": luminance,
            }
            out_f.write(json.dumps(entry) + "\n")
    print(f"  Wrote {test_manifest_path} ({len(test_records)} lines)")

    if loc_file and loc_file.exists():
        loc_target = out_dir / "nwpu_val_gt_loc.txt"
        shutil.copy2(loc_file, loc_target)
        print(f"  Copied official localization benchmark -> {loc_target}")

    tr_cnt = np.array(forensics["train_counts"], dtype=np.float64)
    val_cnt = np.array(forensics["val_counts"], dtype=np.float64)
    nn_arr = np.array(forensics["all_nn_distances"], dtype=np.float64)

    summary = {
        "dataset": "NWPU-Crowd",
        "total_labeled_images": len(tr_cnt) + len(val_cnt),
        "total_test_images": len(test_records),
        "train_images": len(tr_cnt),
        "val_images": len(val_cnt),
        "train_total_heads": int(tr_cnt.sum()),
        "val_total_heads": int(val_cnt.sum()),
        "train_mean_count": float(tr_cnt.mean()),
        "train_median_count": float(np.median(tr_cnt)),
        "train_max_count": int(tr_cnt.max()),
        "train_min_count": int(tr_cnt.min()),
        "val_mean_count": float(val_cnt.mean()),
        "val_median_count": float(np.median(val_cnt)),
        "val_max_count": int(val_cnt.max()),
        "zero_count_images": int(forensics["zero_count_samples"]),
        "zero_count_fraction": float(forensics["zero_count_samples"] / (len(tr_cnt) + len(val_cnt))),
        "density_tiers": {
            "sparse_le_100": int((tr_cnt <= 100).sum()),
            "moderate_101_to_500": int(((tr_cnt > 100) & (tr_cnt <= 500)).sum()),
            "dense_gt_500": int((tr_cnt > 500).sum()),
        },
        "knn_sample_count": len(nn_arr),
        "knn_median_distance_px": float(np.median(nn_arr)) if len(nn_arr) > 0 else 0.0,
        "knn_pct_sub_4px": float((nn_arr < 4.0).sum() / len(nn_arr) * 100) if len(nn_arr) > 0 else 0.0,
        "knn_pct_sub_16px": float((nn_arr < 16.0).sum() / len(nn_arr) * 100) if len(nn_arr) > 0 else 0.0,
    }

    summary_file = out_dir / "nwpu_forensics_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved forensic metrics to {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate NWPU-Crowd Manifests and Forensic Profiles")
    parser.add_argument("--source_dir", default="F:/PET/data/downloads/NWPU-test")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument(
        "--loc_file",
        default="F:/PET/.codex_research_tmp_20260804/SCALNet/eval/val_gt_loc.txt",
    )
    args = parser.parse_args()
    summary = generate_manifests(
        source_dir=Path(args.source_dir),
        out_dir=Path(args.out_dir),
        loc_file=Path(args.loc_file) if args.loc_file else None,
    )
    print("\nNWPU-Crowd Forensics Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
