from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image


def compute_manifest_occupancy(
    manifest_path: str | Path,
    stride: int = 4,
) -> dict[str, Any]:
    """Computes stride-s lattice occupancy statistics on a dataset manifest.

    Args:
        manifest_path: Path to JSONL manifest containing 'image' and 'points'.
        stride: Lattice stride in pixels (default 4).

    Returns:
        Dictionary containing cell count distribution, head counts, and multi-occupancy ratios.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    total_cells = 0
    total_heads = 0
    max_occupancy = 0
    occupancy_counts = {0: 0, 1: 0, 2: 0, 3: 0, ">=4": 0}
    detailed_counts: dict[int, int] = {}

    num_images = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            num_images += 1

            img_path = Path(item["image"])
            if not img_path.is_absolute() and not img_path.exists():
                candidate = path.parent / img_path
                if candidate.exists():
                    img_path = candidate

            raw_points = item.get("points", [])
            points = np.asarray(raw_points, dtype=np.float64)
            if points.size == 0 or len(raw_points) == 0:
                points = np.empty((0, 2), dtype=np.float64)
            elif points.ndim == 1:
                points = points.reshape(-1, 2)

            # Retrieve image dimensions
            if img_path.exists():
                with Image.open(img_path) as img:
                    w, h = img.size
            else:
                if len(points) > 0:
                    w = int(math.ceil(np.max(points[:, 0]) + 1))
                    h = int(math.ceil(np.max(points[:, 1]) + 1))
                else:
                    w, h = 1024, 768

            gw = int(math.ceil(w / stride))
            gh = int(math.ceil(h / stride))
            n_cells_img = gw * gh
            total_cells += n_cells_img

            if len(points) == 0:
                occupancy_counts[0] += n_cells_img
                detailed_counts[0] = detailed_counts.get(0, 0) + n_cells_img
                continue

            x = points[:, 0]
            y = points[:, 1]
            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x = x[valid]
            y = y[valid]

            n_valid_heads = len(x)
            total_heads += n_valid_heads

            if n_valid_heads == 0:
                occupancy_counts[0] += n_cells_img
                detailed_counts[0] = detailed_counts.get(0, 0) + n_cells_img
                continue

            j = np.clip(np.floor((x + 0.5) / stride).astype(np.int64), 0, gw - 1)
            i = np.clip(np.floor((y + 0.5) / stride).astype(np.int64), 0, gh - 1)
            flat_indices = i * gw + j

            counts = np.bincount(flat_indices, minlength=n_cells_img)
            img_max = int(np.max(counts))
            if img_max > max_occupancy:
                max_occupancy = img_max

            bin_counts = np.bincount(counts)
            for k, count in enumerate(bin_counts):
                count_int = int(count)
                detailed_counts[k] = detailed_counts.get(k, 0) + count_int
                if k == 0:
                    occupancy_counts[0] += count_int
                elif k == 1:
                    occupancy_counts[1] += count_int
                elif k == 2:
                    occupancy_counts[2] += count_int
                elif k == 3:
                    occupancy_counts[3] += count_int
                else:
                    occupancy_counts[">=4"] += count_int

    # Derived metrics
    occupied_cells = sum(detailed_counts.get(k, 0) for k in detailed_counts if k >= 1)
    multi_cells = sum(detailed_counts.get(k, 0) for k in detailed_counts if k >= 2)
    heads_in_multi_cells = sum(k * detailed_counts.get(k, 0) for k in detailed_counts if k >= 2)

    multi_head_ratio = float(heads_in_multi_cells / total_heads) if total_heads > 0 else 0.0
    multi_cell_ratio_occupied = float(multi_cells / occupied_cells) if occupied_cells > 0 else 0.0
    multi_cell_ratio_all = float(multi_cells / total_cells) if total_cells > 0 else 0.0

    return {
        "manifest": str(path.name),
        "num_images": num_images,
        "stride": stride,
        "total_cells": total_cells,
        "total_heads": total_heads,
        "occupied_cells": occupied_cells,
        "multi_cells": multi_cells,
        "heads_in_multi_cells": heads_in_multi_cells,
        "multi_head_ratio": multi_head_ratio,
        "multi_cell_ratio_occupied": multi_cell_ratio_occupied,
        "multi_cell_ratio_all": multi_cell_ratio_all,
        "max_occupancy": max_occupancy,
        "occupancy_histogram": occupancy_counts,
        "detailed_histogram": detailed_counts,
    }


def format_occupancy_table(results: list[dict[str, Any]]) -> str:
    """Formats a markdown table from a list of occupancy result dictionaries."""
    lines = [
        "| Dataset Split | Images | Heads | Stride | Occ=0 | Occ=1 | Occ=2 | Occ=3 | Occ>=4 | Max Occ | Multi-Head Ratio |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        hist = r["occupancy_histogram"]
        mhr_pct = f"{r['multi_head_ratio'] * 100:.2f}%"
        lines.append(
            f"| **{r['manifest']}** | {r['num_images']} | {r['total_heads']:,} | s={r['stride']} | "
            f"{hist[0]:,} | {hist[1]:,} | {hist[2]:,} | {hist[3]:,} | {hist['>=4']:,} | "
            f"{r['max_occupancy']} | **{mhr_pct}** |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="L0: Stride-s cell occupancy ceiling analysis.")
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=["data/sha_a_train.jsonl", "data/sha_a_val.jsonl", "data/sha_a_test.jsonl"],
        help="One or more JSONL manifest paths.",
    )
    parser.add_argument("--stride", type=int, default=4, help="Spatial stride in pixels (default 4).")
    args = parser.parse_args()

    results = []
    print(f"=== Running L0 Occupancy Ceiling Test (stride={args.stride}) ===")
    for m in args.manifests:
        p = Path(m)
        if not p.exists():
            print(f"Skipping non-existent manifest: {m}")
            continue
        res = compute_manifest_occupancy(p, stride=args.stride)
        results.append(res)
        print(f"Processed {p.name}: {res['total_heads']:,} heads across {res['num_images']} images. "
              f"Multi-head ratio: {res['multi_head_ratio']*100:.2f}% (max occ: {res['max_occupancy']})")

    print("\n" + format_occupancy_table(results))


if __name__ == "__main__":
    main()
