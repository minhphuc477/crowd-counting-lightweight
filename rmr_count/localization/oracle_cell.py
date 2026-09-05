from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

from rmr_count.localization.metrics import LocalizationMeter


def evaluate_oracle_cell_centers(
    manifest_path: str | Path,
    stride: int = 4,
    sigmas: list[float] | tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0),
) -> dict[str, Any]:
    """Runs L1 Oracle Cell-Center evaluation on a dataset manifest.

    For each image, generates ground-truth cell counts Y_ij^gt on stride-s grid,
    and emits k duplicate points at (s*j + s/2, s*i + s/2) for every cell with count k.
    Evaluates against actual GT points under varying distance thresholds sigma.

    Args:
        manifest_path: Path to JSONL manifest.
        stride: Stride of the grid (default 4).
        sigmas: List of distance thresholds in pixels.

    Returns:
        Summary metrics dictionary.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    meter = LocalizationMeter(sigmas=sigmas)
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

            points = np.asarray(item.get("points", []), dtype=np.float64)

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

            if len(points) == 0:
                meter.update(np.empty((0, 2)), np.empty((0, 2)))
                continue

            x = points[:, 0]
            y = points[:, 1]
            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            valid_gts = points[valid]

            if len(valid_gts) == 0:
                meter.update(np.empty((0, 2)), np.empty((0, 2)))
                continue

            # Rasterize
            j = np.clip(np.floor((valid_gts[:, 0] + 0.5) / stride).astype(np.int64), 0, gw - 1)
            i = np.clip(np.floor((valid_gts[:, 1] + 0.5) / stride).astype(np.int64), 0, gh - 1)

            # Emit points at cell center: (s * j + s/2, s * i + s/2)
            # For each instance in cell (i, j), emit a point at the center
            pred_x = stride * j + (stride / 2.0)
            pred_y = stride * i + (stride / 2.0)
            oracle_preds = np.column_stack([pred_x, pred_y])

            meter.update(oracle_preds, valid_gts)

    summary = meter.compute_summary()
    summary["manifest"] = str(path.name)
    summary["num_images"] = num_images
    summary["stride"] = stride
    return summary


def format_oracle_table(summary: dict[str, Any]) -> str:
    """Formats summary into a clean markdown table."""
    lines = [
        f"### L1 Oracle Cell-Center Results ({summary['manifest']}, stride={summary['stride']}, images={summary['num_images']}, points={summary['total_ground_truth']:,})\n",
        "| Distance Threshold $\\sigma$ | Micro Precision | Micro Recall | Micro $F_1$ | Macro Precision | Macro Recall | Macro $F_1$ | TP | FP | FN |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, data in summary["thresholds"].items():
        s = data["sigma"]
        lines.append(
            f"| **$\\sigma = {int(s) if s.is_integer() else s:.1f}$ px** | "
            f"{data['micro_precision']*100:.2f}% | {data['micro_recall']*100:.2f}% | **{data['micro_f1']*100:.2f}%** | "
            f"{data['macro_precision']*100:.2f}% | {data['macro_recall']*100:.2f}% | {data['macro_f1']*100:.2f}% | "
            f"{data['tp']:,} | {data['fp']:,} | {data['fn']:,} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="L1: Oracle Cell-Center localization ceiling test.")
    parser.add_argument("--manifest", default="data/sha_a_test.jsonl", help="JSONL manifest path.")
    parser.add_argument("--stride", type=int, default=4, help="Grid stride (default 4).")
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[2.0, 4.0, 6.0, 8.0, 10.0],
        help="Distance thresholds sigma in pixels.",
    )
    args = parser.parse_args()

    print(f"=== Running L1 Oracle Cell-Center Test on {args.manifest} (stride={args.stride}) ===")
    summary = evaluate_oracle_cell_centers(args.manifest, stride=args.stride, sigmas=args.sigmas)
    print("\n" + format_oracle_table(summary))


if __name__ == "__main__":
    main()
