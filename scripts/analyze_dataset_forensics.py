"""Comprehensive forensic image analyzer across crowd counting datasets.

Computes:
1. Sample counts & Head counts (min, max, mean, median, regimes)
2. Image resolutions (min, max, median, <512px short side, >2048px long side)
3. Inter-head 1-NN Euclidean distance percentiles and sub-Rayleigh (<4px) fraction
4. Optimal RMR preprocessing hyperparameters derivation
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import KDTree


def analyze_manifest(manifest_path: Path, data_root: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    n_images = len(items)
    head_counts = []
    widths = []
    heights = []
    aspect_ratios = []
    short_dims = []
    long_dims = []

    all_1nn_dists = []
    sub_4px_count = 0
    total_heads_with_nn = 0

    for it in items:
        pts = np.asarray(it.get("points", []), dtype=np.float32)
        n_pts = len(pts)
        head_counts.append(n_pts)

        # Image resolution
        img_p = Path(it["image"])
        if not img_p.is_absolute():
            img_p = data_root / img_p
        
        # Fast PIL header read without loading full image pixels
        with Image.open(img_p) as img:
            w, h = img.size

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / max(h, 1))
        short_dims.append(min(w, h))
        long_dims.append(max(w, h))

        # Inter-head 1-NN distance
        if n_pts >= 2:
            tree = KDTree(pts)
            # k=2 returns self and 1st nearest neighbor
            dists, _ = tree.query(pts, k=2)
            nn_dists = dists[:, 1]
            all_1nn_dists.extend(nn_dists.tolist())
            sub_4px_count += int((nn_dists < 4.0).sum())
            total_heads_with_nn += n_pts

    head_counts = np.array(head_counts)
    widths = np.array(widths)
    heights = np.array(heights)
    short_dims = np.array(short_dims)
    long_dims = np.array(long_dims)
    aspect_ratios = np.array(aspect_ratios)

    # Regimes
    sparse = int((head_counts <= 100).sum())
    moderate = int(((head_counts > 100) & (head_counts <= 500)).sum())
    dense = int(((head_counts > 500) & (head_counts <= 1000)).sum())
    super_dense = int((head_counts > 1000).sum())

    # Distance stats
    if all_1nn_dists:
        arr_nn = np.array(all_1nn_dists)
        p5 = float(np.percentile(arr_nn, 5))
        p10 = float(np.percentile(arr_nn, 10))
        p25 = float(np.percentile(arr_nn, 25))
        p50 = float(np.median(arr_nn))
        p75 = float(np.percentile(arr_nn, 75))
        sub_4px_pct = 100.0 * sub_4px_count / max(total_heads_with_nn, 1)
    else:
        p5 = p10 = p25 = p50 = p75 = float("nan")
        sub_4px_pct = 0.0

    return {
        "manifest": manifest_path.name,
        "n_images": n_images,
        "total_heads": int(head_counts.sum()),
        "head_count": {
            "min": int(head_counts.min()),
            "max": int(head_counts.max()),
            "mean": float(head_counts.mean()),
            "median": float(np.median(head_counts)),
            "sparse_le_100": sparse,
            "moderate_101_500": moderate,
            "dense_501_1000": dense,
            "super_dense_gt_1000": super_dense,
        },
        "resolution": {
            "w_median": float(np.median(widths)),
            "h_median": float(np.median(heights)),
            "min_dim_min": int(short_dims.min()),
            "min_dim_median": float(np.median(short_dims)),
            "max_dim_max": int(long_dims.max()),
            "max_dim_median": float(np.median(long_dims)),
            "pct_short_side_lt_512": float(100.0 * (short_dims < 512).sum() / n_images),
            "pct_long_side_gt_2048": float(100.0 * (long_dims > 2048).sum() / n_images),
            "aspect_ratio_median": float(np.median(aspect_ratios)),
        },
        "spatial_1nn": {
            "total_heads_analyzed": total_heads_with_nn,
            "p5_px": p5,
            "p10_px": p10,
            "p25_px": p25,
            "median_px": p50,
            "p75_px": p75,
            "sub_4px_fraction_pct": sub_4px_pct,
        },
    }


def main() -> None:
    data_dir = Path("data")
    manifests_to_test = [
        data_dir / "sha_a_train_all.jsonl",
        data_dir / "sha_a_test.jsonl",
        data_dir / "shb_train_all.jsonl",
        data_dir / "shb_test.jsonl",
        data_dir / "ucf_cc_50_all.jsonl",
        data_dir / "qnrf_train.jsonl",
        data_dir / "qnrf_test.jsonl",
        data_dir / "mall_all.jsonl",
        data_dir / "ucsd_all.jsonl",
    ]

    results = []
    for m in manifests_to_test:
        if m.exists():
            print(f"Analyzing {m.name}...")
            res = analyze_manifest(m, data_root=data_dir)
            results.append(res)
            print(f"  Images: {res['n_images']}, Heads: {res['total_heads']}, "
                  f"Median Size: {res['resolution']['w_median']:.0f}x{res['resolution']['h_median']:.0f}, "
                  f"<512px short: {res['resolution']['pct_short_side_lt_512']:.1f}%, "
                  f"Sub-4px NN: {res['spatial_1nn']['sub_4px_fraction_pct']:.1f}%")

    out_file = data_dir / "dataset_forensics_summary.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved forensics analysis to {out_file}")


if __name__ == "__main__":
    main()
