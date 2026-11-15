#!/usr/bin/env python3
"""Deep Forensic Analysis & Visual Inspector for NWPU-Crowd Dataset.

Analyzes image resolutions, luminance categories, scene types, density tiers,
negative samples (zero count), and 1-NN head distance distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw


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


def analyze_manifest(manifest_path: Path, check_images: bool = True) -> Dict[str, Any]:
    """Parse and profile NWPU jsonl manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    counts = [int(x.get("count", len(x.get("points", [])))) for x in items]
    scenes = [int(x.get("scene_type", -1)) for x in items]
    luminances = [int(x.get("luminance", -1)) for x in items]

    widths = []
    heights = []
    aspect_ratios = []
    missing_images = 0

    if check_images:
        for it in items:
            img_p = Path(it.get("image", ""))
            if not img_p.is_file():
                missing_images += 1
                continue
            try:
                with Image.open(img_p) as im:
                    w, h = im.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)
            except Exception:
                missing_images += 1

    bins = {
        "zero (==0)": sum(1 for c in counts if c == 0),
        "sparse (1-100)": sum(1 for c in counts if 1 <= c <= 100),
        "moderate (101-500)": sum(1 for c in counts if 101 <= c <= 500),
        "dense (501-2000)": sum(1 for c in counts if 501 <= c <= 2000),
        "extreme (>2000)": sum(1 for c in counts if c > 2000),
    }

    # Scene types: 0=negative, 1=outdoor, 2=indoor
    scene_breakdown = {
        "negative (scene=0)": sum(1 for s in scenes if s == 0),
        "outdoor (scene=1)": sum(1 for s in scenes if s == 1),
        "indoor (scene=2)": sum(1 for s in scenes if s == 2),
    }

    # Luminance codes: 0=night, 1=low light, 2=normal light
    luminance_breakdown = {
        "night (lum=0)": sum(1 for l in luminances if l == 0),
        "low_light (lum=1)": sum(1 for l in luminances if l == 1),
        "normal (lum=2)": sum(1 for l in luminances if l == 2),
    }

    result = {
        "split": manifest_path.stem,
        "num_samples": len(items),
        "total_heads": int(sum(counts)),
        "mean_heads": round(float(np.mean(counts)), 2) if counts else 0.0,
        "std_heads": round(float(np.std(counts)), 2) if counts else 0.0,
        "median_heads": float(np.median(counts)) if counts else 0.0,
        "min_heads": int(min(counts)) if counts else 0,
        "max_heads": int(max(counts)) if counts else 0,
        "zero_count_ratio": round(sum(1 for c in counts if c == 0) / len(counts), 4) if counts else 0.0,
        "density_bins": bins,
        "scenes": scene_breakdown,
        "luminance": luminance_breakdown,
        "missing_images_on_disk": missing_images,
    }

    if widths:
        result["resolution"] = {
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "median_width": int(np.median(widths)),
            "median_height": int(np.median(heights)),
            "median_aspect_ratio": round(float(np.median(aspect_ratios)), 3),
        }

    return result


def render_sample_visualizations(items: List[Dict[str, Any]], out_dir: Path, num_samples: int = 4) -> None:
    """Render sample visual ground-truth overlays for forensic verification."""
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_items = [x for x in items if Path(x.get("image", "")).is_file()]
    if not valid_items:
        print("[WARN] No valid image files found on disk for visualization.")
        return

    sorted_items = sorted(valid_items, key=lambda x: int(x.get("count", len(x.get("points", [])))))
    selected = [
        ("zero_negative", sorted_items[0]),
        ("sparse_scene", sorted_items[len(sorted_items) // 4]),
        ("moderate_scene", sorted_items[len(sorted_items) // 2]),
        ("extreme_dense", sorted_items[-1]),
    ]

    print("\n[Visual Inspection] Generating sample overlays...")
    for tag, it in selected:
        img_p = Path(it["image"])
        try:
            im = Image.open(img_p).convert("RGB")
            draw = ImageDraw.Draw(im)
            pts = it.get("points", [])
            for pt in pts:
                x, y = float(pt[0]), float(pt[1])
                r = 3
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 0))

            out_p = out_dir / f"nwpu_sample_{tag}_{it['id']}_cnt{len(pts)}.jpg"
            im.save(out_p, quality=85)
            print(f"  Saved visual inspection: {out_p.name} (Count: {len(pts)})")
        except Exception as e:
            print(f"  Failed rendering {img_p}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and profile NWPU-Crowd dataset on Ubuntu/Local")
    parser.add_argument("--data-dir", default="data", help="Directory containing nwpu_*.jsonl files")
    parser.add_argument("--visualize", action="store_true", help="Render sample visual overlays")
    parser.add_argument("--out-dir", default="runs/nwpu_inspection", help="Output directory for visual audits")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 70)
    print("NWPU-CROWD DATASET FORENSIC PROFILER")
    print("=" * 70)

    all_summaries = {}
    train_items = []

    for split in ["train", "val", "test"]:
        manifest_p = data_dir / f"nwpu_{split}.jsonl"
        if not manifest_p.exists():
            print(f"  [MISSING] {manifest_p}")
            continue

        print(f"\nAnalyzing {split.upper()} manifest ({manifest_p})...")
        prof = analyze_manifest(manifest_p, check_images=True)
        all_summaries[split] = prof

        print(f"  Samples: {prof['num_samples']}")
        print(f"  Total Heads: {prof['total_heads']:,} (Mean: {prof['mean_heads']}, Median: {prof['median_heads']}, Max: {prof['max_heads']})")
        print(f"  Zero-count Ratio: {prof['zero_count_ratio'] * 100:.2f}% (Negative Scenes: {prof['scenes']['negative (scene=0)']})")
        print(f"  Luminance: Night={prof['luminance']['night (lum=0)']}, Low-Light={prof['luminance']['low_light (lum=1)']}, Normal={prof['luminance']['normal (lum=2)']}")
        if "resolution" in prof:
            res = prof["resolution"]
            print(f"  Resolution: Median {res['median_width']}x{res['median_height']} (Range: {res['min_width']}x{res['min_height']} to {res['max_width']}x{res['max_height']})")
        print(f"  Missing Images on Disk: {prof['missing_images_on_disk']}/{prof['num_samples']}")

        if split == "train":
            with open(manifest_p, "r", encoding="utf-8") as f:
                train_items = [json.loads(line) for line in f if line.strip()]

    # Save summary
    out_summary = data_dir / "nwpu_forensics_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved updated summary to: {out_summary}")

    if args.visualize and train_items:
        render_sample_visualizations(train_items, Path(args.out_dir))


if __name__ == "__main__":
    main()
