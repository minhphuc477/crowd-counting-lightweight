#!/usr/bin/env python3
"""Canonical JHU-CROWD++ Manifest Generator & Forensics Profiler.

Generates:
- data/jhu_train.jsonl (2,272 images)
- data/jhu_val.jsonl   (500 images)
- data/jhu_test.jsonl  (1,600 images)
- data/jhu_all.jsonl   (4,372 images)

From extracted official JHU-CROWD++ (v2.0) directory structure:
  jhu_crowd_v2.0/
  ├── train/ (images/*.jpg, gt/*.txt)
  ├── val/   (images/*.jpg, gt/*.txt)
  └── test/  (images/*.jpg, gt/*.txt)

GT format per line:
  x y w h blur occlusion
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from PIL import Image


def parse_jhu_gt(gt_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Parse JHU-CROWD++ gt text file, extracting valid [x, y] coordinates."""
    if not gt_path.is_file():
        return []
    
    pts = []
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            # Validate coordinates are within image boundaries
            if 0.0 <= x < float(img_w) and 0.0 <= y < float(img_h):
                pts.append([round(x, 2), round(y, 2)])
    return pts


def build_split(
    split_dir: Path,
    split_name: str,
    repo_data_root: Path,
) -> list[dict]:
    img_dir = split_dir / "images"
    gt_dir = split_dir / "gt"
    if not gt_dir.exists():
        gt_dir = split_dir / "ground_truth"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {img_dir}")
    
    jpg_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"[{split_name.upper()}] Found {len(jpg_files)} images in {img_dir}")

    items = []
    for img_p in jpg_files:
        stem = img_p.stem
        gt_p = gt_dir / f"{stem}.txt"
        
        # Get image dimensions safely
        with Image.open(img_p) as im:
            w, h = im.size
        
        pts = parse_jhu_gt(gt_p, w, h)
        
        # Relative path from repo root
        try:
            rel_img = img_p.relative_to(repo_data_root.parent)
        except ValueError:
            rel_img = img_p
        
        items.append({
            "image": str(rel_img).replace("\\", "/"),
            "points": pts,
            "id": f"jhu_{split_name}_{stem}",
            "width": w,
            "height": h,
        })
    return items


def main():
    parser = argparse.ArgumentParser(description="Build canonical JHU-CROWD++ manifests")
    parser.add_argument(
        "--data-dir", default="data/jhu_crowd_v2.0",
        help="Root path to extracted JHU-CROWD++ dataset",
    )
    parser.add_argument(
        "--out-dir", default="data",
        help="Directory to save .jsonl manifests",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        # Check alternative common names
        for alt in ["data/jhu_crowd", "data/jhu", "data/jhu_crowd_v2.0"]:
            if Path(alt).exists():
                data_dir = Path(alt)
                break
        else:
            print(f"Error: JHU-CROWD++ directory not found at '{data_dir}'.")
            print("Please download and extract JHU-CROWD++ into data/jhu_crowd_v2.0/")
            return

    print(f"Processing JHU-CROWD++ from: {data_dir.resolve()}")
    repo_data_root = Path("data").resolve()

    splits_data = {}
    total_items = []
    expected_counts = {"train": 2272, "val": 500, "test": 1600}

    for split in ["train", "val", "test"]:
        split_path = data_dir / split
        if not split_path.exists():
            print(f"Warning: Split directory '{split_path}' does not exist.")
            continue
        items = build_split(split_path, split, repo_data_root)
        splits_data[split] = items
        total_items.extend(items)

        exp_n = expected_counts.get(split)
        if exp_n and len(items) != exp_n:
            print(f"Note: Split '{split}' has {len(items)} images (standard is {exp_n}).")

        # Save individual split manifest
        manifest_p = out_dir / f"jhu_{split}.jsonl"
        with open(manifest_p, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"Saved: {manifest_p} ({len(items)} samples)")

    # Save complete dataset manifest
    all_manifest_p = out_dir / "jhu_all.jsonl"
    with open(all_manifest_p, "w", encoding="utf-8") as f:
        for it in total_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Saved: {all_manifest_p} ({len(total_items)} samples)")

    # Compute forensics for training split
    train_items = splits_data.get("train", [])
    if train_items:
        counts = [len(x["points"]) for x in train_items]
        zero_scenes = sum(1 for c in counts if c == 0)
        total_pts = sum(counts)
        mean_cnt = total_pts / len(counts)
        
        # Calculate stride-2 and stride-4 cell counts (with max_size=2048 capping)
        total_cells_s2 = 0
        total_cells_s4 = 0
        for it in train_items:
            w0, h0 = it["width"], it["height"]
            max_d = max(w0, h0)
            if max_d > 2048:
                scale = 2048.0 / float(max_d)
                wc, hc = int(round(w0 * scale)), int(round(h0 * scale))
            else:
                wc, hc = w0, h0
            total_cells_s2 += math.ceil(hc / 2) * math.ceil(wc / 2)
            total_cells_s4 += math.ceil(hc / 4) * math.ceil(wc / 4)

        m0_s2 = total_pts / total_cells_s2 if total_cells_s2 else 0.0
        m0_s4 = total_pts / total_cells_s4 if total_cells_s4 else 0.0

        forensics = {
            "dataset": "JHU-CROWD++",
            "train_samples": len(train_items),
            "val_samples": len(splits_data.get("val", [])),
            "test_samples": len(splits_data.get("test", [])),
            "total_samples": len(total_items),
            "train_total_heads": total_pts,
            "train_mean_heads": round(mean_cnt, 2),
            "train_min_heads": min(counts),
            "train_max_heads": max(counts),
            "train_zero_scenes": zero_scenes,
            "calibrated_init_m0_stride2": round(m0_s2, 6),
            "calibrated_init_m0_stride4": round(m0_s4, 6),
        }

        forensics_p = out_dir / "jhu_forensics_summary.json"
        with open(forensics_p, "w", encoding="utf-8") as f:
            json.dump(forensics, f, indent=2)
        print(f"\nJHU-CROWD++ Forensics Summary saved to {forensics_p}:")
        for k, v in forensics.items():
            print(f"  {k:28s}: {v}")


if __name__ == "__main__":
    main()
