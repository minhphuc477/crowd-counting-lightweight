#!/usr/bin/env python3
"""Deep Forensic Analysis & Visual Inspector for JHU-CROWD++ Dataset."""
from __future__ import annotations

import json
import math
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def analyze_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    
    counts = [len(x["points"]) for x in items]
    widths = [x["width"] for x in items]
    heights = [x["height"] for x in items]
    aspect_ratios = [w / h for w, h in zip(widths, heights)]
    
    bins = {
        "zero (==0)": sum(1 for c in counts if c == 0),
        "sparse (1-50)": sum(1 for c in counts if 1 <= c <= 50),
        "moderate (51-500)": sum(1 for c in counts if 51 <= c <= 500),
        "dense (501-2000)": sum(1 for c in counts if 501 <= c <= 2000),
        "extreme (>2000)": sum(1 for c in counts if c > 2000),
    }
    
    return {
        "split": manifest_path.stem,
        "num_images": len(items),
        "total_heads": sum(counts),
        "mean_heads": round(float(np.mean(counts)), 2),
        "std_heads": round(float(np.std(counts)), 2),
        "median_heads": float(np.median(counts)),
        "min_heads": int(min(counts)),
        "max_heads": int(max(counts)),
        "bins": bins,
        "width_range": [int(min(widths)), int(max(widths))],
        "height_range": [int(min(heights)), int(max(heights))],
        "median_resolution": [int(np.median(widths)), int(np.median(heights))],
    }


def render_sample_visualizations(items: list[dict], out_dir: Path, num_samples: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pick 4 representative samples: zero, sparse, moderate, extreme
    sorted_items = sorted(items, key=lambda x: len(x["points"]))
    selected = [
        ("zero", sorted_items[0]),
        ("sparse", sorted_items[len(sorted_items) // 4]),
        ("moderate", sorted_items[len(sorted_items) // 2]),
        ("extreme", sorted_items[-1]),
    ]
    
    print("\n[Visual Inspection] Generating sample overlays...")
    for tag, it in selected:
        img_p = Path(it["image"])
        if not img_p.is_file():
            # try relative to data/
            img_p = Path("data") / img_p
        if not img_p.is_file():
            continue
        
        with Image.open(img_p) as raw_img:
            vis = raw_img.convert("RGB")
        
        draw = ImageDraw.Draw(vis)
        r = 3
        for pt in it["points"][:5000]:  # cap at 5000 for drawing speed
            x, y = pt[0], pt[1]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 0))
        
        save_p = out_dir / f"sample_{tag}_cnt{len(it['points'])}_{img_p.name}"
        vis.save(save_p, quality=90)
        print(f"  Rendered {tag.upper()}: count={len(it['points'])} -> {save_p}")


def main():
    manifests = [
        Path("data/jhu_train.jsonl"),
        Path("data/jhu_val.jsonl"),
        Path("data/jhu_test.jsonl"),
    ]
    
    if not all(m.exists() for m in manifests):
        print("Manifests not found yet. Run after download completes.")
        return
    
    reports = {}
    for m in manifests:
        rep = analyze_manifest(m)
        reports[m.stem] = rep
        print(f"\n=== Forensics for {m.stem.upper()} ===")
        for k, v in rep.items():
            print(f"  {k:20s}: {v}")
    
    with open("data/jhu_detailed_forensics.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print("\nDetailed forensics saved to data/jhu_detailed_forensics.json")
    
    with open("data/jhu_train.jsonl", "r", encoding="utf-8") as f:
        train_items = [json.loads(line) for line in f if line.strip()]
    render_sample_visualizations(train_items, Path("reports/jhu_visualizations"))


if __name__ == "__main__":
    main()
