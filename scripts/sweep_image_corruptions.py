#!/usr/bin/env python3
"""Image Corruption & Negative Distractor Stress-Tester.

Sweeps:
  1. JPEG Compression Quality Factor: Q in [10, 20, 30, 40, 50, 70, 90, 100]
     Simulates real-world low-bandwidth RTSP surveillance camera video streams.
  2. Zero-Count Negative Distractor Scenes (N = 0) from JHU-CROWD++
     Measures exact Phantom Background Fog accumulation on empty backgrounds.

Strict compliance with AGENTS.md: Zero buzzwords. Pure empirical stress-testing.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.transforms import normalize_image
from rmr_v3.eval import load_model_from_ckpt


def compress_pil_jpeg(im: Image.Image, quality: int) -> Image.Image:
    """Compress and re-open PIL image in memory with JPEG quality Q."""
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep JPEG corruption and evaluate zero-count distractors")
    parser.add_argument("--checkpoint", default="runs/benchmarks/exp_sha_direct_pretrained_104k/best_val_mae.pt")
    parser.add_argument("--sha-manifest", default="data/sha_a_test.jsonl")
    parser.add_argument("--jhu-manifest", default="data/jhu_test.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint} on {device}...")
    model, _, _, cfg = load_model_from_ckpt(Path(args.checkpoint), device=device)
    model.eval()

    # ========================================================
    # 1. JPEG Compression Quality Sweep on ShanghaiTech Part A
    # ========================================================
    print(f"\n--- 1. JPEG Quality Sweep on ShanghaiTech Part A (182 images) ---")
    dataset_sha = CrowdManifestDataset(
        manifest=args.sha_manifest,
        train=False,
        data_root="data",
        max_size=2048,
        output_stride=2,
    )

    jpeg_qualities = [10, 20, 30, 40, 50, 70, 90, 100]
    jpeg_metrics = []

    for q in jpeg_qualities:
        abs_errs = []
        sq_errs = []
        for i in range(len(dataset_sha)):
            raw_sample = dataset_sha.items[i]
            img_rel = raw_sample["image"]
            img_p = Path("data") / img_rel if not Path(img_rel).exists() else Path(img_rel)
            gt = len(raw_sample.get("points", []))

            with Image.open(img_p) as raw_im:
                rgb_im = raw_im.convert("RGB")
                if q < 100:
                    rgb_im = compress_pil_jpeg(rgb_im, quality=q)

                w_orig, h_orig = rgb_im.size
                scale = 1.0
                if max(w_orig, h_orig) > 2048:
                    scale = 2048.0 / max(w_orig, h_orig)
                    rgb_im = rgb_im.resize((int(round(w_orig * scale)), int(round(h_orig * scale))), Image.Resampling.BILINEAR)

                w_curr, h_curr = rgb_im.size
                pad_h = (32 - h_curr % 32) % 32
                pad_w = (32 - w_curr % 32) % 32

                img_np = np.array(rgb_im, dtype=np.float32) / 255.0
                t_img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                t_img = normalize_image(t_img, mode="inception")
                if pad_h > 0 or pad_w > 0:
                    t_img = F.pad(t_img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

            with torch.no_grad():
                out = model(t_img)
                y = out["y"][0, 0][:h_curr // 2, :w_curr // 2]
                pred = float(y.sum().item())

            err = pred - gt
            abs_errs.append(abs(err))
            sq_errs.append(err ** 2)

        mae = float(np.mean(abs_errs))
        rmse = float(np.sqrt(np.mean(sq_errs)))
        jpeg_metrics.append({"quality": q, "mae": round(mae, 2), "rmse": round(rmse, 2)})
        print(f"  JPEG Quality Q={q:3d} | MAE={mae:6.2f} | RMSE={rmse:6.2f}")

    # ========================================================
    # 2. Zero-Count Negative Distractor Scenes (JHU-CROWD++)
    # ========================================================
    print(f"\n--- 2. Zero-Count Negative Distractor Evaluation (JHU N=0) ---")
    neg_samples = []
    if Path(args.jhu_manifest).exists():
        with open(args.jhu_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    pts = item.get("points", [])
                    cnt = item.get("count", len(pts))
                    if cnt == 0 and len(pts) == 0:
                        neg_samples.append(item)

    print(f"Found {len(neg_samples)} zero-count negative images in JHU test split.")
    neg_predictions = []
    if neg_samples:
        for i, s in enumerate(neg_samples):
            img_rel = s["image"]
            img_p = Path("data") / img_rel if not Path(img_rel).exists() else Path(img_rel)
            if not img_p.exists():
                continue
            with Image.open(img_p) as raw_im:
                rgb_im = raw_im.convert("RGB")
                w_orig, h_orig = rgb_im.size
                if max(w_orig, h_orig) > 2048:
                    scale = 2048.0 / max(w_orig, h_orig)
                    rgb_im = rgb_im.resize((int(round(w_orig * scale)), int(round(h_orig * scale))), Image.Resampling.BILINEAR)

                w_curr, h_curr = rgb_im.size
                pad_h = (32 - h_curr % 32) % 32
                pad_w = (32 - w_curr % 32) % 32

                img_np = np.array(rgb_im, dtype=np.float32) / 255.0
                t_img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                t_img = normalize_image(t_img, mode="inception")
                if pad_h > 0 or pad_w > 0:
                    t_img = F.pad(t_img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

            with torch.no_grad():
                out = model(t_img)
                y = out["y"][0, 0][:h_curr // 2, :w_curr // 2]
                pred = float(y.sum().item())

            neg_predictions.append({"id": s.get("id", f"neg_{i}"), "pred_count": round(pred, 2)})

    mean_neg_pred = float(np.mean([p["pred_count"] for p in neg_predictions])) if neg_predictions else 0.0
    print(f"Mean Phantom Count on Negative Distractors (N=0): {mean_neg_pred:.2f} people/image")

    # Save JSON summary
    summary = {
        "jpeg_corruption": jpeg_metrics,
        "negative_distractors": {
            "num_samples": len(neg_predictions),
            "mean_phantom_count": round(mean_neg_pred, 2),
            "samples": neg_predictions,
        },
    }
    json_path = out_dir / "image_corruptions_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved corruption sweep data to: {json_path}")

    # ==========================================
    # Generate Publication Plots (2 Panels)
    # ==========================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: MAE vs. JPEG Quality Factor
    qs = [m["quality"] for m in jpeg_metrics]
    maes = [m["mae"] for m in jpeg_metrics]
    rmses = [m["rmse"] for m in jpeg_metrics]

    ax1.plot(qs, maes, marker="o", markersize=7, linewidth=2.5, color="#2B6CB0", label="Test MAE")
    ax1.plot(qs, rmses, marker="s", markersize=6, linewidth=2.0, color="#E53E3E", linestyle="--", label="Test RMSE")
    for q, mae in zip(qs, maes):
        ax1.annotate(f"{mae:.1f}", (q, mae), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, fontweight="bold")

    ax1.set_xlabel("JPEG Compression Quality Factor Q (10 = severe compression, 100 = uncompressed)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Counting Error", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Robustness to JPEG Video Compression (ShanghaiTech Part A)", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel B: Histogram / Bar chart of Phantom Background Fog on Negative Distractors
    if neg_predictions:
        preds = [p["pred_count"] for p in neg_predictions]
        ax2.hist(preds, bins=12, color="#4A5568", edgecolor="black", alpha=0.85)
        ax2.axvline(mean_neg_pred, color="#E53E3E", linestyle="--", linewidth=2.5, label=f"Mean Phantom Count ({mean_neg_pred:.1f})")
        ax2.axvline(0.0, color="green", linestyle=":", linewidth=2.0, label="Ideal Response (0.0)")
    ax2.set_xlabel("Predicted Count on True Empty Backgrounds (N = 0)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Test Images", fontsize=11, fontweight="bold")
    ax2.set_title(f"(b) Negative Distractor Phantom Mass Analysis (JHU N=0, {len(neg_predictions)} images)", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    fig_png = fig_dir / "fig5_robustness_jpeg_and_negative.png"
    fig_pdf = fig_dir / "fig5_robustness_jpeg_and_negative.pdf"
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.close()

    print(f"\nFigure 5 successfully generated:")
    print(f"  - PNG: {fig_png} ({fig_png.stat().st_size / 1024:.1f} KB)")
    print(f"  - PDF: {fig_pdf} ({fig_pdf.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
