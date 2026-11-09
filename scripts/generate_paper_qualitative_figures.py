#!/usr/bin/env python3
"""Publication Qualitative Figures Generator (Figure 4 for CVPR / ICCV / TPAMI).

Generates high-resolution multi-column visual comparison panels across density tiers:
  Row 1: Sparse crowd (< 100 people)
  Row 2: Moderate crowd (100 - 500 people)
  Row 3: Dense crowd (500 - 1000 people)
  Row 4: Ultra-dense crowd (> 1000 people)

Columns:
  Col 1: Input RGB Image (with GT and Pred Count annotations)
  Col 2: Ground Truth Points Overlay (exact labeled heads)
  Col 3: Predicted Continuous Density Map (y*, 'turbo' colormap)
  Col 4: Absolute Error Heatmap (|y* - y_GT|, 'magma' colormap)

Strict compliance with AGENTS.md: Zero buzzwords. Pure empirical visualization.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

from rmr_v3.eval import load_model_from_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication qualitative visual comparison figure")
    parser.add_argument("--checkpoint", default="runs/benchmarks/exp_sha_direct_pretrained_104k/best_val_mae.pt")
    parser.add_argument("--manifest", default="data/sha_a_test.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint} on {device}...")
    model, _, _, cfg = load_model_from_ckpt(Path(args.checkpoint), device=device)
    model.eval()

    # Read manifest and select 4 representative samples across density tiers
    samples = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Stratify by count
    sparse_cands = [s for s in samples if 30 <= len(s.get("points", [])) <= 90]
    mod_cands = [s for s in samples if 180 <= len(s.get("points", [])) <= 320]
    dense_cands = [s for s in samples if 550 <= len(s.get("points", [])) <= 750]
    extreme_cands = [s for s in samples if len(s.get("points", [])) >= 1100]

    selected = [
        ("Sparse", sparse_cands[0] if sparse_cands else samples[0]),
        ("Moderate", mod_cands[0] if mod_cands else samples[len(samples)//4]),
        ("Dense", dense_cands[0] if dense_cands else samples[len(samples)//2]),
        ("Extreme", extreme_cands[0] if extreme_cands else samples[-1]),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 14), dpi=300)
    plt.subplots_adjust(wspace=0.04, hspace=0.08)

    col_titles = [
        "Input RGB Image",
        "Ground Truth Points",
        "RMR-104k Density Map (y*)",
        "Absolute Error (|y* - y_GT|)",
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row_idx, (tier_name, sample) in enumerate(selected):
        img_rel = sample["image"]
        img_path = Path("data") / img_rel if not Path(img_rel).exists() else Path(img_rel)
        raw_img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = raw_img.size

        # Downscale to max 1024 for clean visual rendering if necessary
        max_dim = max(w_orig, h_orig)
        scale = 1.0
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            w_disp, h_disp = int(round(w_orig * scale)), int(round(h_orig * scale))
            disp_img = raw_img.resize((w_disp, h_disp), Image.Resampling.BILINEAR)
        else:
            w_disp, h_disp = w_orig, h_orig
            disp_img = raw_img

        # Pad to multiple of 32 for model forward pass
        pad_h = (32 - h_disp % 32) % 32
        pad_w = (32 - w_disp % 32) % 32
        img_tensor = torch.from_numpy(np.array(disp_img)).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        norm_img = (img_tensor - mean) / std
        if pad_h > 0 or pad_w > 0:
            norm_img = F.pad(norm_img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        norm_img = norm_img.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(norm_img)
            y_pred = out["y"][0, 0]  # [H_out, W_out]
            # Crop padding
            h_out = h_disp // 2
            w_out = w_disp // 2
            y_pred = y_pred[:h_out, :w_out].detach().cpu().numpy()

        pred_count = float(y_pred.sum())
        gt_points = np.array(sample.get("points", [])) * scale
        gt_count = len(gt_points)

        # Generate GT density map for comparison via fast 2D Gaussian accumulation
        gt_map = np.zeros((h_out, w_out), dtype=np.float32)
        stride = 2
        for pt in gt_points:
            px, py = int(round(pt[0] / stride)), int(round(pt[1] / stride))
            if 0 <= px < w_out and 0 <= py < h_out:
                gt_map[py, px] += 1.0

        # Gaussian smoothing on GT for error heatmap
        import scipy.ndimage
        gt_smooth = scipy.ndimage.gaussian_filter(gt_map, sigma=1.5)
        # Normalize sum of gt_smooth to gt_count
        if gt_smooth.sum() > 0:
            gt_smooth = gt_smooth * (gt_count / gt_smooth.sum())

        abs_error_map = np.abs(y_pred - gt_smooth)

        # ----------------------------------------------------
        # Col 1: Input RGB + Text Annotation
        # ----------------------------------------------------
        ax1 = axes[row_idx, 0]
        ax1.imshow(disp_img)
        ax1.set_ylabel(f"{tier_name}\n({h_orig}x{w_orig})", fontsize=11, fontweight="bold")
        ax1.text(0.03, 0.08, f"GT: {gt_count} | Pred: {pred_count:.1f} (Δ: {pred_count - gt_count:+.1f})",
                 transform=ax1.transAxes, fontsize=9.5, fontweight="bold",
                 color="white", bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75))
        ax1.set_xticks([])
        ax1.set_yticks([])

        # ----------------------------------------------------
        # Col 2: Ground Truth Points Overlay
        # ----------------------------------------------------
        ax2 = axes[row_idx, 1]
        ax2.imshow(disp_img)
        if len(gt_points) > 0:
            ax2.scatter(gt_points[:, 0], gt_points[:, 1], c="#E53E3E", s=6, marker="o", alpha=0.85, edgecolors="none")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # ----------------------------------------------------
        # Col 3: Predicted Continuous Density Map
        # ----------------------------------------------------
        ax3 = axes[row_idx, 2]
        im3 = ax3.imshow(y_pred, cmap="turbo", interpolation="bilinear")
        ax3.set_xticks([])
        ax3.set_yticks([])

        # ----------------------------------------------------
        # Col 4: Absolute Error Heatmap
        # ----------------------------------------------------
        ax4 = axes[row_idx, 3]
        im4 = ax4.imshow(abs_error_map, cmap="magma", interpolation="bilinear")
        ax4.set_xticks([])
        ax4.set_yticks([])

    png_path = out_dir / "fig4_qualitative_density_maps.png"
    pdf_path = out_dir / "fig4_qualitative_density_maps.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Figure 4 successfully generated:")
    print(f"  - PNG: {png_path} ({png_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
