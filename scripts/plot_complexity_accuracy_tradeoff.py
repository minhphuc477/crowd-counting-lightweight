#!/usr/bin/env python3
"""Empirical Complexity-Accuracy Trade-off Plot Generator for Top-Tier Publications.

Plots:
  Panel A: Model Parameters (Millions, log-scale) vs. MAE on ShanghaiTech Part A.
  Panel B: Forward GFLOPs (at 512x512, log-scale) vs. MAE on ShanghaiTech Part A.

Strict compliance with AGENTS.md: Zero use of buzzword jargon. Grounded strictly in empirical data.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Literature Baseline Data on ShanghaiTech Part A
    # Model: (name, params_M, gflops_512, mae, category)
    baselines = [
        # Heavy SOTA (VGG / ResNet / ViT Backbones, 16M - 35M params)
        {"name": "CSRNet (CVPR'18)", "params": 16.26, "gflops": 54.3, "mae": 68.2, "cat": "heavy"},
        {"name": "BL (ICCV'19)", "params": 21.50, "gflops": 67.8, "mae": 62.8, "cat": "heavy"},
        {"name": "DM-Count (NeurIPS'20)", "params": 21.50, "gflops": 67.8, "mae": 65.9, "cat": "heavy"},
        {"name": "P2PNet (ICCV'21)", "params": 24.10, "gflops": 95.4, "mae": 52.7, "cat": "heavy"},
        {"name": "GauNet (CVPR'22)", "params": 32.40, "gflops": 88.2, "mae": 58.7, "cat": "heavy"},
        {"name": "MAN (CVPR'22)", "params": 21.50, "gflops": 72.4, "mae": 56.8, "cat": "heavy"},
        {"name": "STEERER (ICCV'23)", "params": 21.50, "gflops": 68.1, "mae": 54.5, "cat": "heavy"},
        {"name": "PET (NeurIPS'23)", "params": 28.50, "gflops": 62.0, "mae": 52.1, "cat": "heavy"},

        # Mid / Compact Models (1M - 10M params)
        {"name": "MobileCount (ECCVW'20)", "params": 1.98, "gflops": 1.83, "mae": 82.3, "cat": "mid"},

        # Historical Lightweight Sub-250k Models (<= 0.25M params)
        {"name": "MCNN (CVPR'16)", "params": 0.13, "gflops": 2.80, "mae": 110.2, "cat": "light_prior"},
        {"name": "Lw-Count (TCSVT'22)", "params": 0.071, "gflops": 0.52, "mae": 78.2, "cat": "light_prior"},
        {"name": "LSANet (Neurocomp'22)", "params": 0.200, "gflops": 1.45, "mae": 66.1, "cat": "light_prior"},

        # OUR WORK (RMR-104k Family, <= 104k params, 0.0% Knowledge Distillation)
        {"name": "RMR-104k Scratch (Ours)", "params": 0.104, "gflops": 0.655, "mae": 70.50, "cat": "ours"},
        {"name": "RMR-104k Pretrained (Ours)", "params": 0.104, "gflops": 0.655, "mae": 64.91, "cat": "ours"},
        {"name": "RMR-102k No-Neck (Ours)", "params": 0.102, "gflops": 0.584, "mae": 66.24, "cat": "ours"},
    ]

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    color_map = {
        "heavy": "#4A5568",        # Slate grey
        "mid": "#D69E2E",          # Amber
        "light_prior": "#E53E3E",  # Muted red
        "ours": "#2B6CB0",         # Deep Royal Blue
    }

    # ==========================================
    # Panel A: Parameters vs. MAE (Log-scale X)
    # ==========================================
    for m in baselines:
        p, mae, cat, name = m["params"], m["mae"], m["cat"], m["name"]
        color = color_map[cat]
        marker = "*" if cat == "ours" else ("o" if cat == "heavy" else ("s" if cat == "mid" else "^"))
        size = 180 if cat == "ours" else (80 if cat == "heavy" else 90)
        edgecolor = "black" if cat == "ours" else "none"
        zorder = 5 if cat == "ours" else 3

        ax1.scatter(p, mae, color=color, marker=marker, s=size, edgecolors=edgecolor, linewidth=1.5 if cat == "ours" else 0.5, zorder=zorder)

        # Label offsets
        dx, dy = 1.12, 1.0
        if "RMR-104k Pretrained" in name:
            ax1.annotate(f"{name}\n(MAE {mae:.2f})", (p, mae), xytext=(p * 0.45, mae - 4.5),
                         fontsize=9, fontweight="bold", color="#1A365D",
                         arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1.2))
        elif "RMR-102k No-Neck" in name:
            ax1.annotate(f"{name} ({mae:.2f})", (p, mae), xytext=(p * 0.40, mae + 3.0),
                         fontsize=8.5, fontweight="bold", color="#2C5282")
        elif "RMR-104k Scratch" in name:
            ax1.annotate(f"{name} ({mae:.2f})", (p, mae), xytext=(p * 1.15, mae + 2.0),
                         fontsize=8.5, color="#2B6CB0")
        elif "LSANet" in name:
            ax1.annotate(f"{name} ({mae:.1f})", (p, mae), xytext=(p * 1.15, mae - 1.0), fontsize=8, color="#9B2C2C")
        elif "Lw-Count" in name:
            ax1.annotate(f"{name} ({mae:.1f})", (p, mae), xytext=(p * 0.45, mae + 3.0), fontsize=8, color="#9B2C2C")
        elif "MCNN" in name:
            ax1.annotate(f"{name} ({mae:.1f})", (p, mae), xytext=(p * 1.15, mae - 1.0), fontsize=8, color="#9B2C2C")
        elif "CSRNet" in name:
            ax1.annotate(f"{name} ({mae:.1f})", (p, mae), xytext=(p * 0.45, mae + 3.5), fontsize=8, color="#4A5568")
        elif "P2PNet" in name:
            ax1.annotate(f"{name} ({mae:.1f})", (p, mae), xytext=(p * 1.12, mae - 2.0), fontsize=8, color="#4A5568")
        elif "BL" in name:
            ax1.annotate(f"{name} ({mae:.1f})", (p, mae), xytext=(p * 1.12, mae + 1.5), fontsize=8, color="#4A5568")
        elif "MobileCount" in name:
            ax1.annotate(f"{name} ({mae:.1f})", (p, mae), xytext=(p * 1.15, mae + 1.5), fontsize=8, color="#744210")

    ax1.set_xscale("log")
    ax1.set_xlim(0.04, 50.0)
    ax1.set_ylim(45, 120)
    ax1.set_xlabel("Trainable Parameters (Millions, Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error (MAE, lower is better)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Accuracy vs. Parameter Budget (ShanghaiTech Part A)", fontsize=12, fontweight="bold", pad=10)

    # Highlight Sub-150k parameter budget band
    ax1.axvspan(0.04, 0.15, alpha=0.12, color="#3182CE", label="Ultra-Lightweight Regime (<= 150k params)")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    # ==========================================
    # Panel B: GFLOPs vs. MAE (Log-scale X)
    # ==========================================
    for m in baselines:
        gf, mae, cat, name = m["gflops"], m["mae"], m["cat"], m["name"]
        color = color_map[cat]
        marker = "*" if cat == "ours" else ("o" if cat == "heavy" else ("s" if cat == "mid" else "^"))
        size = 180 if cat == "ours" else (80 if cat == "heavy" else 90)
        edgecolor = "black" if cat == "ours" else "none"
        zorder = 5 if cat == "ours" else 3

        ax2.scatter(gf, mae, color=color, marker=marker, s=size, edgecolors=edgecolor, linewidth=1.5 if cat == "ours" else 0.5, zorder=zorder)

        if "RMR-104k Pretrained" in name:
            ax2.annotate(f"{name}\n(0.655 GFLOPs, MAE {mae:.2f})", (gf, mae), xytext=(gf * 0.40, mae - 5.0),
                         fontsize=9, fontweight="bold", color="#1A365D",
                         arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1.2))
        elif "RMR-102k No-Neck" in name:
            ax2.annotate(f"{name} ({mae:.2f})", (gf, mae), xytext=(gf * 0.35, mae + 3.0),
                         fontsize=8.5, fontweight="bold", color="#2C5282")
        elif "CSRNet" in name:
            ax2.annotate(f"{name} ({mae:.1f})", (gf, mae), xytext=(gf * 0.40, mae + 3.5), fontsize=8, color="#4A5568")
        elif "P2PNet" in name:
            ax2.annotate(f"{name} ({mae:.1f})", (gf, mae), xytext=(gf * 0.55, mae - 3.5), fontsize=8, color="#4A5568")
        elif "LSANet" in name:
            ax2.annotate(f"{name} ({mae:.1f})", (gf, mae), xytext=(gf * 1.15, mae - 1.0), fontsize=8, color="#9B2C2C")
        elif "Lw-Count" in name:
            ax2.annotate(f"{name} ({mae:.1f})", (gf, mae), xytext=(gf * 0.40, mae + 3.0), fontsize=8, color="#9B2C2C")
        elif "MCNN" in name:
            ax2.annotate(f"{name} ({mae:.1f})", (gf, mae), xytext=(gf * 1.15, mae - 1.0), fontsize=8, color="#9B2C2C")

    ax2.set_xscale("log")
    ax2.set_xlim(0.3, 150.0)
    ax2.set_ylim(45, 120)
    ax2.set_xlabel("Computational Complexity (GFLOPs @ 512x512, Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Absolute Error (MAE, lower is better)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Accuracy vs. Computational Complexity (ShanghaiTech Part A)", fontsize=12, fontweight="bold", pad=10)

    # Highlight Ultra-low GFLOPs band
    ax2.axvspan(0.3, 1.0, alpha=0.12, color="#3182CE", label="Sub-1 GFLOP Regime (Edge Micro-NPU Ready)")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    png_path = output_dir / "fig1_complexity_accuracy_tradeoff.png"
    pdf_path = output_dir / "fig1_complexity_accuracy_tradeoff.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Figure 1 successfully generated:")
    print(f"  - PNG: {png_path} ({png_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
