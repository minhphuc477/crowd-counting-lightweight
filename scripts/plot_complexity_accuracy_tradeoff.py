#!/usr/bin/env python3
"""Empirical Complexity-Accuracy Trade-off Plot Generator for ShanghaiTech Part A.

Generates a two-panel scatter plot from verified run summaries and PyTorch FLOPs:
  Panel (a): Trainable Parameters (Millions, log-scale) vs. MAE.
  Panel (b): Forward GFLOPs (reported / FlopCounterMode @ 512x512, log-scale) vs. MAE.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from rmr_v3.config import load_config
from rmr_v3.model import RMRv3, RMRv3Config


def _load_mae(summary_path: Path) -> float:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing verified summary file: {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(data["mae"])


def _profile_rmr(config_path: Path) -> tuple[float, float]:
    """Return (trainable_params_millions, forward_gflops_512) without network downloads."""
    cfg = load_config(str(config_path))
    model_cfg = dict(cfg["model"])
    model_cfg["pretrained"] = False
    model = RMRv3(RMRv3Config.from_dict(model_cfg)).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        _ = model(x)
    return params_m, fc.get_total_flops() / 1e9


def _build_entries() -> list[dict[str, Any]]:
    p_104k, gf_104k = _profile_rmr(Path("configs/benchmarks/exp_sha_direct_pretrained_104k.yaml"))
    p_102k, gf_102k = _profile_rmr(Path("configs/benchmarks/ablations_sha/exp_sha_abl_b4_no_neck_104k.yaml"))

    mae_pre_seeds = [
        _load_mae(Path("runs/benchmarks/exp_sha_direct_pretrained_104k/eval_val/summary.json")),
        _load_mae(Path("runs/benchmarks/sweeps_train/exp_sha_seed_1337_104k/eval_val/summary.json")),
        _load_mae(Path("runs/benchmarks/sweeps_train/exp_sha_seed_2024_104k/eval_val/summary.json")),
    ]
    pre_mean = float(np.mean(mae_pre_seeds))
    pre_std = float(np.std(mae_pre_seeds, ddof=1))
    mae_scratch = _load_mae(Path("runs/benchmarks/exp_sha_direct_scratch_104k/eval_val/summary.json"))
    mae_noneck = _load_mae(Path("runs/benchmarks/ablations_sha/exp_sha_abl_b4_no_neck_104k/eval_val/summary.json"))

    # Verified literature baselines on ShanghaiTech Part A (non-distilled standalone counts)
    # Note: Prior GFLOPs reflect heterogeneous author evaluation protocols; RMR GFLOPs are @ 512x512.
    return [
        {"name": "PET (ICCV'23)", "params": 51.65, "gflops": None, "mae": 49.3, "cat": "heavy"},
        {"name": "P2PNet (ICCV'21)", "params": 19.22, "gflops": 104.7, "mae": 52.7, "cat": "heavy"},
        {"name": "DM-Count (NeurIPS'20)", "params": 21.50, "gflops": None, "mae": 59.7, "cat": "heavy"},
        {"name": "BL (ICCV'19)", "params": 21.50, "gflops": None, "mae": 62.8, "cat": "heavy"},
        {"name": "CSRNet (CVPR'18)", "params": 16.26, "gflops": 68.7, "mae": 68.2, "cat": "heavy"},
        {"name": "MobileCount (ECCVW'20)", "params": 3.40, "gflops": 6.15, "mae": 89.4, "cat": "mid"},
        {"name": "MCNN (CVPR'16)", "params": 0.133, "gflops": 2.80, "mae": 110.2, "cat": "light_prior"},
        {"name": "LSANet (Neurocomp'22)", "params": 0.200, "gflops": 14.40, "mae": 66.1, "cat": "light_prior"},
        {"name": "Lw-Count (TCSVT'22)", "params": 0.071, "gflops": None, "mae": 69.7, "cat": "light_prior"},
        {"name": "TinyCount (JRTIP'24)", "params": 0.060, "gflops": 0.16, "mae": 78.2, "cat": "light_prior"},
        {
            "name": "RMR-104k Pretrained (3-Seed)",
            "params": p_104k,
            "gflops": gf_104k,
            "mae": pre_mean,
            "mae_std": pre_std,
            "cat": "ours",
            "note": f"3-seed: {pre_mean:.2f}±{pre_std:.2f}\n(seed42: {mae_pre_seeds[0]:.2f})",
        },
        {"name": "RMR-104k Scratch (seed42)", "params": p_104k, "gflops": gf_104k, "mae": mae_scratch, "cat": "ours"},
        {"name": "RMR-102k No-Neck (Scratch)", "params": p_102k, "gflops": gf_102k, "mae": mae_noneck, "cat": "ours"},
    ]


def _plot_panel(
    ax: Any,
    entries: list[dict[str, Any]],
    x_key: str,
    offsets_a: dict[str, tuple[float, float]],
    color_map: dict[str, str],
) -> None:
    for m in entries:
        x_val = m[x_key]
        if x_val is None or math.isnan(x_val):
            continue
        mae, cat, name = m["mae"], m["cat"], m["name"]
        color = color_map[cat]
        marker = "*" if cat == "ours" else ("o" if cat == "heavy" else ("s" if cat == "mid" else "^"))
        size = 180 if cat == "ours" else (80 if cat == "heavy" else 90)
        ax.scatter(
            x_val,
            mae,
            color=color,
            marker=marker,
            s=size,
            edgecolors="black" if cat == "ours" else "none",
            linewidth=1.3 if cat == "ours" else 0.5,
            zorder=5 if cat == "ours" else 3,
        )
        if "mae_std" in m:
            ax.errorbar(x_val, mae, yerr=m["mae_std"], fmt="none", ecolor=color, capsize=3, lw=1.2, zorder=4)

        x_txt, y_txt = offsets_a.get(name, (x_val * 1.15, mae + 1.0))
        label = f"{name}\n{m['note']}" if "note" in m else f"{name} ({mae:.2f})" if cat == "ours" else f"{name} ({mae:.1f})"
        font_color = "#1A365D" if cat == "ours" else ("#744210" if cat == "mid" else ("#9B2C2C" if cat == "light_prior" else "#4A5568"))
        ax.annotate(
            label,
            (x_val, mae),
            xytext=(x_txt, y_txt),
            fontsize=8.0 if cat != "ours" else 8.3,
            fontweight="bold" if cat == "ours" else "normal",
            color=font_color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9) if cat == "ours" else None,
        )


def main() -> None:
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _build_entries()

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.4), dpi=300)
    color_map = {"heavy": "#4A5568", "mid": "#D69E2E", "light_prior": "#E53E3E", "ours": "#2B6CB0"}

    offsets_params = {
        "PET (ICCV'23)": (16.5, 45.2),
        "P2PNet (ICCV'21)": (4.2, 51.0),
        "DM-Count (NeurIPS'20)": (3.6, 58.0),
        "BL (ICCV'19)": (25.5, 62.5),
        "CSRNet (CVPR'18)": (4.8, 71.5),
        "MobileCount (ECCVW'20)": (3.8, 91.5),
        "MCNN (CVPR'16)": (0.16, 110.2),
        "LSANet (Neurocomp'22)": (0.33, 62.8),
        "Lw-Count (TCSVT'22)": (0.042, 62.0),
        "TinyCount (JRTIP'24)": (0.042, 83.0),
        "RMR-104k Pretrained (3-Seed)": (0.30, 72.5),
        "RMR-104k Scratch (seed42)": (0.24, 81.5),
        "RMR-102k No-Neck (Scratch)": (0.042, 51.0),
    }
    offsets_flops = {
        "P2PNet (ICCV'21)": (30.0, 50.5),
        "CSRNet (CVPR'18)": (21.0, 72.5),
        "MobileCount (ECCVW'20)": (7.5, 91.5),
        "MCNN (CVPR'16)": (3.4, 110.2),
        "LSANet (Neurocomp'22)": (17.5, 63.0),
        "TinyCount (JRTIP'24)": (0.12, 82.0),
        "RMR-104k Pretrained (3-Seed)": (1.45, 71.5),
        "RMR-104k Scratch (seed42)": (1.30, 81.5),
        "RMR-102k No-Neck (Scratch)": (0.12, 52.0),
    }

    _plot_panel(ax1, entries, "params", offsets_params, color_map)
    ax1.set_xscale("log")
    ax1.set_xlim(0.038, 120.0)
    ax1.set_ylim(43, 118)
    ax1.set_xlabel("Trainable Parameters (Millions, Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error (MAE, lower is better)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Complexity-Accuracy Trade-off: Parameters vs. MAE (SHA)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.axvspan(0.038, 0.15, alpha=0.12, color="#3182CE", label="Ultra-Lightweight Regime (<= 150k params)")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)

    _plot_panel(ax2, entries, "gflops", offsets_flops, color_map)
    ax2.set_xscale("log")
    ax2.set_xlim(0.10, 220.0)
    ax2.set_ylim(43, 118)
    ax2.set_xlabel("Forward GFLOPs (RMR @ 512x512; Prior @ Author Protocol, Log Scale)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Mean Absolute Error (MAE, lower is better)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Complexity-Accuracy Trade-off: FLOPs vs. MAE (SHA)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.axvspan(0.10, 1.0, alpha=0.12, color="#3182CE", label="Sub-1 GFLOP Regime (@ 512x512)")
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    png_path = output_dir / "fig1_complexity_accuracy_tradeoff.png"
    pdf_path = output_dir / "fig1_complexity_accuracy_tradeoff.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("Figure 1 successfully generated:")
    for e in entries:
        if e["cat"] == "ours":
            print(f"  {e['name']}: params={e['params']:.6f}M, GFLOPs@512={e['gflops']:.4f}, MAE={e['mae']:.2f}")
    print(f"  PNG: {png_path} ({png_path.stat().st_size / 1024:.1f} KB)")
    print(f"  PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
