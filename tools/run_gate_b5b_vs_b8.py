"""Automated Scientific Gate Runner: B5b vs B8 (FH-CMICF K=4) on ShanghaiTech Part A.

Strictly matched carrier (MobileNetV4-0.50), optimizer (AdamW, lr=1e-4), schedule,
loss normalization, and evaluation protocol.

Post-training pipeline:
1. Evaluates MAE_crop, MAE_full_direct, MAE_full_tiled, RMSE.
2. Computes violation_rate, violation_magnitude, negative_mass_ratio.
3. Generates training/validation loss curves.
4. Generates 2D spatial error map of C to visualize horizon accumulation.
5. Formats the gate decision report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpc.data.common import ntpc_collate_fn
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.data.sha import ShanghaiTechDataset
from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
)
from hpc.models.micf_lite import MICFLite


def run_command(cmd: List[str], desc: str) -> None:
    print(f"\n>>> [RUNNING] {desc}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def plot_loss_curves(
    b5b_history_path: Path,
    b8_history_path: Path,
    save_path: Path,
) -> None:
    """Plot multi-panel training and validation trajectories."""
    with open(b5b_history_path, "r", encoding="utf-8") as f:
        b5b_hist = json.load(f)
    with open(b8_history_path, "r", encoding="utf-8") as f:
        b8_hist = json.load(f)

    b5b_epochs = [entry["epoch"] for entry in b5b_hist]
    b8_epochs = [entry["epoch"] for entry in b8_hist]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(b5b_epochs, [e["loss"] for e in b5b_hist], label="B5b (Global Extent-Aware)", color="#1f77b4", lw=2)
    ax.plot(b8_epochs, [e["loss"] for e in b8_hist], label="B8 (FH-CMICF K=4)", color="#ff7f0e", lw=2)
    ax.set_title("Training Loss (Sample-Normalized SmoothL1 + Validity)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 2. Validation MAE (Crop)
    ax = axes[0, 1]
    b5b_eval_e = [e["epoch"] for e in b5b_hist if "mae_crop" in e]
    b8_eval_e = [e["epoch"] for e in b8_hist if "mae_crop" in e]
    ax.plot(b5b_eval_e, [e["mae_crop"] for e in b5b_hist if "mae_crop" in e], label="B5b (Crop MAE)", color="#1f77b4", marker="o", lw=1.8, ms=4)
    ax.plot(b8_eval_e, [e["mae_crop"] for e in b8_hist if "mae_crop" in e], label="B8 (Crop MAE)", color="#ff7f0e", marker="s", lw=1.8, ms=4)
    ax.set_title("Validation MAE (Regime A: Fixed 256x256 Crop)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 3. Validation MAE (Full Image)
    ax = axes[1, 0]
    ax.plot(b5b_eval_e, [e["mae_full"] for e in b5b_hist if "mae_full" in e], label="B5b (Full MAE)", color="#1f77b4", marker="o", lw=1.8, ms=4)
    ax.plot(b8_eval_e, [e["mae_full"] for e in b8_hist if "mae_full" in e], label="B8 (Full MAE)", color="#ff7f0e", marker="s", lw=1.8, ms=4)
    ax.set_title("Validation MAE (Regime B: Full Image Direct)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 4. Measure Violation Rate (%) & Magnitude
    ax = axes[1, 1]
    ax.plot(b5b_eval_e, [e.get("violation_rate", 0) * 100 for e in b5b_hist if "mae_crop" in e], label="B5b Viol Rate (%)", color="#1f77b4", linestyle="--")
    ax.plot(b8_eval_e, [e.get("violation_rate", 0) * 100 for e in b8_hist if "mae_crop" in e], label="B8 Viol Rate (%)", color="#ff7f0e", linestyle="--")
    ax.set_title("Measure Violation Rate (% of Cells with Delta C < 0)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Violation Rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[SAVED] Loss curves comparison to {save_path}")


def compute_and_plot_spatial_error_map(
    b5b_ckpt: Path,
    b8_ckpt: Path,
    cfg_b5b: dict,
    cfg_b8: dict,
    device: torch.device,
    save_path: Path,
    canonical_size: int = 32,
    max_eval_samples: int | None = None,
) -> None:
    """Compute and plot average spatial error heatmaps for C across the test set."""
    print("\n>>> Computing 2D Spatial Error Maps of Cumulative Field C...")

    # Load B5b
    m_b5b = cfg_b5b["model"]
    model_b5b = MICFLite(
        backbone_name=m_b5b.get("backbone", "mobilenetv4_conv_small_050.e3000_r224_in1k"),
        pretrained=False,
        neck_width=int(m_b5b.get("neck_width", 32)),
        context_dilations=tuple(m_b5b.get("context_dilations", [1, 2, 3])),
        use_integral_context=bool(m_b5b.get("use_integral_context", True)),
        context_type=str(m_b5b.get("context_type", "directional")),
        head_type="cumulative",
        output_stride=int(m_b5b.get("output_stride", 16)),
        extent_aware=True,
        finite_horizon=None,
    ).to(device)
    sd_b5b = torch.load(b5b_ckpt, map_location=device, weights_only=False)
    model_b5b.load_state_dict(sd_b5b["state_dict"] if "state_dict" in sd_b5b else sd_b5b)
    model_b5b.eval()

    # Load B8
    m_b8 = cfg_b8["model"]
    model_b8 = MICFLite(
        backbone_name=m_b8.get("backbone", "mobilenetv4_conv_small_050.e3000_r224_in1k"),
        pretrained=False,
        neck_width=int(m_b8.get("neck_width", 32)),
        context_dilations=tuple(m_b8.get("context_dilations", [1, 2, 3])),
        use_integral_context=bool(m_b8.get("use_integral_context", True)),
        context_type=str(m_b8.get("context_type", "directional")),
        head_type="cumulative",
        output_stride=int(m_b8.get("output_stride", 16)),
        extent_aware=True,
        finite_horizon=int(m_b8.get("finite_horizon", 4)),
    ).to(device)
    sd_b8 = torch.load(b8_ckpt, map_location=device, weights_only=False)
    model_b8.load_state_dict(sd_b8["state_dict"] if "state_dict" in sd_b8 else sd_b8)
    model_b8.eval()

    # Dataset
    ds_cfg = cfg_b5b["dataset"]
    test_dataset = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="test_data",
        is_train=False,
        image_mean=ds_cfg.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=ds_cfg.get("image_std", [0.5, 0.5, 0.5]),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=ntpc_collate_fn)

    errors_b5b = []
    errors_b8 = []

    s = model_b5b.output_stride

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if max_eval_samples is not None and idx >= max_eval_samples:
                break

            img = batch["image"].to(device)
            gt_pts = [torch.as_tensor(pts, device=device, dtype=torch.float32) for pts in batch["gt_points"]]

            # Ground truth C
            _, _, H, W = img.shape
            out_h = math.ceil(H / s)
            out_w = math.ceil(W / s)
            from hpc.losses.micf import points_to_count_map
            gt_y = points_to_count_map(gt_pts[0], out_h, out_w, stride=s, device=device).unsqueeze(0).unsqueeze(0)
            gt_c = cell_counts_to_cumulative_field(gt_y)  # [1, 1, Ho, Wo]

            # Predictions
            _, pred_c_b5b = model_b5b.predict(img, pad_multiple=64)
            _, pred_c_b8 = model_b8.predict(img, pad_multiple=64)

            # Crop prediction to match valid gt_c dimensions
            Ho, Wo = gt_c.shape[-2:]
            pred_c_b5b = pred_c_b5b[..., :Ho, :Wo]
            pred_c_b8 = pred_c_b8[..., :Ho, :Wo]

            # Compute absolute cumulative error maps |C_hat - C_gt|
            err_b5b = torch.abs(pred_c_b5b - gt_c)
            err_b8 = torch.abs(pred_c_b8 - gt_c)

            # Interpolate to canonical grid
            err_b5b_res = F.interpolate(err_b5b, size=(canonical_size, canonical_size), mode="bilinear", align_corners=False)
            err_b8_res = F.interpolate(err_b8, size=(canonical_size, canonical_size), mode="bilinear", align_corners=False)

            errors_b5b.append(err_b5b_res.squeeze().cpu().numpy())
            errors_b8.append(err_b8_res.squeeze().cpu().numpy())

    mean_err_b5b = np.mean(np.stack(errors_b5b), axis=0)
    mean_err_b8 = np.mean(np.stack(errors_b8), axis=0)
    diff_err = mean_err_b8 - mean_err_b5b  # Negative where B8 has lower error

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmax = max(float(np.max(mean_err_b5b)), float(np.max(mean_err_b8)))

    im0 = axes[0].imshow(mean_err_b5b, cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title("B5b: Mean Cumulative Error |C_hat - C_gt|\n(Global Dependency Horizon)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Normalized Width (0 -> W)")
    axes[0].set_ylabel("Normalized Height (0 -> H)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mean_err_b8, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title("B8: Mean Cumulative Error |C_hat - C_gt|\n(Finite-Horizon K=4 Factorization)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Normalized Width (0 -> W)")
    axes[1].set_ylabel("Normalized Height (0 -> H)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Difference map (coolwarm: Blue = B8 is better, Red = B5b is better)
    diff_max = max(abs(float(np.min(diff_err))), abs(float(np.max(diff_err))), 1e-5)
    im2 = axes[2].imshow(diff_err, cmap="coolwarm", vmin=-diff_max, vmax=diff_max)
    axes[2].set_title("Difference Map: Error(B8) - Error(B5b)\n(Blue: B8 Lower Error | Red: B5b Lower Error)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Normalized Width (0 -> W)")
    axes[2].set_ylabel("Normalized Height (0 -> H)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[SAVED] Spatial error map of C to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scientific Gate Runner: B5b vs B8")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs for pilot")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run evaluation/analysis only")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit test samples for fast verification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg_b5b_path = _REPO_ROOT / "configs" / "pilot_micf" / "b5b.yaml"
    cfg_b8_path = _REPO_ROOT / "configs" / "pilot_micf" / "b8.yaml"

    with open(cfg_b5b_path, "r", encoding="utf-8") as f:
        cfg_b5b = yaml.safe_load(f)
    with open(cfg_b8_path, "r", encoding="utf-8") as f:
        cfg_b8 = yaml.safe_load(f)

    python_exe = sys.executable

    # 1. Train B5b and B8
    if not args.skip_train:
        cmd_b5b = [python_exe, "tools/train_micf_pilot.py", "--config", str(cfg_b5b_path)]
        if args.epochs:
            cmd_b5b.extend(["--epochs", str(args.epochs)])
        run_command(cmd_b5b, "Training B5b (Global Extent-Aware Baseline)")

        cmd_b8 = [python_exe, "tools/train_micf_pilot.py", "--config", str(cfg_b8_path)]
        if args.epochs:
            cmd_b8.extend(["--epochs", str(args.epochs)])
        run_command(cmd_b8, "Training B8 (FH-CMICF K=4)")
    else:
        print("\n>>> [--skip-train] Skipping training stage.")

    # 2. Comprehensive Evaluation
    cmd_eval_b5b = [python_exe, "tools/eval_micf_comprehensive.py", "--config", str(cfg_b5b_path)]
    if args.max_eval_samples:
        cmd_eval_b5b.extend(["--max-samples", str(args.max_eval_samples)])
    run_command(cmd_eval_b5b, "Comprehensive Evaluation of B5b")

    cmd_eval_b8 = [python_exe, "tools/eval_micf_comprehensive.py", "--config", str(cfg_b8_path)]
    if args.max_eval_samples:
        cmd_eval_b8.extend(["--max-samples", str(args.max_eval_samples)])
    run_command(cmd_eval_b8, "Comprehensive Evaluation of B8")

    # 3. Post-Training Diagnostics & Visualization
    b5b_save_dir = _REPO_ROOT / "runs" / "pilot_micf" / "b5b"
    b8_save_dir = _REPO_ROOT / "runs" / "pilot_micf" / "b8_k4"

    curves_save_path = _REPO_ROOT / "runs" / "pilot_micf" / "comparison_b5b_vs_b8_curves.png"
    plot_loss_curves(
        b5b_save_dir / "history.json",
        b8_save_dir / "history.json",
        curves_save_path,
    )

    spatial_save_path = _REPO_ROOT / "runs" / "pilot_micf" / "spatial_error_map_C_b5b_vs_b8.png"
    compute_and_plot_spatial_error_map(
        b5b_save_dir / "best.pt",
        b8_save_dir / "best.pt",
        cfg_b5b,
        cfg_b8,
        device,
        spatial_save_path,
        canonical_size=32,
        max_eval_samples=args.max_eval_samples,
    )

    # 4. Generate Final Gate Report
    with open(b5b_save_dir / "history.json", "r", encoding="utf-8") as f:
        b5b_hist = json.load(f)
    with open(b8_save_dir / "history.json", "r", encoding="utf-8") as f:
        b8_hist = json.load(f)

    b5b_best_entry = min((e for e in b5b_hist if "mae_crop" in e), key=lambda x: x["mae_crop"])
    b8_best_entry = min((e for e in b8_hist if "mae_crop" in e), key=lambda x: x["mae_crop"])

    delta_crop = b8_best_entry["mae_crop"] - b5b_best_entry["mae_crop"]
    delta_full = b8_best_entry["mae_full"] - b5b_best_entry["mae_full"]

    if delta_crop < -5.0:
        verdict = "B8 >> B5b (FH factorization significantly superior -> proceed to seed expansion & K-sweep)"
    elif abs(delta_crop) <= 5.0:
        verdict = "B8 ~ B5b (Comparable performance -> investigate spatial diagnostics before further sweeps)"
    else:
        verdict = "B8 << B5b (Global context superior or FH bottleneck -> inspect local boundary reconstruction)"

    report = f"""# Gate Decision Report: B5b vs B8 (FH-CMICF K=4)

**Dataset**: ShanghaiTech Part A (`seed=42`)
**Carrier**: MobileNetV4-Conv-Small-0.50 (99,697 parameters)
**Strict Control**: Confounder-free architecture (identical parameter count, conv RF scope, and GroupNorm spatial scope).

## Metric Comparison

| Metric | B5b (Global Extent-Aware) | B8 (FH-CMICF K=4) | Absolute Delta (B8 - B5b) |
| :--- | :---: | :---: | :---: |
| **MAE_crop** (Best Validation) | {b5b_best_entry['mae_crop']:.2f} | {b8_best_entry['mae_crop']:.2f} | {delta_crop:+.2f} |
| **MAE_full_direct** | {b5b_best_entry['mae_full']:.2f} | {b8_best_entry['mae_full']:.2f} | {delta_full:+.2f} |
| **RMSE** | {b5b_best_entry.get('rmse', 0):.2f} | {b8_best_entry.get('rmse', 0):.2f} | {b8_best_entry.get('rmse', 0) - b5b_best_entry.get('rmse', 0):+.2f} |
| **Violation Rate** | {b5b_best_entry.get('violation_rate', 0)*100:.2f}% | {b8_best_entry.get('violation_rate', 0)*100:.2f}% | {(b8_best_entry.get('violation_rate', 0) - b5b_best_entry.get('violation_rate', 0))*100:+.2f}% |
| **Violation Magnitude** | {b5b_best_entry.get('violation_magnitude', 0):.4f} | {b8_best_entry.get('violation_magnitude', 0):.4f} | {b8_best_entry.get('violation_magnitude', 0) - b5b_best_entry.get('violation_magnitude', 0):+.4f} |
| **Negative Mass Ratio** | {b5b_best_entry.get('neg_mass_ratio', 0)*100:.2f}% | {b8_best_entry.get('neg_mass_ratio', 0)*100:.2f}% | {(b8_best_entry.get('neg_mass_ratio', 0) - b5b_best_entry.get('neg_mass_ratio', 0))*100:+.2f}% |

## Scientific Gate Verdict

$$\\boxed{{{verdict}}}$$

- Trajectory curves saved to: `runs/pilot_micf/comparison_b5b_vs_b8_curves.png`
- 2D spatial error map saved to: `runs/pilot_micf/spatial_error_map_C_b5b_vs_b8.png`
"""

    report_path = _REPO_ROOT / "runs" / "pilot_micf" / "gate_decision_b5b_vs_b8.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    print(f"[COMPLETED] Gate runner execution finished successfully. Report: {report_path}")


if __name__ == "__main__":
    main()
