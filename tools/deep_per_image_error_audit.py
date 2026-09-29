"""Deep Per-Image Error and Fine-Grained Density Tier Audit.

Audits crowd counting models across fine-grained density tiers, calculates error mass shares,
and identifies the worst outlier images to reveal physical bottlenecks (density saturation vs background haze).
Supports flexible CLI arguments for arbitrary checkpoints.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset
from rmr_v3.eval import load_model_from_ckpt


def get_model_preds(ckpt_path: Path, manifest: str = "data/sha_a_test.jsonl", device: str = "cuda:0") -> tuple[list[dict], dict[str, Any]]:
    """Load model and run full tiled evaluation on manifest."""
    dev = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    model, _, cfg, _ = load_model_from_ckpt(ckpt_path, dev, use_ema=True)
    dataset = CrowdManifestDataset(manifest, train=False, output_stride=4)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)
    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=dev,
        output_stride=4,
        run_tiling=True,
        tile_size=512,
        practical_halo=64,
        use_tta=False,
    )
    return rows, summary


def run_audit(checkpoints: dict[str, Path], manifest: str = "data/sha_a_test.jsonl", device: str = "cuda:0") -> None:
    """Execute comparative per-image error audit across available models."""
    results: dict[str, list[dict]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for name, path in checkpoints.items():
        if path.is_file():
            print(f"Evaluating {name} from {path}...")
            rows, summ = get_model_preds(path, manifest=manifest, device=device)
            results[name] = rows
            summaries[name] = summ
        else:
            print(f"Checkpoint not found (skipping): {path}")

    if not results:
        print("Error: No checkpoints could be found or loaded.")
        sys.exit(1)

    # Use first available model rows for ground-truth and image IDs
    first_model_name = next(iter(results.keys()))
    base_rows = results[first_model_name]
    gts = np.array([r["gt"] for r in base_rows], dtype=np.float64)
    img_ids = [r["id"] for r in base_rows]
    total_samples = len(gts)

    bins = [
        ("Ultra-Sparse (N <= 50)", gts <= 50),
        ("Sparse (50 < N <= 150)", (gts > 50) & (gts <= 150)),
        ("Low-Medium (150 < N <= 300)", (gts > 150) & (gts <= 300)),
        ("Medium-High (300 < N <= 600)", (gts > 300) & (gts <= 600)),
        ("Dense (600 < N <= 1000)", (gts > 600) & (gts <= 1000)),
        ("Ultra-Dense (N > 1000)", gts > 1000),
    ]

    print("\n" + "=" * 110)
    print(f"FINE-GRAINED DENSITY TIER BREAKDOWN ACROSS EVALUATED MODELS ({total_samples} test images)")
    print("=" * 110)

    # Build dynamic header
    header_parts = [f"{'Density Bin':<28} | {'N':>3}"]
    for m in results.keys():
        header_parts.append(f"| {m[:12]:>12} (Bias)")
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))

    for bin_name, mask in bins:
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        line_parts = [f"{bin_name:<28} | {n_bin:>3}"]
        for m in results.keys():
            p = np.array([r["pred"] for r in results[m]], dtype=np.float64)
            err = np.abs(p[mask] - gts[mask])
            bias = p[mask] - gts[mask]
            line_parts.append(f"| {np.mean(err):>8.2f} ({np.mean(bias):>+6.1f})")
        print(" ".join(line_parts))

    # Overall Summary
    print("-" * len(header))
    summary_parts = [f"{'OVERALL DATASET':<28} | {total_samples:>3}"]
    for m in results.keys():
        p = np.array([r["pred"] for r in results[m]], dtype=np.float64)
        err = np.abs(p - gts)
        bias = p - gts
        summary_parts.append(f"| {np.mean(err):>8.2f} ({np.mean(bias):>+6.1f})")
    print(" ".join(summary_parts))

    # Error mass share by density bin for first model
    print("\n" + "=" * 110)
    print(f"SHARE OF TOTAL ERROR MASS BY DENSITY BIN (Reference: {first_model_name})")
    print("=" * 110)
    p_ref = np.array([r["pred"] for r in results[first_model_name]], dtype=np.float64)
    all_errors_ref = np.abs(p_ref - gts)
    total_err_ref = float(np.sum(all_errors_ref))

    for bin_name, mask in bins:
        n_bin = int(mask.sum())
        bin_err_sum = float(np.sum(all_errors_ref[mask]))
        pct = (bin_err_sum / max(total_err_ref, 1e-6)) * 100
        print(f"  {bin_name:<28} (n={n_bin:>2}): sum_AE = {bin_err_sum:>8.1f} | {pct:>5.1f}% of total dataset error!")

    # Top 15 Worst Images in Reference Model
    worst_indices = np.argsort(all_errors_ref)[::-1][:15]
    print("\n" + "=" * 110)
    print(f"TOP 15 WORST PREDICTION OUTLIERS IN {first_model_name} (Ranked by Absolute Error)")
    print("=" * 110)
    outlier_header = f"{'Rank':<5} | {'Image ID':<12} | {'GT':>6}"
    for m in results.keys():
        outlier_header += f" | {m[:10]:>10} (Err)"
    print(outlier_header)
    print("-" * len(outlier_header))

    for rank, idx in enumerate(worst_indices, 1):
        gt = gts[idx]
        img_id = img_ids[idx]
        line = f"{rank:<5} | {img_id:<12} | {gt:>6.1f}"
        for m in results.keys():
            pred_val = results[m][idx]["pred"]
            line += f" | {pred_val:>10.1f} ({abs(pred_val - gt):>6.1f})"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Per-Image Error & Fine-Grained Density Tier Audit.")
    parser.add_argument("--manifest", type=str, default="data/sha_a_test.jsonl", help="Evaluation manifest path.")
    parser.add_argument("--checkpoints", nargs="*", type=str, default=None, help="Paths to checkpoints (NAME=PATH or PATH).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device (e.g. cuda:0 or cpu).")
    args = parser.parse_args()

    default_ckpts = {
        "m04_fractional": Path("runs/sha_a/matrix/m04_iso_fractional_loss/best_val_mae.pt"),
        "m08_triad": Path("runs/sha_a/matrix/m08_step3_triad_cures/best_val_mae.pt"),
        "champion_synth": Path("runs/sha_a/sub60_champion_synthesis/best_val_mae.pt"),
        "triad_calib": Path("runs/sha_a/sub60_triad_calibrated/best_val_mae.pt"),
    }

    ckpts: dict[str, Path] = {}
    if args.checkpoints:
        for item in args.checkpoints:
            if "=" in item:
                k, v = item.split("=", 1)
                ckpts[k] = Path(v)
            else:
                p = Path(item)
                ckpts[p.parent.name] = p
    else:
        ckpts = default_ckpts

    run_audit(ckpts, manifest=args.manifest, device=args.device)


if __name__ == "__main__":
    main()
