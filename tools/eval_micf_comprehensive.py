"""Comprehensive Benchmark and Diagnostic Evaluator for MICF.

Implements the official evaluation pipeline and CSV output schema from
Section 50 of MICF_full_method_design.md:

Schema:
- dataset: Dataset name (e.g. ShanghaiTechA)
- seed: Random seed
- variant: Pilot variant ID (e.g. B1, B2, B3, B4, B5, B6)
- params: Total learnable parameter count
- flops: Estimated GFLOPs (via ptflops or analytical proxy)
- rf_proxy: Estimated receptive field / context dilation summary
- mae: Mean Absolute Error on full images
- rmse: Root Mean Squared Error on full images
- nae: Normalized Absolute Error
- prefix_mae: Mean absolute error across all prefix cells C(i, j)
- local_recon_mae: MAE of recovered discrete count map Delta_xy C_hat vs Y
- rectangle_mae_small: MAE on small region counts (area ~ 1/64)
- rectangle_mae_medium: MAE on medium region counts (area ~ 1/16)
- rectangle_mae_large: MAE on large quadrant counts (area ~ 1/4)
- negative_cell_fraction: Fraction of cells where Delta_xy C_hat < 0 (f_-)
- negative_mass_ratio: Negative mass ratio r_-
- corner_delta_count_gap: Count inconsistency |N_corner - N_delta| (E_cons)
- peak_vram_mb: Peak allocated VRAM during evaluation
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.data.common import ntpc_collate_fn
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.data.sha import ShanghaiTechDataset
from hpc.diagnostics.micf_diagnostics import (
    compute_measure_diagnostics,
    evaluate_rectangle_counts,
    query_rectangle_count,
)
from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
)
from hpc.models.micf_lite import MICFLite


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_rf_proxy(model: MICFLite) -> str:
    """Summarize receptive field configuration from neck and context."""
    dilations = getattr(model.neck, "context_dilations", (1, 2, 3))
    has_ctx = getattr(model, "use_integral_context", False)
    ctx_str = "+4DirIntegralContext" if has_ctx else "LocalOnly"
    return f"FPN(dilations={list(dilations)})_{ctx_str}"


def compute_flops_proxy(model: nn.Module, input_res: Tuple[int, int] = (256, 256)) -> float:
    """Estimate GFLOPs for given input resolution."""
    try:
        from ptflops import get_model_complexity_info
        flops, _ = get_model_complexity_info(
            model,
            (3, input_res[0], input_res[1]),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return float(flops) / 1e9
    except Exception:
        # Fallback approximate calculation
        params = sum(p.numel() for p in model.parameters())
        h, w = input_res
        return float(2 * params * (h * w / 256) / 1e9)


@torch.no_grad()
def evaluate_comprehensive(
    model: MICFLite,
    val_loader: DataLoader,
    device: torch.device,
    output_stride: int = 16,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Run full evaluation suite on the validation set."""
    model.eval()

    errors: List[float] = []
    sq_errors: List[float] = []
    naes: List[float] = []
    prefix_maes: List[float] = []
    local_recon_maes: List[float] = []

    rect_smalls: List[float] = []
    rect_mediums: List[float] = []
    rect_larges: List[float] = []

    f_minuses: List[float] = []
    r_minuses: List[float] = []
    viol_mags: List[float] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for idx, batch in enumerate(val_loader):
        if max_samples is not None and idx >= max_samples:
            break

        img = batch["image"].to(device)
        gt_count = float(batch["gt_count"].item())
        gt_pts = [torch.as_tensor(pts, device=device, dtype=torch.float32) for pts in batch["gt_points"]]

        if model.head_type in {"cumulative", "integrated_local"}:
            pred_count, pred_map = model.predict_tiled(img, tile_size=256)
        else:
            pred_count, pred_map = model.predict(img, pad_multiple=64)
        pred_val = float(pred_count.item())

        err = pred_val - gt_count
        errors.append(abs(err))
        sq_errors.append(err * err)
        naes.append(abs(err) / (gt_count + 1e-4))

        # Build ground-truth exact count map and cumulative field at valid resolution
        out_h = math.ceil(H / output_stride)
        out_w = math.ceil(W / output_stride)
        from hpc.losses.micf import points_to_count_map
        y_target = points_to_count_map(
            gt_pts[0],
            out_h=out_h,
            out_w=out_w,
            stride=output_stride,
            device=device,
        ).view(1, 1, out_h, out_w)

        c_target = cell_counts_to_cumulative_field(y_target, orientation="TL")

        # Extract predicted C and recovered Y
        if model.head_type in {"cumulative", "integrated_local"}:
            c_pred = pred_map
            y_pred = discrete_mixed_difference(c_pred)

            # Measure diagnostics
            diag = compute_measure_diagnostics(c_pred)
            f_minuses.append(diag["negative_cell_fraction"])
            r_minuses.append(diag["negative_mass_ratio"])
            viol_mags.append(diag["violation_magnitude"])

            # Prefix MAE across all grid cells
            prefix_err = (c_pred - c_target).abs().mean().item()
            prefix_maes.append(float(prefix_err))

            # Multi-scale rectangle MAE
            rect_res = evaluate_rectangle_counts(
                c_pred[0, 0],
                c_target[0, 0],
                scale_bins=(1 / 64, 1 / 16, 1 / 4),
                num_samples_per_bin=20,
            )
            rect_smalls.append(rect_res.get("rectangle_mae_small", 0.0))
            rect_mediums.append(rect_res.get("rectangle_mae_medium", 0.0))
            rect_larges.append(rect_res.get("rectangle_mae_large", 0.0))
        else:
            # Local head
            y_pred = pred_map
            c_pred = cell_counts_to_cumulative_field(y_pred, orientation="TL")
            f_minuses.append(0.0)
            r_minuses.append(0.0)
            viol_mags.append(0.0)

            prefix_err = (c_pred - c_target).abs().mean().item()
            prefix_maes.append(float(prefix_err))

            rect_res = evaluate_rectangle_counts(
                c_pred[0, 0],
                c_target[0, 0],
                scale_bins=(1 / 64, 1 / 16, 1 / 4),
                num_samples_per_bin=20,
            )
            rect_smalls.append(rect_res.get("rectangle_mae_small", 0.0))
            rect_mediums.append(rect_res.get("rectangle_mae_medium", 0.0))
            rect_larges.append(rect_res.get("rectangle_mae_large", 0.0))

        # Local reconstruction MAE: |Y_pred - Y_target|
        local_mae = (y_pred - y_target).abs().mean().item()
        local_recon_maes.append(float(local_mae))

    peak_vram_mb = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "nae": float(np.mean(naes)),
        "prefix_mae": float(np.mean(prefix_maes)),
        "local_recon_mae": float(np.mean(local_recon_maes)),
        "rectangle_mae_small": float(np.mean(rect_smalls)),
        "rectangle_mae_medium": float(np.mean(rect_mediums)),
        "rectangle_mae_large": float(np.mean(rect_larges)),
        "negative_cell_fraction": float(np.mean(f_minuses)),
        "negative_mass_ratio": float(np.mean(r_minuses)),
        "violation_magnitude": float(np.mean(viol_mags)),
        "peak_vram_mb": peak_vram_mb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive MICF Benchmark Evaluator")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint best.pt")
    parser.add_argument("--output-csv", type=str, default="./runs/pilot_micf/benchmark_results.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test images to evaluate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg.get("experiment", {})
    variant_id = exp_cfg.get("model_id", "MICF")
    dataset_name = cfg.get("dataset", {}).get("name", "ShanghaiTechA")
    seed = exp_cfg.get("seed", 42)

    # 1. Dataset
    ds_cfg = cfg["dataset"]
    test_dataset = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="test_data",
        is_train=False,
        image_mean=ds_cfg.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=ds_cfg.get("image_std", [0.5, 0.5, 0.5]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=ntpc_collate_fn,
    )

    # 2. Model
    m_cfg = cfg.get("model", {})
    output_stride = int(m_cfg.get("output_stride", 16))
    model = MICFLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050.e3000_r224_in1k"),
        pretrained=False,
        neck_width=int(m_cfg.get("neck_width", 32)),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_integral_context=bool(m_cfg.get("use_integral_context", False)),
        context_type=str(m_cfg.get("context_type", "directional")),
        head_type=m_cfg.get("head_type", "cumulative"),
        output_stride=output_stride,
    ).to(device)

    # Load weights if available
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        default_ckpt = Path(exp_cfg.get("save_dir", "")) / "best.pt"
        if default_ckpt.exists():
            ckpt_path = str(default_ckpt)

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Warning: evaluating without loaded weights (raw initialization)")

    # 3. Model stats
    params = count_parameters(model)
    flops = compute_flops_proxy(model, input_res=(256, 256))
    rf_proxy = estimate_rf_proxy(model)

    print(f"Evaluating {variant_id} on {dataset_name} (params={params/1e6:.3f}M, GFLOPs={flops:.3f}) ...")

    # 4. Evaluation
    res = evaluate_comprehensive(
        model=model,
        val_loader=test_loader,
        device=device,
        output_stride=output_stride,
        max_samples=args.max_samples,
    )

    row = {
        "dataset": dataset_name,
        "seed": seed,
        "variant": variant_id,
        "params": params,
        "flops": round(flops, 4),
        "rf_proxy": rf_proxy,
        "mae": round(res["mae"], 3),
        "rmse": round(res["rmse"], 3),
        "nae": round(res["nae"], 4),
        "prefix_mae": round(res["prefix_mae"], 4),
        "local_recon_mae": round(res["local_recon_mae"], 4),
        "rectangle_mae_small": round(res["rectangle_mae_small"], 3),
        "rectangle_mae_medium": round(res["rectangle_mae_medium"], 3),
        "rectangle_mae_large": round(res["rectangle_mae_large"], 3),
        "negative_cell_fraction": round(res["negative_cell_fraction"], 4),
        "negative_mass_ratio": round(res["negative_mass_ratio"], 4),
        "violation_magnitude": round(res["violation_magnitude"], 4),
        "peak_vram_mb": round(res["peak_vram_mb"], 1),
    }

    # Print summary table
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY: {variant_id}")
    print("=" * 80)
    for k, v in row.items():
        print(f"  {k:<26}: {v}")
    print("=" * 80)

    # 5. Append / Write to CSV (Section 50 schema)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0

    fields = list(row.keys())
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Results recorded in {out_csv}\n")


if __name__ == "__main__":
    main()
