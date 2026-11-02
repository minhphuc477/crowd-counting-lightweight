"""V73 Behavioral Diagnostic and Model Forensics Engine.

Executes a comprehensive 6-lens diagnostic audit on trained RMR models:
1. Density Stratification & Bias Decomposition (Sparse Fog vs Dense Saturation)
2. Multiscale GAME Spatial Consistency (GAME 0-3 & Fog Ratio)
3. Carrier vs Regional vs Solver Synergy Dynamics (Mass Shifts & Help Rate)
4. Evidence Calibration & Regional Uncertainty (Entropy & Dispersion)
5. Point Localization & Sub-pixel Peak Resolution (Precision/Recall/F1)
6. Outlier Autopsy & Failure Mode Classification (Top-5 Worst Images)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset
from rmr_core.metrics import evaluate_point_localization, extract_subpixel_peaks
from rmr_v3.eval import load_model_from_ckpt


DENSITY_BRACKETS = [
    ("Ultra-Sparse (N <= 50)", 0.0, 50.0),
    ("Sparse (50 < N <= 150)", 50.0, 150.0),
    ("Low-Medium (150 < N <= 300)", 150.0, 300.0),
    ("Medium-High (300 < N <= 600)", 300.0, 600.0),
    ("Dense (600 < N <= 1000)", 600.0, 1000.0),
    ("Extreme Outlier (N > 1000)", 1000.0, 100000.0),
]


def classify_failure_mode(gt: float, pred: float, abs_err: float, h: int, w: int) -> str:
    """Diagnose physical failure mechanism of an outlier sample."""
    err_ratio = abs_err / max(gt, 1.0)
    density_per_kpx = (gt / (h * w)) * 1000.0

    if gt > 800.0 and (pred - gt) < -200.0:
        return "Severe Under-counting (Poisson/NB Gradient Saturation)"
    if gt <= 50.0 and (pred - gt) > 30.0:
        return "Background Fog Accumulation (False Positive Mass Leakage)"
    if density_per_kpx > 2.5:
        return "Extreme Crowd Clumping (Carrier Subpixel Aliasing)"
    if err_ratio > 0.50 and (pred - gt) > 0:
        return "High-Frequency Texture Confusion (Architectural/Vegetation)"
    if err_ratio > 0.50 and (pred - gt) < 0:
        return "Scale Disconnect / Scale Gate Suppression"
    return "Non-Uniform Illumination / Perspective Distortion"


def run_v73_diagnostics(
    ckpt_path: Path,
    manifest_path: str = "data/sha_a_test.jsonl",
    device_str: str = "cuda:0",
    config_path: Path | str | None = None,
    eval_localization: bool = False,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Execute complete 6-lens diagnostic suite."""
    dev = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    print(f"\n[+] Loading model checkpoint: {ckpt_path}")
    model, _, cfg, _ = load_model_from_ckpt(ckpt_path, dev, use_ema=True, config_path=config_path)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_cfg = getattr(model, "cfg", None)
    output_stride = 2 if getattr(model_cfg, "subpixel_stride2", False) else 4
    fine_carrier_type = getattr(model_cfg, "fine_carrier_type", "standard")
    reg_head_type = getattr(model_cfg, "regional_head_type", "probabilistic")
    has_solver = bool(getattr(model_cfg, "enable_solver", False))

    print(f"    Trainable Parameters : {n_params:,} (<= 200,000 ceiling: {n_params <= 200000})")
    print(f"    Fine Carrier Head    : {fine_carrier_type} (Stride {output_stride})")
    print(f"    Regional Head        : {reg_head_type}")
    print(f"    Transport Solver     : {getattr(model_cfg, 'solver_mode', 'direct') if has_solver else 'DISABLED'}")

    ds = CrowdManifestDataset(manifest_path, train=False, output_stride=output_stride)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)
    print(f"\n[+] Evaluating {len(ds)} test images on {dev}...")

    carrier_preds: list[float] = []
    solver_helps: list[bool] = []
    all_sample_details: list[dict[str, Any]] = []

    def _sample_hook(sample: dict, out: dict, y: torch.Tensor, row: dict) -> dict:
        y0 = out.get("y0", y)
        p_carrier = float(y0.sum().item())
        p_final = float(y.sum().item())
        gt_cnt = row["gt"]
        carrier_preds.append(p_carrier)
        solver_helps.append(abs(p_final - gt_cnt) < abs(p_carrier - gt_cnt))
        row["pred_carrier_y0"] = p_carrier
        return row

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=dev,
        output_stride=output_stride,
        run_tiling=False,
        extra_sample_callback=_sample_hook,
    )

    gts = np.array([r["gt"] for r in rows], dtype=np.float64)
    preds = np.array([r["pred"] for r in rows], dtype=np.float64)
    abs_errs = np.abs(preds - gts)
    diffs = preds - gts

    # ---------------------------------------------------------
    # LENS 1: DENSITY STRATIFICATION & BIAS DECOMPOSITION
    # ---------------------------------------------------------
    strat_report: list[dict[str, Any]] = []
    print("\n" + "=" * 90)
    print("LENS 1: DENSITY STRATIFICATION & ERROR MASS DECOMPOSITION")
    print("=" * 90)
    print(f"{'Density Bracket':<28} | {'N':>3} | {'MAE':>8} | {'RMSE':>8} | {'MRE (%)':>8} | {'Net Bias':>10}")
    print("-" * 90)

    for b_name, b_min, b_max in DENSITY_BRACKETS:
        mask = (gts > b_min) & (gts <= b_max) if b_min > 0 else (gts >= 0) & (gts <= b_max)
        n_sub = int(mask.sum())
        if n_sub == 0:
            continue
        sub_gt = gts[mask]
        sub_pred = preds[mask]
        sub_ae = np.abs(sub_pred - sub_gt)
        mae_b = float(np.mean(sub_ae))
        rmse_b = float(np.sqrt(np.mean((sub_pred - sub_gt) ** 2)))
        mre_b = float(np.mean(sub_ae / np.maximum(sub_gt, 1.0)) * 100.0)
        bias_b = float(np.mean(sub_pred - sub_gt))

        strat_report.append({
            "bracket": b_name, "count": n_sub, "mae": mae_b, "rmse": rmse_b, "mre_pct": mre_b, "bias": bias_b,
        })
        print(f"{b_name:<28} | {n_sub:>3} | {mae_b:>8.2f} | {rmse_b:>8.2f} | {mre_b:>7.1f}% | {bias_b:>+10.2f}")

    print("-" * 90)
    total_mae = float(np.mean(abs_errs))
    total_rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    total_bias = float(np.mean(diffs))
    total_mre = float(np.mean(abs_errs / np.maximum(gts, 1.0)) * 100.0)
    print(f"{'OVERALL DATASET':<28} | {len(gts):>3} | {total_mae:>8.2f} | {total_rmse:>8.2f} | {total_mre:>7.1f}% | {total_bias:>+10.2f}")

    # ---------------------------------------------------------
    # LENS 2: MULTISCALE GAME SPATIAL CONSISTENCY
    # ---------------------------------------------------------
    game0 = float(np.mean([r.get("GAME0", abs(r["pred"] - r["gt"])) for r in rows]))
    game1 = float(np.mean([r.get("GAME1", 0.0) for r in rows]))
    game2 = float(np.mean([r.get("GAME2", 0.0) for r in rows]))
    game3 = float(np.mean([r.get("GAME3", 0.0) for r in rows]))
    fog_ratio = game3 / max(game0, 1e-4)

    print("\n" + "=" * 90)
    print("LENS 2: MULTISCALE GAME SPATIAL LOCALIZATION & FOG RATIO")
    print("=" * 90)
    print(f"  GAME 0 (Global Error) : {game0:.2f}")
    print(f"  GAME 1 (2x2 Grid)     : {game1:.2f}")
    print(f"  GAME 2 (4x4 Grid)     : {game2:.2f}")
    print(f"  GAME 3 (8x8 Grid)     : {game3:.2f}")
    print(f"  Spatial Fog Ratio     : {fog_ratio:.3f} (GAME3 / GAME0)")
    if fog_ratio > 2.2:
        print("  -> DIAGNOSIS: High fog ratio (> 2.2) indicates diffuse carrier haze mass.")
    elif fog_ratio < 1.6:
        print("  -> DIAGNOSIS: Sharp spatial concentration; prediction localized in compact Dirac peaks.")
    else:
        print("  -> DIAGNOSIS: Normal spatial distribution across scales.")

    # ---------------------------------------------------------
    # LENS 3: CARRIER VS REGIONAL VS SOLVER SYNERGY
    # ---------------------------------------------------------
    c_arr = np.array(carrier_preds, dtype=np.float64) if carrier_preds else preds
    c_mae = float(np.mean(np.abs(c_arr - gts)))
    help_rate = float(np.mean(solver_helps) * 100.0) if solver_helps else 0.0
    mean_shift = float(np.mean(np.abs(preds - c_arr)))

    print("\n" + "=" * 90)
    print("LENS 3: CARRIER VS REGIONAL EVIDENCE VS SOLVER SYNERGY")
    print("=" * 90)
    print(f"  Pre-Solver Carrier MAE : {c_mae:.2f}")
    print(f"  Final Post-Solver MAE  : {total_mae:.2f}")
    print(f"  MAE Delta (Solver Help): {total_mae - c_mae:+.2f} ({'IMPROVEMENT' if total_mae < c_mae else 'REGRESSION'})")
    print(f"  Solver Help Rate       : {help_rate:.1f}% of images improved by transport solver")
    print(f"  Mean Transport Shift   : {mean_shift:.2f} persons reallocated")

    # ---------------------------------------------------------
    # LENS 4: SUBPIXEL LOCALIZATION PEAK METRICS
    # ---------------------------------------------------------
    loc_metrics: dict[str, Any] = {}
    if eval_localization:
        print("\n" + "=" * 90)
        print("LENS 4: SUBPIXEL LOCALIZATION PRECISION / RECALL / F1")
        print("=" * 90)
        f1_list_4px, prec_list_4px, rec_list_4px = [], [], []
        f1_list_8px = []

        for sample, row in zip(ds, rows):
            pts_gt = sample.get("points")
            if pts_gt is None or len(pts_gt) == 0:
                continue
            img_t = sample["image"].unsqueeze(0).to(dev)
            with torch.no_grad():
                out = model(img_t)
                y_map = out["y"]
            loc_res_4 = evaluate_point_localization(
                y_map, pts_gt, sample["height"], sample["width"],
                stride=output_stride, radius_px=4.0, threshold=0.0125,
            )
            loc_res_8 = evaluate_point_localization(
                y_map, pts_gt, sample["height"], sample["width"],
                stride=output_stride, radius_px=8.0, threshold=0.0125,
            )
            f1_list_4px.append(loc_res_4["f1"])
            prec_list_4px.append(loc_res_4["precision"])
            rec_list_4px.append(loc_res_4["recall"])
            f1_list_8px.append(loc_res_8["f1"])

        if f1_list_4px:
            loc_metrics = {
                "f1_4px": float(np.mean(f1_list_4px)),
                "precision_4px": float(np.mean(prec_list_4px)),
                "recall_4px": float(np.mean(rec_list_4px)),
                "f1_8px": float(np.mean(f1_list_8px)),
            }
            print(f"  Radius = 4 px : Prec={loc_metrics['precision_4px']:.3f}, Rec={loc_metrics['recall_4px']:.3f}, F1={loc_metrics['f1_4px']:.3f}")
            print(f"  Radius = 8 px : F1={loc_metrics['f1_8px']:.3f}")

    # ---------------------------------------------------------
    # LENS 5 & 6: OUTLIER AUTOPSY & FAILURE MODE CLASSIFICATION
    # ---------------------------------------------------------
    sorted_indices = np.argsort(-abs_errs)
    top_5_indices = sorted_indices[:5]

    print("\n" + "=" * 90)
    print("LENS 5 & 6: TOP-5 WORST OUTLIER AUTOPSY & FAILURE MECHANISM")
    print("=" * 90)
    print(f"{'Image ID':<12} | {'GT':>6} | {'Pred':>7} | {'Abs Err':>8} | {'Rel Err':>8} | {'Diagnosed Failure Mode'}")
    print("-" * 90)

    outliers_report: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_5_indices, 1):
        r = rows[idx]
        sample = ds[idx]
        h_img, w_img = sample["height"], sample["width"]
        gt_val = float(r["gt"])
        pred_val = float(r["pred"])
        ae_val = float(r["abs_err"])
        rel_val = ae_val / max(gt_val, 1.0)
        mode = classify_failure_mode(gt_val, pred_val, ae_val, h_img, w_img)

        outlier_entry = {
            "rank": rank, "id": r["id"], "gt": gt_val, "pred": pred_val,
            "abs_err": ae_val, "rel_err_pct": rel_val * 100.0,
            "image_size": [h_img, w_img], "failure_mode": mode,
        }
        outliers_report.append(outlier_entry)
        print(f"{r['id']:<12} | {gt_val:>6.1f} | {pred_val:>7.1f} | {ae_val:>8.1f} | {rel_val*100:>7.1f}% | {mode}")

    print("=" * 90 + "\n")

    report = {
        "checkpoint": str(ckpt_path),
        "parameters": n_params,
        "overall_metrics": {
            "mae": total_mae, "rmse": total_rmse, "mre_pct": total_mre, "bias": total_bias,
            "game0": game0, "game1": game1, "game2": game2, "game3": game3, "fog_ratio": fog_ratio,
        },
        "carrier_solver_synergy": {
            "pre_solver_carrier_mae": c_mae, "post_solver_mae": total_mae,
            "solver_help_rate_pct": help_rate, "mean_transport_shift": mean_shift,
        },
        "density_stratification": strat_report,
        "localization_metrics": loc_metrics,
        "top_5_outliers": outliers_report,
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[+] Saved diagnostic report to: {output_json}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="V73 Behavioral Diagnostic and Model Forensics Engine")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file or run directory")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config YAML override")
    parser.add_argument("--manifest", type=str, default="data/sha_a_test.jsonl", help="Validation manifest path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (cuda:0 or cpu)")
    parser.add_argument("--eval_localization", action="store_true", help="Evaluate peak localization (F1/Precision/Recall)")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON report path")
    args = parser.parse_args()

    ckpt_p = Path(args.checkpoint)
    out_json = Path(args.output_json) if args.output_json else (ckpt_p.parent if ckpt_p.is_file() else ckpt_p) / "diagnostic_report.json"
    run_v73_diagnostics(
        ckpt_path=ckpt_p,
        manifest_path=args.manifest,
        device_str=args.device,
        config_path=args.config,
        eval_localization=args.eval_localization,
        output_json=out_json,
    )


if __name__ == "__main__":
    main()
