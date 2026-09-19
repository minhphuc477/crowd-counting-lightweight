"""Summarize RMR-v27 benchmark results across all 11 matrix runs and compare with v19."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RMR-v27 benchmark and ablation results.")
    parser.add_argument("--runs-dir", default="runs/sha_a", help="Directory containing run directories")
    parser.add_argument("--output-md", default="runs/sha_a/rmr_v27_benchmark_summary.md", help="Path to output Markdown")
    parser.add_argument("--output-csv", default="runs/sha_a/rmr_v27_benchmark_summary.csv", help="Path to output CSV")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    # Benchmark Gold Anchor: v19 Canonical Isotropic
    V19_DIRECT_MAE = 72.84
    V19_TTA_MAE = 72.61

    canonical_order = [
        ("rmr_v27_canonical_restored", "RMR-v27 Canonical Restored (v19 Winning Baseline: w=1.0, BB [0.2, 2.0], Curv=0.50, ScaleAlign=0.05, HardBG=0.15)"),
        ("rmr_v27_sota_push", "RMR-v27 SOTA Push (Cyclic BB=2 + Morozov gamma=0.50 + Curv=0.35 -> Sub-70 MAE Target)"),
        ("rmr_v27_ablation_cyclic_bb", "Ablation 1: Cyclic BB-1 (Cycle Length = 2)"),
        ("rmr_v27_ablation_no_bb", "Ablation 2: Fixed Step Size (No BB, w=1.0)"),
        ("rmr_v27_ablation_morozov05", "Ablation 3: Tighter Morozov Deadband (gamma=0.50)"),
        ("rmr_v27_ablation_no_morozov", "Ablation 4: No Morozov Regularization (gamma=0.0)"),
        ("rmr_v27_ablation_curv035", "Ablation 5: Calibrated Anscombe Curvature (lambda_curv=0.35)"),
        ("rmr_v27_ablation_no_curvature", "Ablation 6: No Curvature Loss (Anscombe Variance-Stabilization Isolation, lambda_curv=0.0)"),
        ("rmr_v27_ablation_no_scale_align", "Ablation 7: No Physical Scale Alignment (lambda_scale_align=0.0)"),
        ("rmr_v27_ablation_no_hard_bg", "Ablation 8: No Hard Background Mining (lambda_hard_bg=0.0)"),
        ("rmr_v27_control_no_solver", "Control: Direct Feedforward (T=0, No SIRT Inverse Solver)"),
    ]

    rows = []

    for run_id, desc in canonical_order:
        rd = runs_dir / run_id
        row = {
            "run_id": run_id,
            "description": desc,
            "exists": rd.exists(),
            "epochs": "-",
            "mae": "-",
            "rmse": "-",
            "bias": "-",
            "sparse_mae": "-",
            "moderate_mae": "-",
            "dense_mae": "-",
            "game0": "-",
            "game1": "-",
            "tta_mae": "-",
            "tta_rmse": "-",
            "delta_v19_direct": "-",
            "delta_v19_tta": "-",
        }

        if not rd.exists():
            rows.append(row)
            continue

        # Extract validation summary
        val_summary = rd / "eval_val" / "summary.json"
        if val_summary.is_file():
            try:
                data = json.loads(val_summary.read_text(encoding="utf-8"))
                for k in ["mae", "rmse", "bias"]:
                    if k in data and data[k] is not None:
                        row[k] = f"{data[k]:.2f}"
                if "mae_sparse" in data:
                    row["sparse_mae"] = f"{data['mae_sparse']:.2f}"
                if "mae_moderate" in data:
                    row["moderate_mae"] = f"{data['mae_moderate']:.2f}"
                if "mae_dense" in data:
                    row["dense_mae"] = f"{data['mae_dense']:.2f}"
                if "game0" in data:
                    row["game0"] = f"{data['game0']:.2f}"
                if "game1" in data:
                    row["game1"] = f"{data['game1']:.2f}"

                if "mae" in data:
                    delta = float(data["mae"]) - V19_DIRECT_MAE
                    row["delta_v19_direct"] = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            except Exception:
                pass

        # Extract TTA summary if available
        tta_dirs = [d for d in rd.iterdir() if d.is_dir() and "tta" in d.name] if rd.exists() else []
        for td in tta_dirs:
            tsum = td / "summary.json"
            if tsum.is_file():
                try:
                    tdata = json.loads(tsum.read_text(encoding="utf-8"))
                    if "mae" in tdata and tdata["mae"] is not None:
                        row["tta_mae"] = f"{tdata['mae']:.2f}"
                        delta_t = float(tdata["mae"]) - V19_TTA_MAE
                        row["delta_v19_tta"] = f"+{delta_t:.2f}" if delta_t > 0 else f"{delta_t:.2f}"
                    if "rmse" in tdata and tdata["rmse"] is not None:
                        row["tta_rmse"] = f"{tdata['rmse']:.2f}"
                except Exception:
                    pass

        # Check train_log.csv for epochs
        log_file = rd / "train_log.csv"
        if log_file.is_file():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].split(",")
                        row["epochs"] = str(int(float(last_line[0])))
            except Exception:
                pass

        rows.append(row)

    # Print Table to stdout
    print("\n" + "=" * 130)
    print("  RMR-v27 BENCHMARK & ABLATION SUITE SUMMARY (ShanghaiTech Part A)")
    print(f"  All-time Benchmark Reference: RMR-v19 Canonical Isotropic (Direct MAE: {V19_DIRECT_MAE}, TTA MAE: {V19_TTA_MAE})")
    print("=" * 130)
    header = f"{'Run ID':<32} | {'Direct MAE':<10} | {'TTA MAE':<9} | {'Delta v19':<10} | {'RMSE':<8} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7} | {'Epochs':<6}"
    print(header)
    print("-" * 130)

    for r in rows:
        line = f"{r['run_id']:<32} | {r['mae']:<10} | {r['tta_mae']:<9} | {r['delta_v19_tta']:<10} | {r['rmse']:<8} | {r['sparse_mae']:<7} | {r['moderate_mae']:<7} | {r['dense_mae']:<7} | {r['epochs']:<6}"
        print(line)
    print("=" * 130 + "\n")

    # Generate Markdown Output
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# RMR-v27 Benchmark & Comprehensive Ablation Suite Summary\n\n")
        f.write("**Target Evaluation**: ShanghaiTech Part A Test Set (182 canonical images)\n")
        f.write(f"**Gold Anchor**: RMR-v19 Canonical Isotropic (Direct MAE: {V19_DIRECT_MAE}, TTA MAE: {V19_TTA_MAE})\n\n\n")
        f.write("| Run ID | Description | MAE | RMSE | TTA MAE | Delta v19 TTA | Sparse | Moderate | Dense | Epochs |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| `{r['run_id']}` | {r['description']} | **{r['mae']}** | {r['rmse']} | {r['tta_mae']} | {r['delta_v19_tta']} | {r['sparse_mae']} | {r['moderate_mae']} | {r['dense_mae']} | {r['epochs']} |\n")

    # Generate CSV Output
    out_csv = Path(args.output_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to:\n  - Markdown: {out_md}\n  - CSV: {out_csv}")


if __name__ == "__main__":
    main()
