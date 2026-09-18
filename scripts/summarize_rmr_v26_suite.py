"""Summarize RMR-v26 benchmark results across all 6 matrix runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RMR-v26 benchmark and ablation results.")
    parser.add_argument("--runs-dir", default="runs/sha_a", help="Directory containing run directories")
    parser.add_argument("--output-md", default="runs/sha_a/rmr_v26_benchmark_summary.md", help="Path to output Markdown")
    parser.add_argument("--output-csv", default="runs/sha_a/rmr_v26_benchmark_summary.csv", help="Path to output CSV")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    canonical_order = [
        ("rmr_v26_canonical", "RMR-v26 Canonical (MPE-v2 + Pure BB-1 [0.5, 1.2]*w + Morozov gamma=0.75 + Curv=0)"),
        ("rmr_v26_ablation_no_elevation", "Ablation 1: No Perspective Elevation (Carrier without MPE-v2)"),
        ("rmr_v26_ablation_no_bb", "Ablation 2: No BB Step Size Damping (Fixed omega=0.20)"),
        ("rmr_v26_ablation_no_morozov", "Ablation 3: No Morozov Regularization (gamma=0.0)"),
        ("rmr_v26_ablation_with_curv01", "Ablation 4: With Curvature Regularization (lambda_curv=0.10)"),
        ("rmr_v26_control_no_solver", "Control: Direct Feedforward (T=0, No SIRT Inverse Solver)"),
    ]

    rows = []
    canonical_mae = None

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
            "delta_mae": "-",
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

                if run_id == "rmr_v26_canonical" and "mae" in data:
                    canonical_mae = float(data["mae"])
                elif canonical_mae is not None and "mae" in data:
                    delta = float(data["mae"]) - canonical_mae
                    row["delta_mae"] = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
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
                    if "rmse" in tdata and tdata["rmse"] is not None:
                        row["tta_rmse"] = f"{tdata['rmse']:.2f}"
                except Exception:
                    pass

        # Check train_log.csv for epochs
        log_file = rd / "train_log.csv"
        if log_file.is_file():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    epochs = [int(r["epoch"]) for r in reader if "epoch" in r and r["epoch"].isdigit()]
                    if epochs:
                        row["epochs"] = str(max(epochs))
            except Exception:
                pass

        rows.append(row)

    # Print markdown table
    print("\n" + "=" * 110)
    print("  RMR-v26 BENCHMARK & ABLATION SUITE SUMMARY (ShanghaiTech Part A)")
    print("=" * 110)
    header = f"| {'Run ID':<30} | {'MAE':<6} | {'RMSE':<6} | {'TTA MAE':<7} | {'Delta':<7} | {'Sparse':<6} | {'Mod':<6} | {'Dense':<6} | {'Epochs':<6} |"
    sep = f"|{'-'*32}|{'-'*8}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|"
    print(header)
    print(sep)
    for r in rows:
        print(f"| {r['run_id']:<30} | {r['mae']:<6} | {r['rmse']:<6} | {r['tta_mae']:<7} | {r['delta_mae']:<7} | {r['sparse_mae']:<6} | {r['moderate_mae']:<6} | {r['dense_mae']:<6} | {r['epochs']:<6} |")
    print("=" * 110)

    # Save Markdown file
    md_lines = [
        "# RMR-v26 Benchmark & Ablation Suite Summary\n",
        f"**Target Evaluation**: ShanghaiTech Part A Test Set (182 canonical images)\n\n",
        f"| Run ID | Description | MAE | RMSE | TTA MAE | Delta MAE | Sparse | Moderate | Dense | Epochs |",
        f"|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['run_id']}` | {r['description']} | **{r['mae']}** | {r['rmse']} | {r['tta_mae']} | {r['delta_mae']} | {r['sparse_mae']} | {r['moderate_mae']} | {r['dense_mae']} | {r['epochs']} |"
        )
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved summary to {md_path}")


if __name__ == "__main__":
    main()
