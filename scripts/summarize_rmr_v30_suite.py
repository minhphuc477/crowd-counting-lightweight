"""Summarize RMR-v30 benchmark results across all 6 hypothesis ladder runs and compare with v19."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RMR-v30 benchmark and hypothesis ladder results.")
    parser.add_argument("--runs-dir", default="runs/sha_a", help="Directory containing run directories")
    parser.add_argument("--output-md", default="runs/sha_a/rmr_v30_benchmark_summary.md", help="Path to output Markdown")
    parser.add_argument("--output-csv", default="runs/sha_a/rmr_v30_benchmark_summary.csv", help="Path to output CSV")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    # Benchmark Gold Anchor: v19 Canonical Isotropic
    V19_DIRECT_MAE = 72.84
    V19_TTA_MAE = 72.61

    canonical_order = [
        ("rmr_v30_step0_v19_anchor", "Step 0: Golden Anchor (v19 Bitwise Parity: Stride 4, RN Adjoint, BB [0.2, 2.0], Firm Tau=0.015, T=6)"),
        ("rmr_v30_h1_anscombe_sirt", "Hypothesis 1: Anscombe VST SIRT (Stride 4, Homoscedastic Discrepancy O(1) Updates, T=6)"),
        ("rmr_v30_h2_dual_lattice_dcsr", "Hypothesis 2: Dual-Lattice DCSR (Stride 2 Fine Lattice + Stride 4 Carrier Anchor, Adaptive Tau, T=6)"),
        ("rmr_v30_h3_anscombe_dual_lattice", "Hypothesis 3: Primary Sub-60 (Composite H1 Anscombe + H2 Dual-Lattice DCSR, 104,540 params)"),
        ("rmr_v30_h4_deep_sirt_t8", "Hypothesis 4: Deep Unrolled Inversion (T=8 Iterations on Composite H3 Architecture)"),
        ("rmr_v30_control_no_solver", "Control: Direct Feedforward Baseline (T=0, Solver Disabled, Y0 Mode)"),
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
                        delta_tta = float(tdata["mae"]) - V19_TTA_MAE
                        row["delta_v19_tta"] = f"+{delta_tta:.2f}" if delta_tta > 0 else f"{delta_tta:.2f}"
                    if "rmse" in tdata and tdata["rmse"] is not None:
                        row["tta_rmse"] = f"{tdata['rmse']:.2f}"
                except Exception:
                    pass

        # Extract training log epochs
        train_log = rd / "train_log.csv"
        if train_log.is_file():
            try:
                with open(train_log, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    epochs = [int(r["epoch"]) for r in reader if "epoch" in r and r["epoch"].isdigit()]
                    if epochs:
                        row["epochs"] = str(max(epochs))
            except Exception:
                pass

        rows.append(row)

    # Print terminal table
    print("=" * 110)
    print(f"{'Run ID':<36} | {'Status':<8} | {'Epochs':<6} | {'Direct MAE':<10} | {'d_v19':<8} | {'TTA MAE':<10} | {'d_TTA':<8}")
    print("-" * 110)
    for r in rows:
        status = "EXISTS" if r["exists"] else "PENDING"
        print(f"{r['run_id']:<36} | {status:<8} | {r['epochs']:<6} | {r['mae']:<10} | {r['delta_v19_direct']:<8} | {r['tta_mae']:<10} | {r['delta_v19_tta']:<8}")
    print("=" * 110)

    # Write Markdown summary
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# RMR-v30 Benchmark & Hypothesis Ladder Results Summary\n\n")
        f.write(f"**Gold Standard Baseline:** RMR-v19 Canonical Isotropic (Direct MAE: {V19_DIRECT_MAE:.2f}, TTA MAE: {V19_TTA_MAE:.2f})\n\n")
        f.write("| Run ID | Description | Epochs | Direct MAE | RMSE | Δ v19 | TTA MAE | Δ TTA | Sparse | Moderate | Dense | GAME0 | GAME1 |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in rows:
            f.write(f"| `{r['run_id']}` | {r['description']} | {r['epochs']} | {r['mae']} | {r['rmse']} | {r['delta_v19_direct']} | {r['tta_mae']} | {r['delta_v19_tta']} | {r['sparse_mae']} | {r['moderate_mae']} | {r['dense_mae']} | {r['game0']} | {r['game1']} |\n")

    # Write CSV summary
    out_csv = Path(args.output_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary reports to:")
    print(f"  - Markdown: {out_md}")
    print(f"  - CSV:      {out_csv}")


if __name__ == "__main__":
    main()
