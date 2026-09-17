"""Summarize RMR-v23 benchmark results across all 6 matrix runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RMR-v23 benchmark and ablation results.")
    parser.add_argument("--runs-dir", default="runs/sha_a", help="Directory containing run directories")
    parser.add_argument("--output-md", default="runs/sha_a/rmr_v23_benchmark_summary.md", help="Path to output Markdown")
    parser.add_argument("--output-csv", default="runs/sha_a/rmr_v23_benchmark_summary.csv", help="Path to output CSV")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    canonical_order = [
        ("rmr_v23_canonical", "RMR-v23 Canonical (BB-1 + Gated Diffusion)"),
        ("rmr_v23_ablation_no_bb", "Ablation 1: No Barzilai-Borwein (Fixed omega)"),
        ("rmr_v23_ablation_no_density_gated_diffusion", "Ablation 2: No Density-Gated Diffusion"),
        ("rmr_v23_ablation_no_gated_curvature", "Ablation 3: No Density-Gated Curvature"),
        ("rmr_v23_ablation_no_scale_align", "Ablation 4: No Scale Alignment Loss"),
        ("rmr_v23_control_no_solver", "Control: Direct Feedforward (No Solver)"),
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

                if run_id == "rmr_v23_canonical" and "mae" in data:
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
                lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                if len(lines) > 1:
                    row["epochs"] = str(len(lines) - 1)
            except Exception:
                pass

        rows.append(row)

    # Markdown format
    md_lines = [
        "# RMR-v23 Benchmark & Ablation Study Summary",
        "",
        "| Architecture / Variant | Epochs | Val MAE | Val RMSE | Sparse | Dense | TTA MAE | $\\Delta$MAE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r['description']}** | {r['epochs']} | {r['mae']} | {r['rmse']} | {r['sparse_mae']} | {r['dense_mae']} | {r['tta_mae']} | {r['delta_mae']} |"
        )
    md_lines.append("")

    out_md_path = Path(args.output_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown summary to: {out_md_path}")

    # CSV format
    out_csv_path = Path(args.output_csv)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV summary to: {out_csv_path}")

    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
