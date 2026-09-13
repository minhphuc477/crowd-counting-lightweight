"""Summarize RMR-v11 benchmark results across all 6 matrix runs alongside v10 baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize RMR-v11 benchmark results.")
    parser.add_argument("--runs-dir", default="runs/sha_a", help="Directory containing run directories")
    parser.add_argument("--output-md", default="runs/sha_a/rmr_v11_benchmark_summary.md", help="Path to output Markdown")
    parser.add_argument("--output-csv", default="runs/sha_a/rmr_v11_benchmark_summary.csv", help="Path to output CSV")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    canonical_order = [
        "rmr_v10_dynamic_scale_routing",  # Historical Milestone Baseline
        "rmr_v11_canonical_dsr",          # Flagship
        "rmr_v11_ablation_no_curvature",   # Ablation 1
        "rmr_v11_ablation_no_trust_region",# Ablation 2
        "rmr_v11_ablation_no_hard_bg",     # Ablation 3
        "rmr_v11_ablation_no_fg_gate",     # Ablation 4
        "rmr_v11_control_no_solver",      # Baseline Control
    ]

    matched_runs = [runs_dir / name for name in canonical_order if (runs_dir / name).exists()]

    rows = []
    for rd in matched_runs:
        row = {
            "name": rd.name,
            "trained_epochs": "-",
            "best_epoch": "-",
            "mae": "-",
            "rmse": "-",
            "nae": "-",
            "bias": "-",
            "sparse_mae": "-",
            "moderate_mae": "-",
            "dense_mae": "-",
            "solver_help": "-",
        }

        # Check eval_val summary
        summary_file = rd / "eval_val" / "summary.json"
        if summary_file.exists():
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                for k in ["mae", "rmse", "nae", "bias"]:
                    if k in data:
                        row[k] = f"{data[k]:.2f}"
                if "density_bins" in data and len(data["density_bins"]) >= 3:
                    row["sparse_mae"] = f"{data['density_bins'][0].get('mae', 0.0):.2f}"
                    row["moderate_mae"] = f"{data['density_bins'][1].get('mae', 0.0):.2f}"
                    row["dense_mae"] = f"{data['density_bins'][2].get('mae', 0.0):.2f}"
                if "solver_help_fraction" in data:
                    row["solver_help"] = f"{data['solver_help_fraction']*100:.1f}%"
            except Exception:
                pass

        # Check train metrics log for epochs
        log_file = rd / "train_metrics.jsonl"
        if log_file.exists():
            try:
                lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                if lines:
                    last_obj = json.loads(lines[-1])
                    row["trained_epochs"] = str(last_obj.get("epoch", len(lines)))
            except Exception:
                pass

        rows.append(row)

    if not rows:
        print("No RMR-v11 runs found with eval_val summaries.")
        return

    # Print markdown table
    headers = [
        "Run ID", "Epochs", "MAE", "RMSE", "Bias",
        "Sparse MAE (<=100)", "Mod MAE (101-500)", "Dense MAE (>500)", "Solver Help %"
    ]
    md_lines = [
        "# RMR-v11 Scientific Benchmark & Ablation Summary\n",
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---']*len(headers))} |"
    ]

    for r in rows:
        line = f"| `{r['name']}` | {r['trained_epochs']} | **{r['mae']}** | {r['rmse']} | {r['bias']} | {r['sparse_mae']} | {r['moderate_mae']} | {r['dense_mae']} | {r['solver_help']} |"
        md_lines.append(line)

    md_text = "\n".join(md_lines)
    print(md_text)

    Path(args.output_md).write_text(md_text, encoding="utf-8")
    print(f"\nWrote summary table to {args.output_md}")


if __name__ == "__main__":
    main()
