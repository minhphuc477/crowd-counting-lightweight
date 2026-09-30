"""Summarize Gen 7 Factorial Matrix & Ablation Suite results across all 17 runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Gen 7 benchmark and ablation results.")
    parser.add_argument("--runs-dir", default="runs/sha_a", help="Directory containing run directories")
    parser.add_argument("--output-md", default="runs/sha_a/gen7_benchmark_summary.md", help="Path to output Markdown")
    parser.add_argument("--output-csv", default="runs/sha_a/gen7_benchmark_summary.csv", help="Path to output CSV")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    # Benchmark Gold Reference: Gen 6 Canonical (v19 Isotropic: 72.84 Direct MAE, 72.61 TTA MAE)
    REF_DIRECT_MAE = 72.84
    REF_TTA_MAE = 72.61

    gen7_catalog = [
        # 1. Base 4 Isolated Innovations
        ("g7_h4_drs", "Base H4: Dynamic Regional Scaling (DRS: lambda=0.10, s_max=2.5, tau_c=12)"),
        ("g7_h1_fidt", "Base H1: Normalized Radon FIDT (k=4.0, balanced smooth L1, count-normalized)"),
        ("g7_h2_chfl", "Base H2: Rescaled Continuous ChfL (omega_max=0.5 rad/px, lambda=0.05)"),
        ("g7_cdw_factorized_diag", "Base CDW: Factorized DiAG (GroupNorm + 5x5 Context Pooling)"),

        # 2. Pairwise Interactions (2-way Orthogonal Factorials)
        ("g7_h4_h1_drs_fidt", "Pairwise: DRS (H4) + Normalized FIDT (H1)"),
        ("g7_h4_h2_drs_chfl", "Pairwise: DRS (H4) + Rescaled ChfL (H2)"),
        ("g7_h4_cdw", "Pairwise: DRS (H4) + Factorized DiAG (CDW)"),
        ("g7_h1_h2_fidt_chfl", "Pairwise: Normalized FIDT (H1) + Rescaled ChfL (H2)"),
        ("g7_cdw_h1_fidt", "Pairwise: Factorized DiAG (CDW) + Normalized FIDT (H1)"),
        ("g7_cdw_h2_chfl", "Pairwise: Factorized DiAG (CDW) + Rescaled ChfL (H2)"),

        # 3. Triads & Champion Synthesis
        ("g7_triad_drs_fidt_cdw", "Triad 1: DRS + Normalized FIDT + Factorized DiAG"),
        ("g7_triad_drs_chfl_cdw", "Triad 2: DRS + Rescaled ChfL + Factorized DiAG"),
        ("g7_champion_synthesis", "Gen 7 Champion: Full Synthesis (DRS + FIDT + ChfL + CDW DiAG)"),

        # 4. Hyperparameter Sweeps & Sensitivity Controls
        ("g7_h1_fidt_k2", "Sweep H1: FIDT Narrow Kernel (k=2.0)"),
        ("g7_h1_fidt_k6", "Sweep H1: FIDT Wide Kernel (k=6.0)"),
        ("g7_h2_chfl_lam02", "Sweep H2: ChfL Low Weight (lambda=0.02)"),
        ("g7_h2_chfl_lam10", "Sweep H2: ChfL High Weight (lambda=0.10)"),
    ]

    rows = []

    for run_id, desc in gen7_catalog:
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
            "delta_ref_direct": "-",
            "delta_ref_tta": "-",
        }

        if rd.exists():
            eval_file = rd / "eval_val" / "summary.json"
            if not eval_file.exists():
                eval_file = rd / "test_eval.json"
            if not eval_file.exists():
                eval_file = rd / "eval_val.json"
            if not eval_file.exists():
                eval_file = rd / "eval_results.json"

            if eval_file.exists():
                try:
                    data = json.loads(eval_file.read_text(encoding="utf-8"))
                    mae = data.get("MAE") or data.get("mae") or data.get("val_mae")
                    rmse = data.get("RMSE") or data.get("rmse") or data.get("val_rmse")
                    bias = data.get("Bias") or data.get("bias")
                    if mae is not None:
                        row["mae"] = f"{float(mae):.2f}"
                        row["delta_ref_direct"] = f"{float(mae) - REF_DIRECT_MAE:+.2f}"
                    if rmse is not None:
                        row["rmse"] = f"{float(rmse):.2f}"
                    if bias is not None:
                        row["bias"] = f"{float(bias):+.2f}"

                    sp_mae = data.get("mae_sparse") or data.get("sparse_mae")
                    mod_mae = data.get("mae_moderate") or data.get("moderate_mae")
                    dn_mae = data.get("mae_dense") or data.get("dense_mae")
                    g0 = data.get("GAME0") or data.get("game_0")
                    g1 = data.get("GAME1") or data.get("game_1")

                    row["sparse_mae"] = f"{float(sp_mae):.2f}" if sp_mae is not None else "-"
                    row["moderate_mae"] = f"{float(mod_mae):.2f}" if mod_mae is not None else "-"
                    row["dense_mae"] = f"{float(dn_mae):.2f}" if dn_mae is not None else "-"
                    row["game0"] = f"{float(g0):.2f}" if g0 is not None else "-"
                    row["game1"] = f"{float(g1):.2f}" if g1 is not None else "-"
                except Exception:
                    pass

            tta_file = rd / "eval_tta.json"
            if tta_file.exists():
                try:
                    tta_data = json.loads(tta_file.read_text(encoding="utf-8"))
                    t_mae = tta_data.get("mae") or tta_data.get("val_mae")
                    t_rmse = tta_data.get("rmse") or tta_data.get("val_rmse")
                    if t_mae is not None:
                        row["tta_mae"] = f"{float(t_mae):.2f}"
                        row["delta_ref_tta"] = f"{float(t_mae) - REF_TTA_MAE:+.2f}"
                    if t_rmse is not None:
                        row["tta_rmse"] = f"{float(t_rmse):.2f}"
                except Exception:
                    pass

            log_csv = rd / "train_log.csv"
            if log_csv.exists():
                try:
                    with open(log_csv, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        epochs_seen = [int(r["epoch"]) for r in reader if "epoch" in r and r["epoch"].isdigit()]
                        if epochs_seen:
                            row["epochs"] = str(max(epochs_seen) + 1)
                except Exception:
                    pass

        rows.append(row)

    md_lines = [
        "# Gen 7 Factorial Matrix & Breakthrough Ablation Suite Summary",
        "",
        f"**Gold Baseline Reference**: Gen 6 Canonical (v19 Isotropic): Direct MAE = {REF_DIRECT_MAE}, TTA MAE = {REF_TTA_MAE}",
        "**Strict Constraints**: $\\le 105,000$ parameters (Gen 7: 104,441 - 104,767), 0.0% Distillation, ShanghaiTech Part A",
        "",
        "| Run ID | Description | Status | Epochs | MAE | RMSE | Bias | Sparse | Mod | Dense | $\\Delta$ Ref | TTA MAE |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in rows:
        status = "Completed" if r["mae"] != "-" else ("In Progress" if r["exists"] else "Pending")
        md_lines.append(
            f"| `{r['run_id']}` | {r['description']} | {status} | {r['epochs']} | "
            f"**{r['mae']}** | {r['rmse']} | {r['bias']} | {r['sparse_mae']} | "
            f"{r['moderate_mae']} | {r['dense_mae']} | {r['delta_ref_direct']} | {r['tta_mae']} |"
        )

    md_content = "\n".join(md_lines) + "\n"
    Path(args.output_md).write_text(md_content, encoding="utf-8")

    csv_fields = [
        "run_id", "description", "epochs", "mae", "rmse", "bias",
        "sparse_mae", "moderate_mae", "dense_mae", "game0", "game1",
        "delta_ref_direct", "tta_mae", "tta_rmse", "delta_ref_tta",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in csv_fields})

    print(f"Summary written to {args.output_md} and {args.output_csv}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
