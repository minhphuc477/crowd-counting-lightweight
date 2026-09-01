"""Sequential runner for NTPC Ablation Study (R0, R1, R2, R3).

Runs each model sequentially with exact matched protocol, runs localization evaluation,
and compiles a unified comparison table with R4 and R5.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = [
    ("R0 (Exact Regression)", "configs/ntpc_r0_exact_regression.yaml", "runs/ntpc_r0_exact_regression"),
    ("R1 (SDC Deterministic)", "configs/ntpc_r1_sdc_deterministic.yaml", "runs/ntpc_r1_sdc_deterministic"),
    ("R2 (Flat DM-16)", "configs/ntpc_r2_flat_dm16.yaml", "runs/ntpc_r2_flat_dm16"),
    ("R3 (Multinomial Tree)", "configs/ntpc_r3_hierarchical_multinomial.yaml", "runs/ntpc_r3_hierarchical_multinomial"),
]

ALL_RUNS = [
    ("R0_Exact_Regression", "runs/ntpc_r0_exact_regression"),
    ("R1_SDC_Deterministic", "runs/ntpc_r1_sdc_deterministic"),
    ("R2_Flat_DM16", "runs/ntpc_r2_flat_dm16"),
    ("R3_Multinomial_Tree", "runs/ntpc_r3_hierarchical_multinomial"),
    ("R4_Neural_DTM_Tree16", "runs/ntpc_r4_neural_dtm_tree"),
    ("R5_Full_Adaptive_NTPC", "runs/ntpc_r5_full_adaptive_ntpc"),
]


def run_command(cmd: list[str]) -> None:
    print(f"\n[RUNNING] {' '.join(cmd)}", flush=True)
    start = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start
    if res.returncode != 0:
        print(f"[FAILED] Command exited with code {res.returncode} after {elapsed:.1f}s", flush=True)
        sys.exit(res.returncode)
    print(f"[COMPLETED] in {elapsed:.1f}s", flush=True)


def summarize_all():
    print("\n" + "=" * 90)
    print("ALL ABLATIONS SUMMARY TABLE (R0 - R5)")
    print("=" * 90)
    
    summary = []
    header = f"{'Model':<24} | {'MAE':<7} {'RMSE':<7} {'Bias':<8} {'Ratio':<6} | {'Sparse':<7} {'Mid':<7} {'Dense':<7} | {'OT-M F1@8':<10} {'F1@4':<7}"
    print(header)
    print("-" * 90)
    
    for name, run_dir in ALL_RUNS:
        loc_file = Path(run_dir) / "localization_eval.json"
        if not loc_file.exists():
            print(f"{name:<24} | [NOT FOUND / RUNNING]")
            continue
        try:
            with open(loc_file, "r") as f:
                data = json.load(f)
            c = data.get("counting", {})
            sub = data.get("subgroups", {})
            loc = data.get("localization", {}).get("otm", {})
            
            mae = c.get("mae", float("nan"))
            rmse = c.get("rmse", float("nan"))
            bias = c.get("bias", float("nan"))
            ratio = sub.get("pred_gt_ratio", float("nan"))
            sparse = sub.get("bin_sparse_mae", float("nan"))
            mid = sub.get("bin_medium_mae", float("nan"))
            dense = sub.get("bin_dense_mae", float("nan"))
            f1_8 = loc.get("sigma_8_f1", float("nan")) * 100.0 if "sigma_8_f1" in loc else float("nan")
            f1_4 = loc.get("sigma_4_f1", float("nan")) * 100.0 if "sigma_4_f1" in loc else float("nan")
            
            row_str = f"{name:<24} | {mae:<7.2f} {rmse:<7.2f} {bias:<8.2f} {ratio:<6.3f} | {sparse:<7.2f} {mid:<7.2f} {dense:<7.2f} | {f1_8:<10.2f} {f1_4:<7.2f}"
            print(row_str)
            summary.append({
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
                "pred_gt_ratio": ratio,
                "sparse_mae": sparse,
                "mid_mae": mid,
                "dense_mae": dense,
                "otm_f1_8": f1_8,
                "otm_f1_4": f1_4,
            })
        except Exception as e:
            print(f"{name:<24} | [ERROR reading {loc_file}: {e}]")
            
    print("=" * 90)
    out_summary_file = Path("runs/ablation_study_summary.json")
    with open(out_summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved complete ablation summary to {out_summary_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all NTPC ablations sequentially")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runs")
    args = parser.parse_args()

    python_exe = sys.executable
    for tag, config_path, run_dir in EXPERIMENTS:
        print(f"\n{'='*70}\nSTARTING ABLATION: {tag}\nConfig: {config_path}\n{'='*70}", flush=True)
        
        # 1. Run training
        train_cmd = [python_exe, "-u", "train_ntpc.py", "--config", config_path]
        if args.overwrite:
            train_cmd.append("--overwrite")
        run_command(train_cmd)
        
        # 2. Run localization eval
        best_pt = os.path.join(run_dir, "best.pt")
        out_json = os.path.join(run_dir, "localization_eval.json")
        eval_cmd = [
            python_exe, "tools/eval_localization.py",
            "--config", config_path,
            "--checkpoint", best_pt,
            "--output", out_json
        ]
        run_command(eval_cmd)
        
        # 3. Print ongoing summary
        summarize_all()


if __name__ == "__main__":
    main()
