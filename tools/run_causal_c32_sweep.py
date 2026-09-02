#!/usr/bin/env python3
"""Run Causal Backward-Only C32 Gradient Scaling Sweep.

Evaluates alpha in {0.0, 0.25, 0.5, 0.75, 1.0} to test whether suppressing
the antagonistic C32 gradient into the shared backbone improves C16 optimization,
MAE, and dense crowd counting.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


def run_command(cmd: List[str], desc: str) -> None:
    print(f"\n[RUNNING] {' '.join(cmd)}", flush=True)
    start_t = time.time()
    res = subprocess.run(cmd, check=True)
    elapsed = time.time() - start_t
    print(f"[COMPLETED in {elapsed:.1f}s] {desc}\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal C32 gradient scaling sweep.")
    parser.add_argument("--base-config", type=str, default="configs/factorial_b_crop256_c32.yaml", help="Base config YAML")
    parser.add_argument("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0", help="Comma-separated alpha values")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override (defaults to base config epochs)")
    parser.add_argument("--prefix", type=str, default="causal_b", help="Prefix for run names")
    parser.add_argument("--output-summary", type=str, default="runs/causal_c32_sweep_summary.json", help="Summary JSON output")
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    print(f"Starting Causal C32 Scaling Sweep for alphas: {alphas}", flush=True)

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    py_exe = sys.executable
    summary_results: List[Dict[str, Any]] = []

    for alpha in alphas:
        alpha_str = f"{alpha:.2f}".replace(".", "_")
        run_name = f"{args.prefix}_alpha_{alpha_str}"
        run_dir = Path(f"runs/{run_name}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"

        # Construct specific config with c32_grad_scale
        cfg = copy.deepcopy(base_cfg)
        cfg["experiment"]["name"] = run_name
        cfg["experiment"]["save_dir"] = f"./runs/{run_name}"
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model"]["c32_grad_scale"] = float(alpha)

        if args.epochs is not None:
            if "schedule" not in cfg:
                cfg["schedule"] = {}
            cfg["schedule"]["epochs"] = int(args.epochs)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

        best_ckpt = run_dir / "best.pt"
        loc_json = run_dir / "localization_eval.json"
        grad_json = run_dir / "route_gradient_audit.json"
        d0_json = run_dir / "d0_diagnostics.json"

        # 1. Train if best checkpoint does not already exist
        if not best_ckpt.exists():
            print("\n" + "=" * 70)
            print(f"  TRAINING CAUSAL RUN: {run_name} (alpha={alpha})")
            print("=" * 70, flush=True)
            cmd_train = [py_exe, "train_ntpc.py", "--config", str(config_path), "--overwrite"]
            run_command(cmd_train, f"Training {run_name}")

        # 2. Evaluate Localization (OT-M)
        if not loc_json.exists():
            print(f"Evaluating localization on {best_ckpt}...", flush=True)
            cmd_eval = [
                py_exe,
                "tools/eval_localization.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(best_ckpt),
                "--output",
                str(loc_json),
            ]
            run_command(cmd_eval, f"Localization eval for {run_name}")

        # 3. Route Gradient Audit
        if not grad_json.exists():
            print(f"Running route gradient audit on {best_ckpt}...", flush=True)
            cmd_grad = [
                py_exe,
                "tools/audit_route_gradients.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(best_ckpt),
                "--output",
                str(grad_json),
            ]
            run_command(cmd_grad, f"Route gradient audit for {run_name}")

        # 4. D0 Diagnostics
        if not d0_json.exists():
            print(f"Running D0 diagnostics on {best_ckpt}...", flush=True)
            cmd_d0 = [
                py_exe,
                "tools/run_d0_diagnostics.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(best_ckpt),
                "--output",
                str(d0_json),
            ]
            run_command(cmd_d0, f"D0 diagnostics for {run_name}")

        # Collect summary record
        record: Dict[str, Any] = {"alpha": alpha, "run_name": run_name}
        if loc_json.exists():
            with open(loc_json, "r", encoding="utf-8") as f:
                loc_data = json.load(f)
            record["counting"] = loc_data.get("counting", {})
            record["subgroups"] = loc_data.get("subgroups", {})
            record["localization"] = loc_data.get("localization", {})
        if grad_json.exists():
            with open(grad_json, "r", encoding="utf-8") as f:
                grad_data = json.load(f)
            record["route_gradients"] = {
                "overall": grad_data.get("overall", {}),
                "stratification": grad_data.get("stratification", {}),
            }
        if d0_json.exists():
            with open(d0_json, "r", encoding="utf-8") as f:
                d0_data = json.load(f)
            record["d0_synthesis"] = d0_data.get("diagnostic_synthesis", {})

        summary_results.append(record)

    out_summary_path = Path(args.output_summary)
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    # Print Final Causal Comparison Table
    print("\n" + "=" * 80)
    print("           CAUSAL BACKWARD-ONLY C32 SCALING SWEEP SUMMARY")
    print("=" * 80)
    header = f"{'Alpha':<8} | {'MAE':<8} | {'RMSE':<8} | {'Pred/GT':<8} | {'Dense MAE':<10} | {'OT-M F1@8':<10} | {'Conflict%':<10}"
    print(header)
    print("-" * 80)
    for rec in summary_results:
        a = rec.get("alpha", 0.0)
        cnt = rec.get("counting", {})
        sub = rec.get("subgroups", {})
        loc = rec.get("localization", {}).get("otm", {})
        gr = rec.get("route_gradients", {}).get("overall", {})
        mae = f"{cnt.get('mae', 0.0):.2f}"
        rmse = f"{cnt.get('rmse', 0.0):.2f}"
        ratio = f"{sub.get('pred_gt_ratio', 0.0):.3f}"
        dense_mae = f"{sub.get('bin_dense_mae', 0.0):.1f}"
        f1_8 = f"{loc.get('sigma_8_f1', 0.0) * 100:.2f}%" if loc else "N/A"
        conf = gr.get("conflict_percentage", "N/A")
        print(f"{a:<8.2f} | {mae:<8} | {rmse:<8} | {ratio:<8} | {dense_mae:<10} | {f1_8:<10} | {conf:<10}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
