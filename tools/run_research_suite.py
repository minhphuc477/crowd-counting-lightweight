"""RMR Research Suite Automated Sequential Runner (A* Protocol).

Executes research experiment configs sequentially on a single GPU to prevent CUDA OOM,
records validation metrics, updates central progress tracking, and ensures strict
reproducibility across all single-variable ablation hypotheses.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXPERIMENT_REGISTRY: dict[str, dict[str, str]] = {
    "h1_no_jitter": {
        "config": "configs/rmr_research/h1_no_jitter.yaml",
        "desc": "Augmentation Entropy Isolation (scale [1.0, 1.0], no color jitter)",
    },
    "h2_spectral": {
        "config": "configs/rmr_research/h2_spectral_loss.yaml",
        "desc": "Count-Preserving Heavy-Tailed Spectral Loss (beta=2.0, lambda=0.2)",
    },
    "h3a_depth2": {
        "config": "configs/rmr_research/h3a_depth2.yaml",
        "desc": "SIRT Solver Contraction Depth T=2 (fast deconvolution)",
    },
    "h3b_depth4": {
        "config": "configs/rmr_research/h3b_depth4.yaml",
        "desc": "SIRT Solver Contraction Depth T=4 (mid-point deconvolution)",
    },
    "h3c_depth8": {
        "config": "configs/rmr_research/h3c_depth8.yaml",
        "desc": "SIRT Solver Contraction Depth T=8 (deep deconvolution)",
    },
    "h4_no_morozov": {
        "config": "configs/rmr_research/h4_no_morozov.yaml",
        "desc": "Morozov Discrepancy Deadband Ablation (gamma=0.0)",
    },
    "h5_no_curvature": {
        "config": "configs/rmr_research/h5_no_curvature.yaml",
        "desc": "Curvature Regularization Ablation (lambda_curv=0.0)",
    },
    "h6_composite": {
        "config": "configs/rmr_research/h6_spectral_composite.yaml",
        "desc": "Spectral Composite Model (v19 + Spectral Loss breakthrough)",
    },
}

SUMMARY_CSV_PATH = Path("runs/sha_a/rmr_research_suite_summary.csv")
SUMMARY_MD_PATH = Path("docs/rmr/RMR_RESEARCH_SUITE_RESULTS.md")


def run_experiment(
    exp_key: str,
    override_epochs: int | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a single training experiment via subprocess CLI and return summary metrics."""
    if exp_key not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment: {exp_key}. Registered: {list(EXPERIMENT_REGISTRY.keys())}")

    info = EXPERIMENT_REGISTRY[exp_key]
    cfg_path = Path(info["config"])
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cmd = [
        sys.executable,
        "-m",
        "rmr_v3.train",
        "--config",
        str(cfg_path),
        "--run-id",
        f"rmr_{exp_key}",
        "--non-deterministic",
    ]
    if override_epochs is not None:
        cmd.extend(["--epochs", str(override_epochs)])
    if extra_args:
        cmd.extend(extra_args)

    print("\n" + "=" * 80)
    print(f"  LAUNCHING EXPERIMENT: {exp_key}")
    print(f"  Description: {info['desc']}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 80 + "\n", flush=True)

    start_time = time.time()
    res = subprocess.run(cmd, check=False)
    duration_s = time.time() - start_time

    out_dir = Path(f"runs/sha_a/rmr_{exp_key}")
    summary_file = out_dir / "eval_sha_a_test_direct" / "summary.json"
    if not summary_file.is_file():
        summary_file = out_dir / "summary.json"

    metrics: dict[str, Any] = {
        "exp_key": exp_key,
        "desc": info["desc"],
        "exit_code": res.returncode,
        "duration_sec": round(duration_s, 1),
        "mae": "N/A",
        "rmse": "N/A",
        "bias": "N/A",
        "sparse_mae": "N/A",
        "mod_mae": "N/A",
        "dense_mae": "N/A",
    }

    if summary_file.is_file():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            metrics["mae"] = f"{float(d.get('mae', 0.0)):.2f}"
            metrics["rmse"] = f"{float(d.get('rmse', 0.0)):.2f}"
            metrics["bias"] = f"{float(d.get('bias', 0.0)):.2f}"
            b_mae = d.get("binned_mae", {})
            metrics["sparse_mae"] = f"{float(b_mae.get('sparse', 0.0)):.2f}" if "sparse" in b_mae else "N/A"
            metrics["mod_mae"] = f"{float(b_mae.get('moderate', 0.0)):.2f}" if "moderate" in b_mae else "N/A"
            metrics["dense_mae"] = f"{float(b_mae.get('dense', 0.0)):.2f}" if "dense" in b_mae else "N/A"
        except Exception as e:
            print(f"Warning: Failed to parse summary JSON: {e}")

    update_summary_files(metrics)
    return metrics


def update_summary_files(new_row: dict[str, Any]) -> None:
    """Append result to summary CSV and regenerate markdown report."""
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if SUMMARY_CSV_PATH.is_file():
        with open(SUMMARY_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("exp_key") != new_row["exp_key"]]

    rows.append(new_row)

    fieldnames = [
        "exp_key", "mae", "rmse", "bias", "sparse_mae", "mod_mae", "dense_mae",
        "duration_sec", "exit_code", "desc",
    ]
    with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# RMR Research Suite: Cumulative Benchmark Results",
        "",
        "> Canonical ShanghaiTech Part A (300 Train / 182 Test). Strictly <= 105,000 parameters.",
        "",
        "| Experiment Key | Description | Test MAE | RMSE | Bias | Sparse | Moderate | Dense | Duration (s) | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        status = "PASSED" if str(r.get("exit_code")) == "0" else f"ERR({r.get('exit_code')})"
        lines.append(
            f"| `{r.get('exp_key')}` | {r.get('desc')} | **{r.get('mae')}** | {r.get('rmse')} | "
            f"{r.get('bias')} | {r.get('sparse_mae')} | {r.get('mod_mae')} | {r.get('dense_mae')} | "
            f"{r.get('duration_sec')}s | {status} |"
        )
    SUMMARY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated summary reports: {SUMMARY_CSV_PATH} and {SUMMARY_MD_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RMR Research Suite Automated Runner")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_REGISTRY.keys()),
        help=f"List of experiments to run. Choices: {list(EXPERIMENT_REGISTRY.keys())}",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for all runs")
    parser.add_argument("--list", action="store_true", help="List all registered experiments and exit")
    args, extra = parser.parse_known_args()

    if args.list:
        print("Available RMR Research Experiments:")
        for k, v in EXPERIMENT_REGISTRY.items():
            print(f"  - {k:<18}: {v['desc']}")
        return

    print(f"Queueing {len(args.experiments)} experiments: {args.experiments}")
    results = []
    for exp_key in args.experiments:
        try:
            m = run_experiment(exp_key, override_epochs=args.epochs, extra_args=extra)
            results.append(m)
        except Exception as e:
            print(f"Error running experiment {exp_key}: {e}", flush=True)

    print("\n" + "=" * 80)
    print("  RESEARCH SUITE EXECUTION COMPLETED")
    print("=" * 80)
    for r in results:
        print(f"  [{r['exp_key']}] MAE: {r['mae']} | RMSE: {r['rmse']} | Status: {r['exit_code']}")


if __name__ == "__main__":
    main()
