#!/usr/bin/env python3
"""Intelligent Auto-Resume and Status Inspector for RMR Crowd Counting Framework.

Usage:
    python scripts/auto_resume.py --list
    python scripts/auto_resume.py --print-commands
    python scripts/auto_resume.py --run-incomplete
    python scripts/auto_resume.py --extend-to-1000
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mapping of known configs
CONFIG_MAP = {
    # V54
    "v54_track_a_trust_damped_rn": "configs/rmr_v54/v54_track_a_trust_damped_rn.yaml",
    "v54_track_b_trust_damped_rn": "configs/rmr_v54/v54_track_b_trust_damped_rn.yaml",
    "v54_track_a_raw_rn_ablation": "configs/rmr_v54/v54_track_a_raw_rn_ablation.yaml",
    "v54_track_a_control_sirt": "configs/rmr_v54/v54_track_a_control_sirt.yaml",
    "v54_track_b_control_sirt": "configs/rmr_v54/v54_track_b_control_sirt.yaml",
    # V55
    "v55_track_a_purified_rn": "configs/rmr_v55/v55_track_a_purified_rn.yaml",
    "v55_track_b_purified_rn": "configs/rmr_v55/v55_track_b_purified_rn.yaml",
    "v55_track_a_no_transport_loss": "configs/rmr_v55/v55_track_a_no_transport_loss.yaml",
    "v55_track_a_control_bloat": "configs/rmr_v55/v55_track_a_control_bloat.yaml",
    "v55_track_b_control_bloat": "configs/rmr_v55/v55_track_b_control_bloat.yaml",
    # V56
    "v56_track_a_proposed_asym_rn": "configs/rmr_v56/v56_track_a_proposed_asym_rn.yaml",
    "v56_track_b_proposed_deep_asym_rn": "configs/rmr_v56/v56_track_b_proposed_deep_asym_rn.yaml",
    "v56_track_a_ablation_sym_rn": "configs/rmr_v56/v56_track_a_ablation_sym_rn.yaml",
    "v56_track_b_ablation_sym_rn": "configs/rmr_v56/v56_track_b_ablation_sym_rn.yaml",
    # V57
    "v57_track_a_proposed_multistep_rn": "configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml",
    "v57_track_b_proposed_deep_texture_rn": "configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml",
    "v57_track_a_ablation_1step_rn": "configs/rmr_v57/v57_track_a_ablation_1step_rn.yaml",
    "v57_track_b_ablation_1step_rn": "configs/rmr_v57/v57_track_b_ablation_1step_rn.yaml",
}


def get_model_status(run_dir: Path, target_epochs: int | None = None) -> dict[str, Any]:
    """Inspect directory to determine exact run status."""
    log_csv = run_dir / "train_log.csv"
    last_pt = run_dir / "last.pt"
    best_pt = run_dir / "best_val_mae.pt"
    resolved_cfg = run_dir / "resolved_config.yaml"

    cfg_epochs = 1000
    if resolved_cfg.exists():
        try:
            with resolved_cfg.open("r", encoding="utf-8") as f:
                c = yaml.safe_load(f)
                cfg_epochs = c.get("training", {}).get("epochs", c.get("train", {}).get("epochs", 1000))
        except Exception:
            pass

    max_epochs = target_epochs if target_epochs is not None else cfg_epochs

    done_epochs = 0
    best_val_mae = None
    if log_csv.exists():
        try:
            with log_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                maes = []
                for row in reader:
                    done_epochs += 1
                    if "val_mae" in row and row["val_mae"]:
                        try:
                            maes.append(float(row["val_mae"]))
                        except ValueError:
                            pass
                if maes:
                    best_val_mae = min(maes)
        except Exception:
            pass

    is_finished = done_epochs >= max_epochs
    can_resume = (not is_finished) and last_pt.exists()

    return {
        "run_dir": run_dir,
        "done_epochs": done_epochs,
        "max_epochs": max_epochs,
        "cfg_epochs": cfg_epochs,
        "has_last": last_pt.exists(),
        "has_best": best_pt.exists(),
        "best_val_mae": best_val_mae,
        "is_finished": is_finished,
        "can_resume": can_resume,
    }


def find_matching_config(key: str) -> str | None:
    if key in CONFIG_MAP:
        return CONFIG_MAP[key]
    for k, v in CONFIG_MAP.items():
        if k in key or key in k:
            return v
    return None


def inspect_all_runs(versions: list[str]) -> list[dict[str, Any]]:
    runs = []
    base_dir = REPO_ROOT / "runs" / "sha_a"
    for v in versions:
        v_dir = base_dir / v
        if not v_dir.exists():
            continue
        for item in sorted(v_dir.iterdir()):
            if item.is_dir():
                status = get_model_status(item)
                status["version"] = v
                status["key"] = item.name
                status["cfg_path"] = find_matching_config(item.name)
                runs.append(status)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-Resume & Status Inspector")
    ap.add_argument("--list", action="store_true", help="List status of all runs")
    ap.add_argument("--print-commands", action="store_true", help="Print resume commands for incomplete runs")
    ap.add_argument("--run-incomplete", action="store_true", help="Execute resume for incomplete runs sequentially")
    ap.add_argument("--extend-to-1000", action="store_true", help="Print commands to extend 650-epoch runs to 1000 epochs")
    ap.add_argument("--versions", nargs="+", default=["rmr_v54", "rmr_v55", "rmr_v56", "rmr_v57"])
    args = ap.parse_args()

    runs = inspect_all_runs(args.versions)

    if args.list or (not args.print_commands and not args.run_incomplete and not args.extend_to_1000):
        print("=" * 80)
        print(f"RMR EXPERIMENT RUNS AUDIT ({len(runs)} models found)")
        print("=" * 80)
        for r in runs:
            status_str = "FINISHED" if r["is_finished"] else f"INCOMPLETE ({r['done_epochs']}/{r['max_epochs']} ep)"
            mae_str = f"Best MAE: {r['best_val_mae']:.2f}" if r["best_val_mae"] is not None else "No MAE"
            print(f"[{r['version']}] {r['key']}")
            print(f"    Status:      {status_str}")
            print(f"    Checkpoints: last.pt={'YES' if r['has_last'] else 'NO'}, best.pt={'YES' if r['has_best'] else 'NO'}")
            print(f"    Metric:      {mae_str}")
            print(f"    Can Resume:  {'YES' if r['can_resume'] else 'NO'}")
            print("")

    incomplete_runs = [r for r in runs if r["can_resume"]]

    if args.print_commands:
        print("=" * 80)
        print("COMMANDS TO RESUME INCOMPLETE RUNS")
        print("=" * 80)
        if not incomplete_runs:
            print("All runs are complete! No incomplete runs found.")
        for r in incomplete_runs:
            cfg = r["cfg_path"] or f"configs/{r['version']}/{r['key']}.yaml"
            last_path = (r["run_dir"] / "last.pt").relative_to(REPO_ROOT).as_posix()
            cmd = f"python train.py --config {cfg} --resume {last_path} --force-resume"
            print(f"# {r['key']} (Epoch {r['done_epochs']}/{r['max_epochs']})")
            print(f"{cmd}\n")

    if args.extend_to_1000:
        print("=" * 80)
        print("COMMANDS TO EXTEND 650-EPOCH RUNS (V56 & V57) TO 1000 EPOCHS")
        print("=" * 80)
        for r in runs:
            if r["cfg_epochs"] == 650 and r["done_epochs"] >= 650:
                cfg = r["cfg_path"] or f"configs/{r['version']}/{r['key']}.yaml"
                last_path = (r["run_dir"] / "last.pt").relative_to(REPO_ROOT).as_posix()
                cmd = f"python train.py --config {cfg} --resume {last_path} --force-resume --epochs 1000"
                print(f"# Extend {r['key']} from 650 -> 1000 epochs (Best MAE: {r['best_val_mae']:.2f})")
                print(f"{cmd}\n")

    if args.run_incomplete:
        if not incomplete_runs:
            print("No incomplete runs to resume!")
            return
        print(f"Resuming {len(incomplete_runs)} incomplete run(s) sequentially...")
        for r in incomplete_runs:
            cfg = r["cfg_path"] or f"configs/{r['version']}/{r['key']}.yaml"
            last_path = r["run_dir"] / "last.pt"
            cmd = [
                sys.executable,
                "train.py",
                "--config",
                str(cfg),
                "--resume",
                str(last_path),
                "--force-resume",
            ]
            print(f"\n>>> [RESUMING] {r['key']} from {last_path}")
            print(f"    Command: {' '.join(cmd)}")
            ret = subprocess.run(cmd, cwd=REPO_ROOT)
            if ret.returncode != 0:
                print(f"Error resuming {r['key']}. Exited with {ret.returncode}")
                sys.exit(ret.returncode)
            print(f">>> [SUCCESS] {r['key']} completed!")


if __name__ == "__main__":
    main()
