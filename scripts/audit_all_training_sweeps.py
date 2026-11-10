#!/usr/bin/env python3
"""Audit all training sweep configs, checkpoints, training progress, and evaluation results."""
from pathlib import Path
import json
import yaml
import csv

_REPO_ROOT = Path(__file__).resolve().parent.parent

def count_epochs(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if len(lines) <= 1:
        return 0
    try:
        last_line = lines[-1].split(",")[0]
        return int(last_line)
    except Exception:
        return len(lines) - 1

def audit_configs():
    cfg_root = _REPO_ROOT / "configs" / "training_sweeps"
    all_cfgs = sorted(cfg_root.rglob("*.yaml"))

    print("=" * 105)
    print(f"{'Config Name':40s} | {'Status':14s} | {'Epochs':12s} | {'MAE':>7s} | {'RMSE':>7s} | {'Bias':>7s}")
    print("=" * 105)

    not_started = []
    partial_runs = []
    completed_runs = []

    for c_path in all_cfgs:
        with open(c_path, "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        target_epochs = int(c.get("train", {}).get("epochs", 0))
        out_dir = _REPO_ROOT / c["output_dir"]
        name = c_path.stem
        rel_cfg = c_path.relative_to(_REPO_ROOT).as_posix()

        log_p = out_dir / "train_log.csv"
        eval_p = out_dir / "eval_val" / "summary.json"
        last_p = out_dir / "last.pt"

        cur_epoch = count_epochs(log_p)
        mae_str, rmse_str, bias_str = "-", "-", "-"

        if eval_p.exists():
            with open(eval_p, "r", encoding="utf-8") as f:
                d = json.load(f)
            mae = d.get("MAE", d.get("mae", 0.0))
            rmse = d.get("RMSE", d.get("rmse", 0.0))
            bias = d.get("Bias", d.get("bias", 0.0))
            mae_str = f"{mae:7.2f}"
            rmse_str = f"{rmse:7.2f}"
            bias_str = f"{bias:+7.2f}"

        if not out_dir.exists() or cur_epoch == 0:
            status = "NOT_STARTED"
            not_started.append((name, rel_cfg, target_epochs))
        elif cur_epoch < target_epochs:
            status = "INCOMPLETE"
            partial_runs.append((name, rel_cfg, cur_epoch, target_epochs, str(last_p.relative_to(_REPO_ROOT).as_posix()) if last_p.exists() else None))
        else:
            status = "COMPLETED"
            completed_runs.append((name, rel_cfg, cur_epoch, mae_str, rmse_str, bias_str))

        ep_str = f"{cur_epoch}/{target_epochs}"
        print(f"{name:40s} | {status:14s} | {ep_str:12s} | {mae_str:>7s} | {rmse_str:>7s} | {bias_str:>7s}")

    print("\n" + "=" * 105)
    print("CHECK 4 JHU ABLATIONS PROGRESS:")
    print("=" * 105)
    jhu_4 = [
        "exp_jhu_abl_b2_no_count_loss_104k",
        "exp_jhu_abl_b3_count_nb_disp10_104k",
        "exp_jhu_abl_c3_radon_8proj_104k",
        "exp_jhu_abl_d4_crop384_104k",
    ]
    for name in jhu_4:
        out_dir = _REPO_ROOT / "runs" / "benchmarks" / "ablations_jhu" / name
        cur_ep = count_epochs(out_dir / "train_log.csv")
        eval_p = out_dir / "eval_val" / "summary.json"
        mae_str, rmse_str = "-", "-"
        if eval_p.exists():
            d = json.load(open(eval_p))
            mae_str = f"{d.get('MAE', 0.0):.2f}"
            rmse_str = f"{d.get('RMSE', 0.0):.2f}"
        print(f"{name:40s} | Epoch: {cur_ep:4d}/800 | MAE: {mae_str:>7s} | RMSE: {rmse_str:>7s}")

    print("\n" + "=" * 105)
    print("CHECK CROSS-DOMAIN EVALUATION DIRECTORIES:")
    print("=" * 105)
    for d in sorted((_REPO_ROOT / "runs").glob("eval_*")):
        s_p = d / "summary.json"
        if s_p.exists():
            data = json.load(open(s_p))
            print(f"{d.name:30s} | COMPLETED | MAE={data.get('MAE', 0.0):.2f}, RMSE={data.get('RMSE', 0.0):.2f}")
        else:
            print(f"{d.name:30s} | INCOMPLETE/EMPTY")

if __name__ == "__main__":
    audit_configs()
