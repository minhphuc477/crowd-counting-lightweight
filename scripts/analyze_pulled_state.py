#!/usr/bin/env python3
"""Comprehensive post-pull quantitative analysis of all benchmarks, ablations, and training sweeps.

Strictly read-only on experimental artifacts; outputs verified tables and statistics
directly from eval_val/summary.json, train_log.csv, resolved_config.yaml, and last.pt metadata.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rmr_v3.model import RMRv3, RMRv3Config


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_model_params(cfg_path: Path) -> int:
    cfg = load_yaml(cfg_path)
    m_dict = dict(cfg.get("model", {}))
    m_dict["pretrained"] = False
    m_cfg = RMRv3Config(**{k: v for k, v in m_dict.items() if k in RMRv3Config.__dataclass_fields__})
    model = RMRv3(m_cfg)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_train_log(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"cur_epoch": 0, "best_ep": None, "best_mae_log": None, "last_val_mae": None, "last_train_loss": None}
    cur_epoch = 0
    best_ep = None
    best_mae_log = float("inf")
    last_val_mae = None
    last_train_loss = None
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["epoch"])
                cur_epoch = ep
            except (ValueError, KeyError):
                continue
            tl = row.get("train_total", "")
            if tl not in ("", None):
                try:
                    last_train_loss = float(tl)
                except ValueError:
                    pass
            vm = row.get("val_mae", "")
            if vm not in ("", None):
                try:
                    val = float(vm)
                    last_val_mae = val
                    if val < best_mae_log:
                        best_mae_log = val
                        best_ep = ep
                except ValueError:
                    pass
    return {
        "cur_epoch": cur_epoch,
        "best_ep": best_ep,
        "best_mae_log": None if best_ep is None else best_mae_log,
        "last_val_mae": last_val_mae,
        "last_train_loss": last_train_loss,
    }


def inspect_ckpt_commit(ckpt_path: Path) -> str:
    if not ckpt_path.is_file():
        return "-"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        git_info = ckpt.get("git_info", {})
        if isinstance(git_info, dict) and git_info.get("commit"):
            return str(git_info["commit"])[:7]
        return str(ckpt.get("git_commit", "unknown"))[:7]
    except Exception as ex:
        return f"ERR({type(ex).__name__})"


def read_run(cfg_path: Path) -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    out_dir = _REPO_ROOT / cfg["output_dir"]
    target_epochs = int(cfg.get("train", {}).get("epochs", 0))
    log_info = parse_train_log(out_dir / "train_log.csv")
    sum_path = out_dir / "eval_val" / "summary.json"
    summary: dict[str, Any] = {}
    if sum_path.is_file():
        with open(sum_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    cur_ep = log_info["cur_epoch"]
    if cur_ep == 0:
        status = "DIRTY" if (out_dir.exists() and any(out_dir.iterdir())) else "NOT_STARTED"
    elif cur_ep < target_epochs:
        status = "INCOMPLETE"
    else:
        status = "COMPLETED"

    return {
        "name": cfg_path.stem,
        "cfg_path": cfg_path.relative_to(_REPO_ROOT).as_posix(),
        "out_dir": cfg["output_dir"],
        "seed": cfg.get("seed"),
        "pretrained": cfg.get("model", {}).get("pretrained"),
        "status": status,
        "cur_ep": cur_ep,
        "target_ep": target_epochs,
        "best_ep": log_info["best_ep"],
        "last_val_mae": log_info["last_val_mae"],
        "last_train_loss": log_info["last_train_loss"],
        "MAE": summary.get("MAE", summary.get("mae")),
        "RMSE": summary.get("RMSE", summary.get("rmse")),
        "NAE": summary.get("NAE", summary.get("nae")),
        "Bias": summary.get("Bias", summary.get("bias")),
        "SparseMAE": summary.get("mae_sparse"),
        "ModMAE": summary.get("mae_moderate"),
        "DenseMAE": summary.get("mae_dense"),
        "GAME3": summary.get("GAME3"),
        "last_pt_commit": inspect_ckpt_commit(out_dir / "last.pt") if status == "INCOMPLETE" else "-",
    }


def fmt_num(val: float | None, spec: str = ".2f") -> str:
    if val is None:
        return "-"
    return format(val, spec)


def print_run_table(title: str, runs: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 134}")
    print(title)
    print(f"{'=' * 134}")
    print(
        f"{'Run Name':42s} | {'Status':11s} | {'Ep':9s} | {'Best':4s} | "
        f"{'MAE':>6s} | {'RMSE':>6s} | {'NAE':>5s} | {'Bias':>7s} | "
        f"{'Dense':>6s} | {'GAME3':>6s} | {'LastVal':>7s} | {'CkptGit':7s}"
    )
    print("-" * 134)
    for r in runs:
        ep_s = f"{r['cur_ep']}/{r['target_ep']}"
        best_s = "-" if r["best_ep"] is None else str(r["best_ep"])
        print(
            f"{r['name']:42s} | {r['status']:11s} | {ep_s:9s} | {best_s:>4s} | "
            f"{fmt_num(r['MAE']):>6s} | {fmt_num(r['RMSE']):>6s} | {fmt_num(r['NAE'], '.3f'):>5s} | "
            f"{fmt_num(r['Bias'], '+.2f'):>7s} | {fmt_num(r['DenseMAE']):>6s} | {fmt_num(r['GAME3']):>6s} | "
            f"{fmt_num(r['last_val_mae']):>7s} | {r['last_pt_commit']:7s}"
        )


def print_seed_stats(label: str, maes: list[float], rmses: list[float], biases: list[float], denses: list[float]) -> None:
    def ms(xs: list[float]) -> str:
        arr = np.asarray(xs, dtype=np.float64)
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return f"{np.mean(arr):.2f} +/- {std:.2f} (n={len(arr)})"

    print(f"{label:28s} | MAE: {ms(maes):24s} | RMSE: {ms(rmses):24s} | Bias: {ms(biases):24s} | DenseMAE: {ms(denses)}")


def main() -> None:
    head_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    print(f"Repository HEAD: {head_commit}")

    print("\n=== ARCHITECTURAL PARAMETER VERIFICATION ===")
    param_check_cfgs = [
        "configs/benchmarks/exp_sha_direct_pretrained_104k.yaml",
        "configs/benchmarks/ablations_sha/exp_sha_abl_b4_no_neck_104k.yaml",
        "configs/benchmarks/ablations_sha/exp_sha_abl_b5_stride4_104k.yaml",
        "configs/benchmarks/ablations_sha/exp_sha_abl_b1_solver_sirt_104k.yaml",
        "configs/benchmarks/ablations_jhu/exp_jhu_abl_c2_no_neck_104k.yaml",
        "configs/benchmarks/ablations_jhu/exp_jhu_abl_c1_stride4_104k.yaml",
        "configs/benchmarks/ablations_jhu/exp_jhu_abl_b3_count_nb_disp10_104k.yaml",
        "configs/benchmarks/ablations_qnrf/exp_qnrf_abl_c5_no_neck_104k.yaml",
        "configs/benchmarks/ablations_qnrf/exp_qnrf_abl_b1_solver_sirt_104k.yaml",
    ]
    for cp in param_check_cfgs:
        p = _REPO_ROOT / cp
        c = load_yaml(p)
        m = c["model"]
        n_params = count_model_params(p)
        print(
            f"{p.stem:40s} | params={n_params:7d} | stride={m.get('output_stride')} | "
            f"dilations={m.get('aspp_dilations')} | gap={m.get('use_aspp_gap')} | solver={m.get('enable_solver')}"
        )

    bench = {p.stem: read_run(p) for p in sorted((_REPO_ROOT / "configs/benchmarks").glob("*.yaml"))}
    sha_sweeps = [read_run(p) for p in sorted((_REPO_ROOT / "configs/training_sweeps").glob("*.yaml"))]
    qnrf_sweeps = [read_run(p) for p in sorted((_REPO_ROOT / "configs/training_sweeps/qnrf").glob("*.yaml"))]
    jhu_sweeps = [read_run(p) for p in sorted((_REPO_ROOT / "configs/training_sweeps/jhu").glob("*.yaml"))]

    print_run_table("SHA BENCHMARKS + TRAINING SWEEPS", [bench["exp_sha_direct_pretrained_104k"]] + sha_sweeps)
    print_run_table("QNRF BENCHMARKS + TRAINING SWEEPS", [bench["exp_qnrf_direct_pretrained_104k"]] + qnrf_sweeps)
    print_run_table("JHU BENCHMARKS + TRAINING SWEEPS", [bench["exp_jhu_direct_pretrained_104k"]] + jhu_sweeps)

    sweeps_map = {r["name"]: r for r in sha_sweeps + qnrf_sweeps + jhu_sweeps}

    print("\n=== MULTI-SEED STATISTICAL AGGREGATION (sample std, ddof=1) ===")
    for ds_key, base_name, seed_names in [
        ("SHA (Pretrained, 3/3 done)", "exp_sha_direct_pretrained_104k", ["exp_sha_seed_1337_104k", "exp_sha_seed_2024_104k"]),
        ("JHU Val (Pretrained, 3/3 done)", "exp_jhu_direct_pretrained_104k", ["exp_jhu_seed_1337_104k", "exp_jhu_seed_2024_104k"]),
        ("QNRF (Pretrained, 2/3 done)", "exp_qnrf_direct_pretrained_104k", ["exp_qnrf_seed_2024_104k"]),
    ]:
        items = [bench[base_name]] + [sweeps_map[n] for n in seed_names if sweeps_map[n]["status"] == "COMPLETED"]
        print_seed_stats(
            ds_key,
            [r["MAE"] for r in items],
            [r["RMSE"] for r in items],
            [r["Bias"] for r in items],
            [r["DenseMAE"] for r in items],
        )

    for mode, prefix in [("UCF_CC_50 Pretrained 5-fold", "exp_ucf_cc_50_fold"), ("UCF_CC_50 Scratch 5-fold", "exp_ucf_cc_50_scratch_fold")]:
        items = [bench[f"{prefix}{i}"] for i in range(1, 6)]
        print_seed_stats(
            mode,
            [r["MAE"] for r in items],
            [r["RMSE"] for r in items],
            [r["Bias"] for r in items],
            [r["DenseMAE"] for r in items],
        )

    print("\n=== ALL ABLATIONS EXTENDED DETAILS (Best Ep, LastVal, GAME3, DenseMAE) ===")
    for ds in ["sha", "shb", "qnrf", "jhu"]:
        runs = [read_run(p) for p in sorted((_REPO_ROOT / f"configs/benchmarks/ablations_{ds}").glob("*.yaml"))]
        print_run_table(f"ABLATIONS - {ds.upper()}", runs)

    rem_dir = _REPO_ROOT / "configs/benchmarks/remediation"
    if rem_dir.is_dir():
        rem_runs = [read_run(p) for p in sorted(rem_dir.glob("*.yaml"))]
        if rem_runs:
            print_run_table("REMEDIATION SUITE", rem_runs)

    print("\n=== EXISTING RESULTS JSONS SUMMARY ===")
    for jname in ["localization_pr_sweep.json", "scale_invariance_sweep.json", "image_corruptions_sweep.json", "annotation_noise_sweep.json"]:
        jp = _REPO_ROOT / "results" / jname
        if jp.is_file():
            data = json.loads(jp.read_text(encoding="utf-8"))
            print(f"[{jname}] top-level keys: {list(data.keys())}")


if __name__ == "__main__":
    main()
