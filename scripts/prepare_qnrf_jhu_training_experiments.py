#!/usr/bin/env python3
"""Prepare manifests and configurations for UCF-QNRF and JHU-CROWD++ Training Sweeps.

Reflects 2023-2025 top-tier crowd counting literature (STEERER ICCV 2023, PET ICCV 2023,
MPCount CVPR 2024, APGCC ECCV 2024, mPrompt CVPR 2024):
  1. Data Efficiency / Label Scarcity on Large-Scale Datasets:
     - QNRF: 10% (120 imgs), 25% (300 imgs), 50% (600 imgs).
     - JHU:  10% (227 imgs), 25% (568 imgs), 50% (1136 imgs).
  2. Multi-Seed Statistical Variance on QNRF & JHU:
     - Seeds {1337, 2024} (to complement nominal seed 42).
  3. Noisy Label Training on UCF-QNRF:
     - Gaussian click jitter: sigma in {8, 16} px.
     - Label dropout: rho in {10%, 20%}.
  4. Crop Size & Resolution Scaling on QNRF:
     - Crop 384, Crop 768 (complementing nominal crop 512).

Strict compliance with AGENTS.md: Zero buzzwords. Pure empirical reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    data_dir = _REPO_ROOT / "data"
    cfg_base_dir = _REPO_ROOT / "configs" / "training_sweeps"
    cfg_qnrf_dir = cfg_base_dir / "qnrf"
    cfg_jhu_dir = cfg_base_dir / "jhu"
    cfg_qnrf_dir.mkdir(parents=True, exist_ok=True)
    cfg_jhu_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. UCF-QNRF Manifests & Configs
    # ========================================================
    qnrf_train_path = data_dir / "qnrf_train.jsonl"
    with open(qnrf_train_path, "r", encoding="utf-8") as f:
        qnrf_samples = [json.loads(line) for line in f if line.strip()]
    n_qnrf = len(qnrf_samples)
    print(f"Loaded {n_qnrf} QNRF training samples")

    # 1a. Data Scarcity Manifests for QNRF
    rng_qnrf = np.random.default_rng(seed=42)
    qnrf_idx = np.arange(n_qnrf)
    rng_qnrf.shuffle(qnrf_idx)

    qnrf_fractions = {"frac010": 120, "frac025": 300, "frac050": 600}
    for name, cnt in qnrf_fractions.items():
        sub = [qnrf_samples[i] for i in qnrf_idx[:cnt]]
        out_p = data_dir / f"qnrf_train_{name}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for item in sub:
                f.write(json.dumps(item) + "\n")
        print(f"Created QNRF scarcity manifest: {out_p.name} ({len(sub)} images)")

    # 1b. Noisy Label Training Manifests for QNRF
    for sigma in [8.0, 16.0]:
        rng_j = np.random.default_rng(seed=int(sigma * 100))
        noisy = []
        for s in qnrf_samples:
            pts = np.array(s.get("points", []))
            if len(pts) > 0:
                noise = rng_j.normal(loc=0.0, scale=sigma, size=pts.shape)
                noisy_pts = (pts + noise).tolist()
            else:
                noisy_pts = []
            noisy.append({"id": s.get("id"), "image": s.get("image"), "points": noisy_pts, "count": len(noisy_pts)})
        out_p = data_dir / f"qnrf_train_jitter_sigma{int(sigma)}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for item in noisy:
                f.write(json.dumps(item) + "\n")
        print(f"Created QNRF jitter manifest: {out_p.name} (sigma={sigma}px)")

    for rho in [0.10, 0.20]:
        rng_d = np.random.default_rng(seed=int(rho * 1000))
        dropped = []
        for s in qnrf_samples:
            pts = np.array(s.get("points", []))
            if len(pts) > 0:
                keep = rng_d.uniform(size=len(pts)) >= rho
                kept_pts = pts[keep].tolist()
            else:
                kept_pts = []
            dropped.append({"id": s.get("id"), "image": s.get("image"), "points": kept_pts, "count": len(kept_pts)})
        out_p = data_dir / f"qnrf_train_drop_rho{int(rho * 100)}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for item in dropped:
                f.write(json.dumps(item) + "\n")
        print(f"Created QNRF drop manifest: {out_p.name} (rho={rho*100:.0f}%)")

    # Load base QNRF config
    base_qnrf_p = _REPO_ROOT / "configs" / "benchmarks" / "exp_qnrf_direct_pretrained_104k.yaml"
    with open(base_qnrf_p, "r", encoding="utf-8") as f:
        base_qnrf_cfg = yaml.safe_load(f)

    # Generate QNRF YAMLs
    qnrf_configs = []
    # Scarcity
    for name in ["frac010", "frac025", "frac050"]:
        c = yaml.safe_load(yaml.dump(base_qnrf_cfg))
        c["data"]["train_manifest"] = f"data/qnrf_train_{name}.jsonl"
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_qnrf_data_{name}_104k"
        qnrf_configs.append((cfg_qnrf_dir / f"exp_qnrf_data_{name}_104k.yaml", c))

    # Multi-seed
    for seed in [1337, 2024]:
        c = yaml.safe_load(yaml.dump(base_qnrf_cfg))
        c["seed"] = seed
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_qnrf_seed_{seed}_104k"
        qnrf_configs.append((cfg_qnrf_dir / f"exp_qnrf_seed_{seed}_104k.yaml", c))

    # Noisy labels
    for sigma in [8, 16]:
        c = yaml.safe_load(yaml.dump(base_qnrf_cfg))
        c["data"]["train_manifest"] = f"data/qnrf_train_jitter_sigma{sigma}.jsonl"
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_qnrf_jitter_sigma{sigma}_104k"
        qnrf_configs.append((cfg_qnrf_dir / f"exp_qnrf_jitter_sigma{sigma}_104k.yaml", c))

    for rho in [10, 20]:
        c = yaml.safe_load(yaml.dump(base_qnrf_cfg))
        c["data"]["train_manifest"] = f"data/qnrf_train_drop_rho{rho}.jsonl"
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_qnrf_drop_rho{rho}_104k"
        qnrf_configs.append((cfg_qnrf_dir / f"exp_qnrf_drop_rho{rho}_104k.yaml", c))

    # Crop sizes
    for crop in [384, 768]:
        c = yaml.safe_load(yaml.dump(base_qnrf_cfg))
        c["data"]["crop_size"] = crop
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_qnrf_crop_{crop}_104k"
        qnrf_configs.append((cfg_qnrf_dir / f"exp_qnrf_crop_{crop}_104k.yaml", c))

    for p, cfg_obj in qnrf_configs:
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_obj, f, sort_keys=False)
        print(f"Generated QNRF YAML: {p.relative_to(_REPO_ROOT)}")

    # ========================================================
    # 2. JHU-CROWD++ Manifests & Configs
    # ========================================================
    jhu_train_path = data_dir / "jhu_train.jsonl"
    with open(jhu_train_path, "r", encoding="utf-8") as f:
        jhu_samples = [json.loads(line) for line in f if line.strip()]
    n_jhu = len(jhu_samples)
    print(f"\nLoaded {n_jhu} JHU training samples")

    # 2a. Data Scarcity Manifests for JHU
    rng_jhu = np.random.default_rng(seed=42)
    jhu_idx = np.arange(n_jhu)
    rng_jhu.shuffle(jhu_idx)

    jhu_fractions = {"frac010": 227, "frac025": 568, "frac050": 1136}
    for name, cnt in jhu_fractions.items():
        sub = [jhu_samples[i] for i in jhu_idx[:cnt]]
        out_p = data_dir / f"jhu_train_{name}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for item in sub:
                f.write(json.dumps(item) + "\n")
        print(f"Created JHU scarcity manifest: {out_p.name} ({len(sub)} images)")

    # Load base JHU config
    base_jhu_p = _REPO_ROOT / "configs" / "benchmarks" / "exp_jhu_direct_pretrained_104k.yaml"
    with open(base_jhu_p, "r", encoding="utf-8") as f:
        base_jhu_cfg = yaml.safe_load(f)

    # Generate JHU YAMLs
    jhu_configs = []
    # Scarcity
    for name in ["frac010", "frac025", "frac050"]:
        c = yaml.safe_load(yaml.dump(base_jhu_cfg))
        c["data"]["train_manifest"] = f"data/jhu_train_{name}.jsonl"
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_jhu_data_{name}_104k"
        jhu_configs.append((cfg_jhu_dir / f"exp_jhu_data_{name}_104k.yaml", c))

    # Multi-seed
    for seed in [1337, 2024]:
        c = yaml.safe_load(yaml.dump(base_jhu_cfg))
        c["seed"] = seed
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_jhu_seed_{seed}_104k"
        jhu_configs.append((cfg_jhu_dir / f"exp_jhu_seed_{seed}_104k.yaml", c))

    for p, cfg_obj in jhu_configs:
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_obj, f, sort_keys=False)
        print(f"Generated JHU YAML: {p.relative_to(_REPO_ROOT)}")

    print(f"\nSuccessfully generated {len(qnrf_configs)} QNRF configs and {len(jhu_configs)} JHU configs!")


if __name__ == "__main__":
    main()
