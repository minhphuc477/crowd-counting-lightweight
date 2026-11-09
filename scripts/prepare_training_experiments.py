#!/usr/bin/env python3
"""Prepare manifests and configurations for Top-Tier Crowd Counting TRAINING Experiments.

Covers the 5 canonical TRAINING sweeps demanded by CVPR / NeurIPS / TPAMI:
  1. Data Efficiency / Label Scarcity: 10% (30 imgs), 25% (75 imgs), 50% (150 imgs).
  2. Noisy Label Training: Gaussian jitter sigma in {4, 8} px, Dropout rho in {10%, 20%}.
  3. Loss Weight Sensitivity: lambda_count in {0.1, 1.0, 5.0}, lambda_radon in {0.2, 2.0}.
  4. Multi-Seed Statistical Variance: Seeds {1337, 2024} (to complement nominal seed 42).
  5. Training Receptive Field / Crop Size: Crop 256, Crop 384 (to complement nominal crop 512).

Strict compliance with AGENTS.md: Zero buzzwords. Pure empirical reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    src_manifest = _REPO_ROOT / "data" / "sha_a_train_all.jsonl"
    with open(src_manifest, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} training samples from {src_manifest}")

    data_dir = _REPO_ROOT / "data"
    cfg_base_dir = _REPO_ROOT / "configs" / "training_sweeps"
    cfg_base_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # 1. Data Efficiency Manifests (10%, 25%, 50%)
    # ----------------------------------------------------
    rng_split = np.random.default_rng(seed=42)
    indices = np.arange(len(samples))
    rng_split.shuffle(indices)

    fractions = {"frac010": 30, "frac025": 75, "frac050": 150}
    for name, count in fractions.items():
        subset_samples = [samples[i] for i in indices[:count]]
        out_p = data_dir / f"sha_a_train_{name}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for s in subset_samples:
                f.write(json.dumps(s) + "\n")
        print(f"Created data efficiency manifest: {out_p} ({len(subset_samples)} images)")

    # ----------------------------------------------------
    # 2. Noisy Label Training Manifests
    # ----------------------------------------------------
    # 2a. Gaussian click jitter: sigma in {4.0, 8.0} px
    for sigma in [4.0, 8.0]:
        rng_jitter = np.random.default_rng(seed=int(sigma * 100))
        noisy_samples = []
        for s in samples:
            pts = np.array(s.get("points", []))
            if len(pts) > 0:
                noise = rng_jitter.normal(loc=0.0, scale=sigma, size=pts.shape)
                noisy_pts = (pts + noise).tolist()
            else:
                noisy_pts = []
            noisy_samples.append({
                "id": s.get("id"),
                "image": s.get("image"),
                "points": noisy_pts,
                "count": len(noisy_pts),
            })
        out_p = data_dir / f"sha_a_train_jitter_sigma{int(sigma)}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for item in noisy_samples:
                f.write(json.dumps(item) + "\n")
        print(f"Created noisy label jitter manifest: {out_p} (sigma={sigma}px)")

    # 2b. Missing Annotation Dropout: rho in {0.10, 0.20}
    for rho in [0.10, 0.20]:
        rng_drop = np.random.default_rng(seed=int(rho * 1000))
        dropped_samples = []
        for s in samples:
            pts = np.array(s.get("points", []))
            if len(pts) > 0:
                keep = rng_drop.uniform(size=len(pts)) >= rho
                kept_pts = pts[keep].tolist()
            else:
                kept_pts = []
            dropped_samples.append({
                "id": s.get("id"),
                "image": s.get("image"),
                "points": kept_pts,
                "count": len(kept_pts),
            })
        out_p = data_dir / f"sha_a_train_drop_rho{int(rho * 100)}.jsonl"
        with open(out_p, "w", encoding="utf-8") as f:
            for item in dropped_samples:
                f.write(json.dumps(item) + "\n")
        print(f"Created missing label drop manifest: {out_p} (drop_rate={rho*100:.0f}%)")

    # ----------------------------------------------------
    # 3. Base Nominal Config Reference
    # ----------------------------------------------------
    base_cfg_path = _REPO_ROOT / "configs" / "benchmarks" / "exp_sha_direct_pretrained_104k.yaml"
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # ----------------------------------------------------
    # 4. Generate YAML Configurations for All 5 Sweeps
    # ----------------------------------------------------
    configs_to_generate = []

    # Sweep 1: Data Efficiency
    for name in ["frac010", "frac025", "frac050"]:
        c = yaml.safe_load(yaml.dump(base_cfg))
        c["data"]["train_manifest"] = f"data/sha_a_train_{name}.jsonl"
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_sha_data_{name}_104k"
        c_path = cfg_base_dir / f"exp_sha_data_{name}_104k.yaml"
        configs_to_generate.append((c_path, c))

    # Sweep 2: Noisy Label Training
    for sigma in [4, 8]:
        c = yaml.safe_load(yaml.dump(base_cfg))
        c["data"]["train_manifest"] = f"data/sha_a_train_jitter_sigma{sigma}.jsonl"
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_sha_jitter_sigma{sigma}_104k"
        c_path = cfg_base_dir / f"exp_sha_jitter_sigma{sigma}_104k.yaml"
        configs_to_generate.append((c_path, c))

    for rho in [10, 20]:
        c = yaml.safe_load(yaml.dump(base_cfg))
        c["data"]["train_manifest"] = f"data/sha_a_train_drop_rho{rho}.jsonl"
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_sha_drop_rho{rho}_104k"
        c_path = cfg_base_dir / f"exp_sha_drop_rho{rho}_104k.yaml"
        configs_to_generate.append((c_path, c))

    # Sweep 3: Multi-Seed Variance
    for seed in [1337, 2024]:
        c = yaml.safe_load(yaml.dump(base_cfg))
        c["seed"] = seed
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_sha_seed_{seed}_104k"
        c_path = cfg_base_dir / f"exp_sha_seed_{seed}_104k.yaml"
        configs_to_generate.append((c_path, c))

    # Sweep 4: Loss Weight Sensitivity (Trained)
    # 4a. Count loss weight
    for l_count in [0.1, 1.0, 5.0]:
        c = yaml.safe_load(yaml.dump(base_cfg))
        c["loss"]["lambda_count"] = l_count
        tag = str(l_count).replace(".", "p")
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_sha_loss_count_{tag}_104k"
        c_path = cfg_base_dir / f"exp_sha_loss_count_{tag}_104k.yaml"
        configs_to_generate.append((c_path, c))

    # 4b. Radon OT loss weight
    for l_radon in [0.2, 2.0]:
        c = yaml.safe_load(yaml.dump(base_cfg))
        c["loss"]["lambda_radon_ot"] = l_radon
        tag = str(l_radon).replace(".", "p")
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_sha_loss_radon_{tag}_104k"
        c_path = cfg_base_dir / f"exp_sha_loss_radon_{tag}_104k.yaml"
        configs_to_generate.append((c_path, c))

    # Sweep 5: Crop Size Sensitivity during Training
    for crop in [256, 384]:
        c = yaml.safe_load(yaml.dump(base_cfg))
        c["data"]["crop_size"] = crop
        c["output_dir"] = f"runs/benchmarks/sweeps_train/exp_sha_crop_{crop}_104k"
        c_path = cfg_base_dir / f"exp_sha_crop_{crop}_104k.yaml"
        configs_to_generate.append((c_path, c))

    # Write all config files
    for p, cfg_obj in configs_to_generate:
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_obj, f, sort_keys=False)
        print(f"Generated YAML config: {p.relative_to(_REPO_ROOT)}")

    print(f"\nSuccessfully generated {len(configs_to_generate)} training experiment configurations!")


if __name__ == "__main__":
    main()
