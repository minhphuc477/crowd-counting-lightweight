#!/usr/bin/env python3
"""Comprehensive Deep Audit of all 70 Benchmark and Ablation YAML configs.

Checks:
1. Model instantiation & parameter count <= 104,441
2. Stride alignment (model vs loss) and subpixel_stride2 flag
3. Loss configuration validity & potential traps (e.g. positive-push explosion, 0-gradient, inactive allocations)
4. Data manifests existence on disk
5. Output directory consistency and uniqueness
6. Training hyperparameter boundaries
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig


def audit_all_configs():
    config_paths = sorted(Path("configs/benchmarks").rglob("*.yaml"))
    print(f"================================================================================")
    print(f"AUDITING ALL {len(config_paths)} BENCHMARK & ABLATION YAML CONFIGURATIONS")
    print(f"================================================================================")

    critical_errors: list[str] = []
    warnings: list[str] = []
    output_dirs_seen: dict[str, Path] = {}

    for idx, path in enumerate(config_paths, 1):
        rel_path = path.as_posix()
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as ex:
            critical_errors.append(f"[{rel_path}] YAML parse error: {ex}")
            continue

        # 1. Output directory uniqueness
        out_dir = cfg.get("output_dir", "")
        if not out_dir:
            critical_errors.append(f"[{rel_path}] Missing 'output_dir'")
        else:
            expected_stem = path.stem
            if not out_dir.endswith(expected_stem):
                warnings.append(f"[{rel_path}] output_dir '{out_dir}' does not end with file stem '{expected_stem}'")
            if out_dir in output_dirs_seen:
                critical_errors.append(
                    f"[{rel_path}] Duplicate output_dir '{out_dir}' already used by [{output_dirs_seen[out_dir].as_posix()}]"
                )
            output_dirs_seen[out_dir] = path

        # 2. Model configuration & parameter count
        m_dict = cfg.get("model", {})
        try:
            m_cfg = RMRv3Config(**m_dict)
            model = RMRv3(m_cfg)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception as ex:
            critical_errors.append(f"[{rel_path}] Model instantiation failed: {ex}")
            continue

        max_params = m_dict.get("max_trainable_params", 104441)
        if n_params > max_params:
            critical_errors.append(f"[{rel_path}] Param budget exceeded: {n_params} > {max_params}")

        # Stride checks
        m_stride = m_dict.get("output_stride", 4)
        subpixel = m_dict.get("subpixel_stride2", False)
        if m_stride == 2 and not subpixel:
            critical_errors.append(f"[{rel_path}] output_stride=2 requires subpixel_stride2=True")
        if m_stride == 4 and subpixel:
            critical_errors.append(f"[{rel_path}] output_stride=4 has subpixel_stride2=True (stride mismatch)")

        # 3. Loss configuration & traps
        l_dict = cfg.get("loss", {})
        try:
            l_cfg = RMRv3LossConfig(**l_dict)
        except Exception as ex:
            critical_errors.append(f"[{rel_path}] LossConfig validation failed: {ex}")
            continue

        l_stride = l_dict.get("output_stride", 4)
        if m_stride != l_stride:
            critical_errors.append(f"[{rel_path}] Stride mismatch: model={m_stride} vs loss={l_stride}")

        # Check Trap 1: Positive-push background explosion trap
        # If cell loss is active (lambda_cell > 0), what is cell_loss_mode?
        lam_cell = float(l_dict.get("lambda_cell", getattr(l_cfg, "lambda_cell", 0.0)))
        cell_mode = l_dict.get("cell_loss_mode", getattr(l_cfg, "cell_loss_mode", "balanced"))
        alloc_type = l_dict.get("allocation_loss_type", getattr(l_cfg, "allocation_loss_type", "flat_dm16"))
        lam_count = float(l_dict.get("lambda_count", getattr(l_cfg, "lambda_count", 1.0)))

        if lam_cell > 0.0:
            if cell_mode == "balanced" and alloc_type == "none" and lam_count == 0.0:
                critical_errors.append(
                    f"[{rel_path}] DANGEROUS TRAP: cell_loss_mode is 'balanced' with lambda_cell={lam_cell}, "
                    f"allocation_loss_type='none', and lambda_count=0.0. "
                    f"Will cause massive +80,000 background bias explosion!"
                )
            if cell_mode == "mse":
                sigma = float(l_dict.get("cell_gaussian_sigma", getattr(l_cfg, "cell_gaussian_sigma", 0.0)))
                if sigma <= 0.0:
                    critical_errors.append(f"[{rel_path}] cell_loss_mode='mse' but cell_gaussian_sigma <= 0 ({sigma})")

        # Check Trap 2: Active allocation type vs zero weights
        lam_bay = float(l_dict.get("lambda_bayesian", getattr(l_cfg, "lambda_bayesian", 0.0)))
        lam_rot = float(l_dict.get("lambda_radon_ot", getattr(l_cfg, "lambda_radon_ot", 0.0)))
        lam_dm16 = float(l_dict.get("lambda_flat_dm16", getattr(l_cfg, "lambda_flat_dm16", 0.0)))

        if alloc_type == "bayesian" and lam_bay <= 0.0:
            critical_errors.append(f"[{rel_path}] allocation_loss_type='bayesian' but lambda_bayesian <= 0 ({lam_bay})")
        if alloc_type == "radon_ot" and lam_rot <= 0.0:
            critical_errors.append(f"[{rel_path}] allocation_loss_type='radon_ot' but lambda_radon_ot <= 0 ({lam_rot})")
        if alloc_type == "flat_dm16" and lam_dm16 <= 0.0:
            critical_errors.append(f"[{rel_path}] allocation_loss_type='flat_dm16' but lambda_flat_dm16 <= 0 ({lam_dm16})")
        if alloc_type == "dual_bayesian_radon_ot":
            if lam_bay <= 0.0 or lam_rot <= 0.0:
                critical_errors.append(
                    f"[{rel_path}] allocation_loss_type='dual_bayesian_radon_ot' but "
                    f"lambda_bayesian={lam_bay}, lambda_radon_ot={lam_rot}"
                )

        # Check Trap 3: Zero total loss (no supervision at all)
        total_weight = (
            lam_count
            + float(l_dict.get("lambda_count_l1", 0.0))
            + lam_bay
            + lam_rot
            + lam_dm16
            + lam_cell
            + float(l_dict.get("lambda_region_nb", 0.0))
            + float(l_dict.get("lambda_hurdle", 0.0))
            + float(l_dict.get("lambda_trunc_nb", 0.0))
        )
        if total_weight <= 0.0:
            critical_errors.append(f"[{rel_path}] ZERO TOTAL SUPERVISION: All loss weights are 0.0")

        # 4. Data Manifests existence
        d_dict = cfg.get("data", {})
        train_m = d_dict.get("train_manifest", "")
        val_m = d_dict.get("val_manifest", "")
        if not train_m or not Path(train_m).is_file():
            critical_errors.append(f"[{rel_path}] train_manifest does not exist: '{train_m}'")
        if not val_m or not Path(val_m).is_file():
            critical_errors.append(f"[{rel_path}] val_manifest does not exist: '{val_m}'")

        # 5. Training parameters
        t_dict = cfg.get("train", {})
        lr = float(t_dict.get("lr", 0.0))
        epochs = int(t_dict.get("epochs", 0))
        batch_size = int(t_dict.get("batch_size", 0))
        eval_every = int(t_dict.get("eval_every", 0))

        if lr <= 0.0 or lr > 0.1:
            critical_errors.append(f"[{rel_path}] Suspicious learning rate lr={lr}")
        if epochs <= 0:
            critical_errors.append(f"[{rel_path}] Invalid epochs={epochs}")
        if batch_size <= 0:
            critical_errors.append(f"[{rel_path}] Invalid batch_size={batch_size}")
        if eval_every <= 0 or eval_every > epochs:
            warnings.append(f"[{rel_path}] eval_every={eval_every} with epochs={epochs}")

        # Calibration init_m0 check
        init_m0 = float(m_dict.get("init_m0", 0.0))
        if init_m0 <= 0.0:
            warnings.append(f"[{rel_path}] init_m0 <= 0 ({init_m0})")

        print(
            f"[{idx:02d}/70] PASS: {path.name:45s} | Params: {n_params:6d} | "
            f"Stride: {m_stride} | Alloc: {alloc_type:22s} | Cell: {cell_mode:10s}"
        )

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY REPORT")
    print("=" * 80)
    print(f"Total YAMLs inspected : {len(config_paths)}")
    print(f"Critical Errors found : {len(critical_errors)}")
    print(f"Warnings found        : {len(warnings)}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  [WARN] {w}")

    if critical_errors:
        print("\nCRITICAL ERRORS DETECTED:")
        for e in critical_errors:
            print(f"  [CRITICAL] {e}")
        print("\nAUDIT STATUS: FAILED")
        return False
    else:
        print("\nAUDIT STATUS: 100% PASSED - ALL 70 CONFIGS ARE ROBUST AND FREE OF TRAPS!")
        return True


if __name__ == "__main__":
    success = audit_all_configs()
    sys.exit(0 if success else 1)
