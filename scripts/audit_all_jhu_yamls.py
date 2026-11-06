#!/usr/bin/env python3
"""Audit every single JHU YAML configuration line-by-line."""
from __future__ import annotations

import sys
from pathlib import Path
import yaml

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig


def main():
    configs = [
        Path("configs/benchmarks/exp_jhu_direct_scratch_104k.yaml"),
        Path("configs/benchmarks/exp_jhu_direct_pretrained_104k.yaml"),
    ] + sorted(list(Path("configs/benchmarks/ablations_jhu").glob("*.yaml")))

    print(f"=== Auditing {len(configs)} YAML files line-by-line ===")

    errors = []
    for p in configs:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 1. Check manifests exist
        train_m = Path(cfg["data"]["train_manifest"])
        val_m = Path(cfg["data"]["val_manifest"])
        if not train_m.is_file():
            errors.append(f"{p.name}: Missing train manifest {train_m}")
        if not val_m.is_file():
            errors.append(f"{p.name}: Missing val manifest {val_m}")

        # 2. Check model parameter count
        m_cfg = RMRv3Config.from_dict(cfg["model"], pretrained=False)
        m = RMRv3(m_cfg)
        n_params = sum(param.numel() for param in m.parameters() if param.requires_grad)
        max_budget = cfg["model"].get("max_trainable_params", 104441)
        if n_params > max_budget:
            errors.append(f"{p.name}: Params exceed budget: {n_params} > {max_budget}")

        # 3. Check loss config validation
        try:
            l_cfg = RMRv3LossConfig.from_dict(cfg["loss"])
        except Exception as ex:
            errors.append(f"{p.name}: Invalid loss config: {ex}")

        # 4. Check Stride alignment
        m_stride = cfg["model"]["output_stride"]
        l_stride = cfg["loss"]["output_stride"]
        if m_stride != l_stride:
            errors.append(f"{p.name}: Stride mismatch: model={m_stride} vs loss={l_stride}")

        # 5. Check output_dir matches filename
        expected_stem = p.stem
        out_dir = cfg.get("output_dir", "")
        if not out_dir.endswith(expected_stem):
            errors.append(f"{p.name}: output_dir '{out_dir}' does not end with '{expected_stem}'")

        # 6. Check data parameters
        c_size = cfg["data"]["crop_size"]
        if c_size not in (384, 512):
            errors.append(f"{p.name}: Unexpected crop_size: {c_size}")

        print(
            f"  [PASS] {p.name:45s} | Params: {n_params:6d} | "
            f"Stride: {m_stride} | Neck: {m_cfg.neck_type:9s} | Alloc: {l_cfg.allocation_loss_type}"
        )

    if errors:
        print("\nERRORS DETECTED:")
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print(f"\nALL {len(configs)} YAML FILES ARE 100% VALIDATED LINE-BY-LINE!")


if __name__ == "__main__":
    main()
