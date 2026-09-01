#!/usr/bin/env python3
"""Automated pipeline to run Factorial Cell B and Cell C, evaluate localization & D0 diagnostics,
and aggregate into the unified 2x2 Factorial Matrix.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


RUNS = [
    {
        "cell": "B",
        "name": "ntpc_b_crop256_c32",
        "config": "configs/ntpc_b_crop256_c32.yaml",
        "checkpoint": "runs/ntpc_b_crop256_c32/best.pt",
        "loc_output": "runs/ntpc_b_crop256_c32/localization_eval.json",
        "d0_output": "runs/d0_diagnostics_b.json",
    },
    {
        "cell": "C",
        "name": "ntpc_c_crop448_c16",
        "config": "configs/ntpc_c_crop448_c16.yaml",
        "checkpoint": "runs/ntpc_c_crop448_c16/best.pt",
        "loc_output": "runs/ntpc_c_crop448_c16/localization_eval.json",
        "d0_output": "runs/d0_diagnostics_c.json",
    },
]


def run_cmd(cmd: list[str]) -> None:
    print(f"\n[RUNNING] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[FAILED] Command exited with code {res.returncode}", flush=True)
        sys.exit(res.returncode)
    print(f"[COMPLETED in {time.time() - t0:.1f}s]\n", flush=True)


def main() -> None:
    py = sys.executable

    for item in RUNS:
        print(f"\n{'='*70}\n  STARTING FACTORIAL CELL {item['cell']}: {item['name']}\n{'='*70}", flush=True)
        # 1. Train model
        run_cmd([py, "-u", "train_ntpc.py", "--config", item["config"], "--overwrite"])

        # 2. Evaluate localization (OT-M F1@8, F1@4)
        run_cmd([
            py, "tools/eval_localization.py",
            "--config", item["config"],
            "--checkpoint", item["checkpoint"],
            "--output", item["loc_output"],
        ])

        # 3. Evaluate D0 diagnostics
        run_cmd([
            py, "tools/run_d0_diagnostics.py",
            "--config", item["config"],
            "--checkpoint", item["checkpoint"],
            "--output", item["d0_output"],
        ])

    print("\n" + "="*70)
    print("  FACTORIAL 2x2 MATRIX EXPERIMENTS COMPLETED!")
    print("="*70, flush=True)


if __name__ == "__main__":
    main()
