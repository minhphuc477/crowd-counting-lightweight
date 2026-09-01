#!/usr/bin/env python3
"""Automated pipeline to run Factorial Matrix A/B/C/D (Crop 256/448 x C16/C32),
evaluate localization & D0 diagnostics, and aggregate into the unified 2x2 Factorial Matrix.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

RUNS = [
    ("A", "factorial_a_crop256_c16"),
    ("B", "factorial_b_crop256_c32"),
    ("C", "factorial_c_crop448_c16"),
    ("D", "factorial_d_crop448_c32"),
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
    parser = argparse.ArgumentParser(description="Run complete Factorial Matrix A/B/C/D")
    parser.add_argument("--cells", default="all", help="Comma-separated cells to run (e.g. 'A,B,C,D' or 'B,C')")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runs")
    args = parser.parse_args()

    py = sys.executable
    selected_cells = {c.strip().upper() for c in args.cells.split(",")} if args.cells.lower() != "all" else {"A", "B", "C", "D"}

    for cell, name in RUNS:
        if cell not in selected_cells:
            continue
        config = f"configs/{name}.yaml"
        run_dir = f"runs/{name}"
        checkpoint = f"{run_dir}/best.pt"

        print(f"\n{'='*70}\n  STARTING FACTORIAL CELL {cell}: {name}\n{'='*70}", flush=True)
        
        train_cmd = [py, "-u", "train_ntpc.py", "--config", config]
        if args.overwrite:
            train_cmd.append("--overwrite")
        run_cmd(train_cmd)

        run_cmd([
            py, "tools/eval_localization.py",
            "--config", config,
            "--checkpoint", checkpoint,
            "--output", f"{run_dir}/localization_eval.json",
        ])

        run_cmd([
            py, "tools/run_d0_diagnostics.py",
            "--config", config,
            "--checkpoint", checkpoint,
            "--output", f"{run_dir}/d0_diagnostics.json",
        ])

    print("\n" + "="*70)
    print("  SELECTED FACTORIAL MATRIX EXPERIMENTS COMPLETED!")
    print("="*70, flush=True)


if __name__ == "__main__":
    main()
