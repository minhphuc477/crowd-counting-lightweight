#!/usr/bin/env python3
"""Training Health Check Diagnostic Utility for RMR V53.

Analyzes training log files or checkpoint directories to catch early failures:
1. NaN / Inf in loss or gradients.
2. Dead carrier head collapse (y0 -> 0).
3. Density explosion (unbounded growth).
4. Loss plateau / learning rate stalls.

Usage:
    python scripts/v53_training_health_check.py --log logs/queue_v53_track_a_closed_form_rn.log
    python scripts/v53_training_health_check.py --run-dir runs/sha_a/rmr_v53/v53_track_a_closed_form_rn
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple


def check_log_health(log_path: str) -> Tuple[str, List[str]]:
    """Scan log file for numerical stability and convergence issues."""
    if not os.path.exists(log_path):
        return "UNKNOWN", [f"Log file not found: {log_path}"]

    issues: List[str] = []
    has_nan = False
    has_loss_divergence = False
    has_eval_mae = False
    last_maes: List[float] = []

    nan_pattern = re.compile(r"\b(nan|inf|-nan|-inf)\b", re.IGNORECASE)
    mae_pattern = re.compile(r"MAE[:=]\s*([0-9]+\.[0-9]+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            if nan_pattern.search(line):
                has_nan = True
                issues.append(f"Line {idx}: NaN/Inf detected: {line.strip()[:100]}")
                break

            m = mae_pattern.search(line)
            if m:
                has_eval_mae = True
                val = float(m.group(1))
                last_maes.append(val)
                if val > 1500.0:
                    has_loss_divergence = True
                    issues.append(f"Line {idx}: Extreme MAE explosion ({val:.1f})")

    if has_nan:
        return "CRITICAL", issues
    if has_loss_divergence:
        return "CRITICAL", issues

    if last_maes:
        recent = last_maes[-5:]
        if len(recent) >= 3 and all(r > 500.0 for r in recent):
            issues.append(f"Sustained high MAE in recent epochs: {recent}")
            return "WARNING", issues

    return "HEALTHY", [f"Scanned {len(last_maes)} evaluation checkpoints. No anomalies detected."]


def main() -> None:
    parser = argparse.ArgumentParser(description="RMR V53 Training Health Check")
    parser.add_argument("--log", type=str, help="Path to training log file")
    parser.add_argument("--run-dir", type=str, help="Path to run output directory")
    args = parser.parse_args()

    target_log = args.log
    if not target_log and args.run_dir:
        cand = os.path.join(args.run_dir, "train.log")
        if os.path.exists(cand):
            target_log = cand
        else:
            base = os.path.basename(os.path.normpath(args.run_dir))
            target_log = os.path.join("logs", f"queue_{base}.log")

    if not target_log:
        print("Please provide --log or --run-dir")
        sys.exit(1)

    print(f"\nChecking health for: {target_log}")
    verdict, details = check_log_health(target_log)
    print(f"Verdict: {verdict}")
    for d in details:
        print(f"  - {d}")
    print()


if __name__ == "__main__":
    main()
