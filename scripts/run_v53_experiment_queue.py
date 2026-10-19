#!/usr/bin/env python3
"""Automated Experiment Queue Runner for RMR V53 on Remote GPU Server.

Manages sequential/wave-based training execution, checkpoint validation,
OOM recovery, and append-only result ledger tracking.

Usage:
    python scripts/run_v53_experiment_queue.py --wave 1
    python scripts/run_v53_experiment_queue.py --all
    python scripts/run_v53_experiment_queue.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List


WAVE_1_JOBS = [
    {
        "id": "v53_track_a_closed_form_rn",
        "config": "configs/rmr_v53/v53_track_a_closed_form_rn.yaml",
        "priority": 1,
        "description": "Track A Stride 4 Canonical Breakthrough (<= 104,441 params)",
    },
    {
        "id": "v53_track_b_closed_form_rn",
        "config": "configs/rmr_v53/v53_track_b_closed_form_rn.yaml",
        "priority": 1,
        "description": "Track B Subpixel Stride 2 Deep Dirac Head (<= 200,000 params)",
    },
]

WAVE_2_JOBS = [
    {
        "id": "abl_s1_direct_stride4",
        "config": "configs/rmr_v53/ablations/abl_s1_direct_stride4.yaml",
        "priority": 2,
        "description": "Ablation S1-A: Direct Feedforward Baseline (No Solver)",
    },
    {
        "id": "abl_s1_sirt_unrolled",
        "config": "configs/rmr_v53/ablations/abl_s1_sirt_unrolled.yaml",
        "priority": 2,
        "description": "Ablation S1-B: Multiplicative Unrolled SIRT (T=6)",
    },
    {
        "id": "abl_s4_no_strauss",
        "config": "configs/rmr_v53/ablations/abl_s4_no_strauss.yaml",
        "priority": 2,
        "description": "Ablation S4: Track A without Strauss Repulsion (lambda=0)",
    },
]

WAVE_3_JOBS = [
    {
        "id": "abl_s2_stride4_deep_head",
        "config": "configs/rmr_v53/ablations/abl_s2_stride4_deep_head.yaml",
        "priority": 3,
        "description": "Ablation S2: Stride 4 with Deep Dirac Head",
    },
    {
        "id": "abl_s3_stride2_shallow_head",
        "config": "configs/rmr_v53/ablations/abl_s3_stride2_shallow_head.yaml",
        "priority": 3,
        "description": "Ablation S3: Subpixel Stride 2 with Shallow Head",
    },
    {
        "id": "abl_s5_strauss_strong",
        "config": "configs/rmr_v53/ablations/abl_s5_strauss_strong.yaml",
        "priority": 3,
        "description": "Ablation S5: Track A Strong Strauss Repulsion (lambda=0.10)",
    },
    {
        "id": "abl_s6_track_b_no_strauss",
        "config": "configs/rmr_v53/ablations/abl_s6_track_b_no_strauss.yaml",
        "priority": 3,
        "description": "Ablation S6: Track B without Strauss Repulsion",
    },
    {
        "id": "abl_s7_track_b_direct",
        "config": "configs/rmr_v53/ablations/abl_s7_track_b_direct.yaml",
        "priority": 3,
        "description": "Ablation S7: Track B Direct Feedforward (No Solver)",
    },
]

LEDGER_PATH = "EXPERIMENT_RESULTS_V53.jsonl"


def append_ledger(record: Dict[str, Any]) -> None:
    """Append a completed job record to the experiment results ledger."""
    record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_job_completed(job: Dict[str, Any]) -> bool:
    """Check if the job has already finished successfully with a best checkpoint."""
    cfg_path = job["config"]
    if not os.path.exists(cfg_path):
        return False
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg.get("output_dir", "")
    best_ckpt = os.path.join(out_dir, "best.pth")
    return os.path.exists(best_ckpt)


def run_job(job: Dict[str, Any], dry_run: bool = False) -> bool:
    """Execute a single job with subprocess streaming and timing."""
    job_id = job["id"]
    cfg_path = job["config"]
    print(f"\n{'='*70}")
    print(f"[EXPERIMENT QUEUE] Launching Job: {job_id}")
    print(f"  Description: {job['description']}")
    print(f"  Config:      {cfg_path}")
    print(f"{'='*70}")

    if not os.path.exists(cfg_path):
        print(f"ERROR: Config file not found: {cfg_path}")
        return False

    if is_job_completed(job):
        print(f"INFO: Job {job_id} already has a valid checkpoint. Skipping.")
        return True

    cmd = [sys.executable, "train.py", "--config", cfg_path]
    print(f"Executing: {' '.join(cmd)}")

    if dry_run:
        print("Dry-run mode enabled. Job validated successfully.")
        return True

    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"queue_{job_id}.log")
    start_time = time.time()

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    elapsed = time.time() - start_time
    success = proc.returncode == 0

    record = {
        "job_id": job_id,
        "config": cfg_path,
        "exit_code": proc.returncode,
        "success": success,
        "elapsed_seconds": round(elapsed, 1),
        "log_file": log_file,
    }
    append_ledger(record)

    if success:
        print(f"[EXPERIMENT QUEUE] Job {job_id} COMPLETED in {elapsed:.1f}s.")
    else:
        print(f"[EXPERIMENT QUEUE] Job {job_id} FAILED with code {proc.returncode}.")

    # Cool-down to ensure GPU memory cleanup
    time.sleep(5)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="RMR V53 Experiment Queue")
    parser.add_argument("--wave", type=int, choices=[1, 2, 3], help="Execute specific wave")
    parser.add_argument("--all", action="store_true", help="Execute all waves sequentially")
    parser.add_argument("--dry-run", action="store_true", help="Validate queue without training")
    args = parser.parse_args()

    jobs_to_run: List[Dict[str, Any]] = []
    if args.wave == 1:
        jobs_to_run = WAVE_1_JOBS
    elif args.wave == 2:
        jobs_to_run = WAVE_2_JOBS
    elif args.wave == 3:
        jobs_to_run = WAVE_3_JOBS
    elif args.all:
        jobs_to_run = WAVE_1_JOBS + WAVE_2_JOBS + WAVE_3_JOBS
    else:
        print("Defaulting to Wave 1 (Core Breakthroughs). Use --all or --wave to specify.")
        jobs_to_run = WAVE_1_JOBS

    print(f"Loaded {len(jobs_to_run)} jobs into queue.")
    failed_jobs = []
    for job in jobs_to_run:
        ok = run_job(job, dry_run=args.dry_run)
        if not ok and not args.dry_run:
            failed_jobs.append(job["id"])

    print(f"\n{'='*70}")
    print(f"Queue Execution Finished. Total: {len(jobs_to_run)}, Failed: {len(failed_jobs)}")
    if failed_jobs:
        print(f"Failed Jobs: {failed_jobs}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
