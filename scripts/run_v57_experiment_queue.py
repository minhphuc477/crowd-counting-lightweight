#!/usr/bin/env python3
"""Automated Experiment Queue Runner for RMR V57 on Remote GPU Server.

Manages sequential execution of V57 Breakthrough Experiments:
- Wave 1: Proposed Multi-Step Radon-Nikodym with Faithful Beta-NB (Track A + Track B)
- Wave 2: 1-Step Ablation Controls for Contraction Mapping Verification (Track A + Track B)

Usage:
    python scripts/run_v57_experiment_queue.py --wave 1
    python scripts/run_v57_experiment_queue.py --all
    python scripts/run_v57_experiment_queue.py --dry-run
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


WAVE_1_JOBS: List[Dict[str, Any]] = [
    {
        "id": "v57_track_a_proposed_multistep_rn",
        "config": "configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml",
        "priority": 1,
        "description": "Track A Proposed: 2-Step Radon-Nikodym with Faithful Beta-NB (103,959 params <= 104,441)",
    },
    {
        "id": "v57_track_b_proposed_deep_texture_rn",
        "config": "configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml",
        "priority": 1,
        "description": "Track B Proposed: Subpixel Stride 2 Deep Texture RN with Faithful Beta-NB (144,960 params <= 200,000)",
    },
]

WAVE_2_JOBS: List[Dict[str, Any]] = [
    {
        "id": "v57_track_a_ablation_1step_rn",
        "config": "configs/rmr_v57/v57_track_a_ablation_1step_rn.yaml",
        "priority": 2,
        "description": "Track A Ablation: 1-Step RN Baseline (103,959 params <= 104,441)",
    },
    {
        "id": "v57_track_b_ablation_1step_rn",
        "config": "configs/rmr_v57/v57_track_b_ablation_1step_rn.yaml",
        "priority": 2,
        "description": "Track B Ablation: Subpixel Stride 2 1-Step RN Baseline (144,960 params <= 200,000)",
    },
]

LEDGER_PATH = "EXPERIMENT_RESULTS_V57.jsonl"


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
    desc = job["description"]

    print("\n" + "=" * 80)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING JOB: {job_id}")
    print(f"  Description: {desc}")
    print(f"  Config:      {cfg_path}")
    print("=" * 80 + "\n")

    if not os.path.exists(cfg_path):
        print(f"ERROR: Config file not found: {cfg_path}")
        return False

    if dry_run:
        print("[DRY-RUN] Command: python train.py --config " + cfg_path)
        return True

    cmd = [sys.executable, "train.py", "--config", cfg_path]
    env = os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "2")

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        ret = proc.wait()
        duration_sec = time.time() - start_time

        if ret == 0:
            print(f"\n[JOB COMPLETED SUCCESSFULLY] {job_id} in {duration_sec / 3600:.2f} hours")
            append_ledger({
                "job_id": job_id,
                "config": cfg_path,
                "status": "SUCCESS",
                "duration_seconds": duration_sec,
            })
            return True
        else:
            print(f"\n[JOB FAILED] {job_id} with exit code {ret} after {duration_sec / 60:.1f} minutes")
            append_ledger({
                "job_id": job_id,
                "config": cfg_path,
                "status": "FAILED",
                "exit_code": ret,
                "duration_seconds": duration_sec,
            })
            return False
    except Exception as exc:
        duration_sec = time.time() - start_time
        print(f"\n[JOB CRASHED] {job_id}: {exc}")
        append_ledger({
            "job_id": job_id,
            "config": cfg_path,
            "status": "CRASHED",
            "error": str(exc),
            "duration_seconds": duration_sec,
        })
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="RMR V57 Automated Experiment Runner")
    parser.add_argument(
        "--wave",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run specific experiment wave (1=Proposed Breakthroughs, 2=Ablations)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all waves sequentially (Wave 1 then Wave 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without starting training processes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if job has already completed",
    )
    args = parser.parse_args()

    jobs_to_run: List[Dict[str, Any]] = []
    if args.wave == 1:
        jobs_to_run = list(WAVE_1_JOBS)
    elif args.wave == 2:
        jobs_to_run = list(WAVE_2_JOBS)
    elif args.all:
        jobs_to_run = list(WAVE_1_JOBS) + list(WAVE_2_JOBS)
    else:
        # Default to Wave 1
        print("No wave specified; defaulting to Wave 1 (Proposed Breakthrough Runs).")
        jobs_to_run = list(WAVE_1_JOBS)

    print(f"Loaded {len(jobs_to_run)} jobs into execution queue.")

    passed = 0
    failed = 0
    skipped = 0

    for idx, job in enumerate(jobs_to_run, start=1):
        job_id = job["id"]
        print(f"\n>>> Queue Item [{idx}/{len(jobs_to_run)}]: {job_id}")

        if not args.force and is_job_completed(job):
            print(f"Skipping {job_id}: already completed (best.pth exists). Use --force to rerun.")
            skipped += 1
            continue

        success = run_job(job, dry_run=args.dry_run)
        if success:
            passed += 1
        else:
            failed += 1
            print(f"WARNING: Job {job_id} encountered an issue. Proceeding to next job in queue.")

    print("\n" + "=" * 80)
    print("V57 EXPERIMENT QUEUE RUN SUMMARY:")
    print(f"  Total Queued: {len(jobs_to_run)}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {failed}")
    print(f"  Skipped:      {skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
