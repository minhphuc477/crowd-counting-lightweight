from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Restrict glibc memory arenas and OpenMP threads to prevent heap explosion across parallel runs
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

# Ensure repository root is on sys.path and remove script dir to prevent shadowing stdlib modules (e.g. profile)
_script_dir = str(Path(__file__).resolve().parent)
while _script_dir in sys.path:
    sys.path.remove(_script_dir)

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Windows stdout encoding safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml

from rmr_v3.checkpoint import CheckpointManager, EMAManager
from rmr_v3.engine import (
    build_optimizer,
    evaluate_v3,
    make_loss_cfg,
    make_model,
    train_one_epoch,
)
from rmr_v3.tracking import DiagnosticTracker, LossTracker, format_dynamic_training_banner
from rmr_v3.trainer import TRAIN_LOG_FIELDNAMES, run_training_loop

__all__ = [
    "CheckpointManager",
    "DiagnosticTracker",
    "EMAManager",
    "LossTracker",
    "TRAIN_LOG_FIELDNAMES",
    "build_optimizer",
    "evaluate_v3",
    "format_dynamic_training_banner",
    "make_loss_cfg",
    "make_model",
    "run_training_loop",
    "sanitize_run_id",
    "train_one_epoch",
]


def sanitize_run_id(run_id: str) -> str:
    """Sanitize run_id to prevent directory traversal attacks (CWE-22)."""
    clean = run_id.strip()
    if not clean or not re.match(r"^[A-Za-z0-9_\-]+$", clean):
        raise ValueError(
            f"Invalid run_id '{run_id}'. Run ID must contain only alphanumeric characters, underscores, and hyphens."
        )
    return clean


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RMR-v3 (RW-RMR)")
    ap.add_argument("-c", "--config", required=True, help="Path to config YAML")
    ap.add_argument("-r", "--resume", default=None, help="Resume from checkpoint path")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", "--output_dir", default=None, help="Explicit output directory")
    ap.add_argument("--save-dir", "--save_dir", default=None, help="Base save directory for runs (default: runs/sha_a)")
    ap.add_argument("--run-id", "--run_id", default=None, help="Run identifier (sets output_dir to <save_dir>/<run_id>)")
    ap.add_argument("--device", default=None, help="Computation device (e.g. cuda, cuda:0, cpu)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--disable-early-stopping", action="store_true", default=False)
    ap.add_argument("--deterministic", action="store_true", default=False, help="Enable strict determinism (default: enabled)")
    ap.add_argument("--non-deterministic", "--no-deterministic", dest="non_deterministic", action="store_true", default=False, help="Disable strict determinism")
    ap.add_argument("-o", "--overwrite", action="store_true", default=False)
    ap.add_argument("--allow-cross-commit-resume", action="store_true", default=False, help="Allow resuming checkpoint created from different git commit")
    ap.add_argument("--workers", type=int, default=None, help="Number of DataLoader worker processes (overrides config)")
    ap.add_argument("--batch-size", "--batch_size", type=int, default=None, help="Training batch size per step (overrides config)")
    ap.add_argument("--grad-accum", "--grad_accum", type=int, default=None, help="Gradient accumulation steps (overrides config)")
    ap.add_argument("--num-threads", type=int, default=None, help="PyTorch CPU intra-op thread count (recommended: 2 for parallel runs)")
    ap.add_argument("--no-cudnn-benchmark", action="store_true", default=False, help="Disable cuDNN benchmark to eliminate multi-process stalls")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = args.lr
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.eval_every is not None:
        cfg.setdefault("train", {})["eval_every"] = args.eval_every
    if args.patience is not None:
        cfg.setdefault("train", {})["patience"] = args.patience
    if args.workers is not None:
        cfg.setdefault("train", {})["workers"] = args.workers
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        cfg.setdefault("train", {})["gradient_accumulation_steps"] = args.grad_accum
    if args.num_threads is not None:
        cfg.setdefault("train", {})["num_threads"] = args.num_threads
    if args.no_cudnn_benchmark:
        cfg.setdefault("train", {})["cudnn_benchmark"] = False
    if args.disable_early_stopping:
        cfg.setdefault("train", {})["early_stopping"] = False
        cfg.setdefault("train", {})["patience"] = 0
    if args.non_deterministic:
        cfg.setdefault("train", {})["deterministic"] = False
    elif args.deterministic:
        cfg.setdefault("train", {})["deterministic"] = True
    elif "deterministic" not in cfg.get("train", {}):
        cfg.setdefault("train", {})["deterministic"] = True

    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device

    base_save = args.save_dir or "runs/sha_a"
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    elif args.run_id is not None:
        cfg["output_dir"] = f"{base_save}/{sanitize_run_id(args.run_id)}"
    elif "output_dir" not in cfg:
        cfg["output_dir"] = f"{base_save}/{Path(args.config).stem}"

    run_training_loop(cfg, args)


if __name__ == "__main__":
    main()
