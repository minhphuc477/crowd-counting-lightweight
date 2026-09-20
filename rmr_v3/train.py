from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    "train_one_epoch",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RMR-v3 (RW-RMR)")
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--resume", default=None, help="Resume from checkpoint path")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-id", default=None, help="Run identifier (sets output_dir to runs/sha_a/<run_id>)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--disable-early-stopping", action="store_true", default=False)
    ap.add_argument("--deterministic", action="store_true", default=False, help="Enable strict determinism (default: enabled)")
    ap.add_argument("--non-deterministic", action="store_true", default=False, help="Disable strict determinism")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--allow-cross-commit-resume", action="store_true", default=False, help="Allow resuming checkpoint created from different git commit")
    ap.add_argument("--teacher-ckpt", default=None, help="Path to teacher checkpoint for Stage 3 Knowledge Distillation")
    ap.add_argument("--workers", type=int, default=None, help="Number of DataLoader worker processes (overrides config)")
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
    if args.disable_early_stopping:
        cfg.setdefault("train", {})["early_stopping"] = False
        cfg.setdefault("train", {})["patience"] = 0
    if args.non_deterministic:
        cfg.setdefault("train", {})["deterministic"] = False
    elif args.deterministic or "deterministic" not in cfg.get("train", {}):
        cfg.setdefault("train", {})["deterministic"] = True

    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    elif args.run_id is not None:
        cfg["output_dir"] = f"runs/sha_a/{args.run_id}"
    elif "output_dir" not in cfg:
        cfg["output_dir"] = f"runs/sha_a/{Path(args.config).stem}"

    run_training_loop(cfg, args)


if __name__ == "__main__":
    main()
