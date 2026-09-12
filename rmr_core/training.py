from __future__ import annotations

import hashlib
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_git_info() -> tuple[str, bool]:
    """Retrieve the current git commit hash and working tree dirty status."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        commit = "unknown"
    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty = bool(status)
    except Exception:
        dirty = False
    return commit, dirty


def compute_file_sha256(path: Path | str) -> str:
    """Compute deterministic SHA256 hex digest of a file in 64KB blocks.

    For text-based manifest/config files, line endings are normalized (\\r\\n -> \\n)
    to guarantee cross-platform deterministic hashing between Linux (LF) and Windows (CRLF).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for SHA256 computation: {p}")
    h = hashlib.sha256()
    is_text = p.suffix.lower() in (".jsonl", ".json", ".yaml", ".yml", ".txt", ".csv")
    if is_text:
        with open(p, "rb") as f:
            content = f.read().replace(b"\r\n", b"\n")
            h.update(content)
    else:
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
    return h.hexdigest()



def build_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    config_hash: str,
    best_mae: float,
    epochs_without_improvement: int,
    solver_strength: float = 1.0,
    ema_state: dict[str, torch.Tensor] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical checkpoint bundle builder ensuring all state elements and provenance are recorded."""
    git_commit, git_dirty = get_git_info()
    ckpt: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": save_rng_state(),
        "solver_strength": float(solver_strength),
        "config": config,
        "config_hash": config_hash,
        "best_mae": float(best_mae),
        "epochs_without_improvement": int(epochs_without_improvement),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    if ema_state is not None:
        ckpt["ema_model"] = {
            k: v.detach().float().cpu() for k, v in ema_state.items()
        }
    if extra_fields:
        ckpt.update(extra_fields)
    return ckpt


def seed_everything(seed: int, deterministic: bool = False, warn_only: bool = False) -> None:
    """Set seeds across random, numpy, and torch, with deterministic algorithm controls."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def save_rng_state() -> dict[str, Any]:
    """Capture random, numpy, torch, and CUDA RNG states for exact resume reproducibility."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any] | None, strict: bool = False) -> None:
    """Restore random, numpy, torch, and CUDA RNG states."""
    if not isinstance(state, dict):
        if strict:
            raise ValueError("RNG state must be a valid dictionary for exact resume.")
        return
    py_state = state.get("python", state.get("python_random"))
    if py_state is not None:
        try:
            random.setstate(py_state)
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to restore Python RNG state: {e}") from e

    np_state = state.get("numpy")
    if np_state is not None:
        try:
            np.random.set_state(np_state)
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to restore NumPy RNG state: {e}") from e

    torch_state = state.get("torch")
    if torch_state is not None and isinstance(torch_state, torch.Tensor):
        try:
            torch.set_rng_state(torch_state)
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to restore Torch RNG state: {e}") from e

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available() and isinstance(cuda_state, (list, tuple)):
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to restore CUDA RNG state: {e}") from e


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup: int = 5,
    *,
    warmup_epochs: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing learning rate scheduler."""
    if warmup_epochs is not None:
        warmup = warmup_epochs

    def fn(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-3, (epoch + 1) / max(1, warmup))
        p = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def safe_torch_save(
    state: dict[str, Any],
    path: str | os.PathLike,
    retries: int = 5,
    delay: float = 0.5,
) -> None:
    """Windows-safe atomic checkpoint saving with retry logic to avoid file lock collisions."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.stem}_{os.getpid()}_{time.time_ns()}.tmp")

    try:
        torch.save(state, tmp_path)
    except Exception:
        # Fallback to direct save with retry if temp file creation fails
        for attempt in range(retries):
            try:
                torch.save(state, target)
                return
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
        return

    # Atomic or retry replacement on Windows
    for attempt in range(retries):
        try:
            os.replace(tmp_path, target)
            return
        except OSError:
            if attempt == retries - 1:
                try:
                    target.unlink(missing_ok=True)
                    os.replace(tmp_path, target)
                    return
                except Exception:
                    torch.save(state, target)
                    tmp_path.unlink(missing_ok=True)
                    return
            time.sleep(delay)

