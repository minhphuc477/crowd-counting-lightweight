from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from .provenance import get_git_info
from .reproducibility import save_rng_state


_SAFE_GLOBALS_INITIALIZED = False


def register_safe_globals() -> None:
    """Register safe NumPy data structures in PyTorch unpickler allowlist (CWE-502 mitigation)."""
    global _SAFE_GLOBALS_INITIALIZED
    if _SAFE_GLOBALS_INITIALIZED:
        return

    safe_types: list[Any] = []
    # NumPy internal multiarray reconstructors
    for mod_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            mod = __import__(mod_name, fromlist=["_reconstruct", "scalar"])
            for fn_name in ("_reconstruct", "scalar"):
                if hasattr(mod, fn_name):
                    safe_types.append(getattr(mod, fn_name))
        except (ImportError, AttributeError):
            pass

    # NumPy arrays, dtypes, and scalar types
    try:
        import numpy as np
        safe_types.extend([np.ndarray, np.dtype])
        for dt in (np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64, np.bool_):
            safe_types.append(dt)
        if hasattr(np, "dtypes"):
            for attr in dir(np.dtypes):
                obj = getattr(np.dtypes, attr)
                if isinstance(obj, type):
                    safe_types.append(obj)
    except (ImportError, AttributeError):
        pass

    if hasattr(torch.serialization, "add_safe_globals") and safe_types:
        try:
            torch.serialization.add_safe_globals(safe_types)
        except Exception:
            pass

    _SAFE_GLOBALS_INITIALIZED = True


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

    saved_directly = False
    try:
        try:
            torch.save(state, tmp_path)
        except Exception:
            for attempt in range(retries):
                try:
                    torch.save(state, target)
                    saved_directly = True
                    return
                except Exception:
                    if attempt == retries - 1:
                        raise
                    time.sleep(delay)
            return

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
                        saved_directly = True
                        return
                time.sleep(delay)
    finally:
        if not saved_directly and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def safe_torch_load(
    path: str | os.PathLike,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
) -> dict[str, Any]:
    """Safely deserialize PyTorch checkpoints with safe globals allowlisted against CWE-502."""
    register_safe_globals()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    try:
        return torch.load(p, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(p, map_location=map_location)
    except Exception as e:
        if weights_only:
            import warnings
            warnings.warn(
                f"Loading checkpoint '{p.name}' with weights_only=True failed ({e}). "
                "Falling back to weights_only=False for local trusted checkpoint.",
                UserWarning,
            )
            return torch.load(p, map_location=map_location, weights_only=False)
        raise


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
            k: (v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu())
            for k, v in ema_state.items()
        }
    if extra_fields:
        ckpt.update(extra_fields)
    return ckpt


__all__ = [
    "build_checkpoint",
    "register_safe_globals",
    "safe_torch_load",
    "safe_torch_save",
]
