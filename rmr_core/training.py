from __future__ import annotations

import math
import random

import numpy as np
import torch


import os
from typing import Any


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
    warmup: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine annealing learning rate scheduler."""
    def fn(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-3, (epoch + 1) / max(1, warmup))
        p = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
