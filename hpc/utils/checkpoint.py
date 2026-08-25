import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
):
    """Save a pre-built training state dictionary."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        torch.save(state, os.path.join(save_dir, "best.pt"))


def build_checkpoint_state(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    **extra,
) -> Dict[str, Any]:
    """Build a checkpoint that includes learnable criterion state (e.g. NB dispersion)."""
    state: Dict[str, Any] = {"model_state_dict": model.state_dict(), **extra}
    if criterion is not None:
        state["criterion_state_dict"] = criterion.state_dict()
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if lr_scheduler is not None:
        state["scheduler_state_dict"] = lr_scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    return state


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
    checkpoint = torch.load(filepath, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if criterion is not None:
        if "criterion_state_dict" not in checkpoint:
            raise KeyError("Checkpoint has no criterion_state_dict")
        criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
