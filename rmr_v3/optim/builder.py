"""Unified optimizer and scheduler factory with autonomous LR probing for RMR."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rmr_core.training import make_scheduler
from .lr_finder import SafeLRFinder
from .safe_prodigy import SafeProdigy
from .wsd_scheduler import make_wsd_scheduler

logger = logging.getLogger(__name__)


def build_v3_optimizer(model: nn.Module, cfg: dict[str, Any], lr_init: float) -> torch.optim.Optimizer:
    """Build optimizer supporting standard AdamW and SafeProdigy distance-adaptive optimizer."""
    train_cfg = cfg.get("train", {})
    opt_type = str(train_cfg.get("optimizer", "adamw")).lower().strip()
    backbone_scale = float(train_cfg.get("backbone_lr_scale", cfg.get("model", {}).get("backbone_lr_scale", 0.1)))
    wd = float(train_cfg.get("weight_decay", 1e-4))

    bb_decay, bb_no_decay = [], []
    other_decay, other_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bb = name.startswith("encoder.")
        if param.ndim <= 1 or name.endswith(".bias") or "tau" in name:
            (bb_no_decay if is_bb else other_no_decay).append(param)
        else:
            (bb_decay if is_bb else other_decay).append(param)

    param_groups = [
        {"name": "backbone_decay", "params": bb_decay, "lr": lr_init * backbone_scale, "weight_decay": wd},
        {"name": "backbone_no_decay", "params": bb_no_decay, "lr": lr_init * backbone_scale, "weight_decay": 0.0},
        {"name": "main_decay", "params": other_decay, "lr": lr_init, "weight_decay": wd},
        {"name": "main_no_decay", "params": other_no_decay, "lr": lr_init, "weight_decay": 0.0},
    ]

    if opt_type in ("prodigy", "safeprodigy", "safe_prodigy"):
        prodigy_groups = [
            {"name": "backbone_decay", "params": bb_decay, "lr": backbone_scale, "weight_decay": wd},
            {"name": "backbone_no_decay", "params": bb_no_decay, "lr": backbone_scale, "weight_decay": 0.0},
            {"name": "main_decay", "params": other_decay, "lr": 1.0, "weight_decay": wd},
            {"name": "main_no_decay", "params": other_no_decay, "lr": 1.0, "weight_decay": 0.0},
        ]
        d_coef = float(train_cfg.get("prodigy_d_coef", 0.5))
        growth_rate = float(train_cfg.get("prodigy_growth_rate", 1.015))
        d_max_cap = float(train_cfg.get("prodigy_d_max_cap", 0.08))
        d_warmup = int(train_cfg.get("prodigy_d_warmup_steps", 400))
        grad_thresh = float(train_cfg.get("prodigy_grad_spike_thresh", 5.0))
        use_rms = bool(train_cfg.get("prodigy_use_rms_clipping", False))
        d0 = float(train_cfg.get("prodigy_d0", 1e-5))

        logger.info(
            "[Optimizer] Initializing SafeProdigy: d0=%.2e, d_coef=%.2f, growth_rate=%.4f, d_max_cap=%.4f, warmup_steps=%d",
            d0, d_coef, growth_rate, d_max_cap, d_warmup,
        )
        return SafeProdigy(
            prodigy_groups,
            lr=1.0,
            weight_decay=wd,
            d0=d0,
            d_coef=d_coef,
            growth_rate=growth_rate,
            d_max_cap=d_max_cap,
            d_warmup_steps=d_warmup,
            grad_spike_thresh=grad_thresh,
            use_rms_clipping=use_rms,
            decouple=True,
        )

    return torch.optim.AdamW(param_groups)


def build_v3_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epochs: int,
) -> Any:
    """Build LR scheduler supporting Cosine Annealing and Warmup-Stable-Decay (WSD)."""
    train_cfg = cfg.get("train", {})
    sched_type = str(train_cfg.get("scheduler_type", "cosine")).lower().strip()
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))

    if sched_type in ("wsd", "warmup_stable_decay"):
        stable_ratio = float(train_cfg.get("wsd_stable_ratio", 0.75))
        logger.info(
            "[Scheduler] Initializing WSD Schedule: warmup=%d ep, stable_ratio=%.2f, floor=%.4f",
            warmup_epochs, stable_ratio, min_lr_ratio,
        )
        return make_wsd_scheduler(
            optimizer,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            stable_ratio=stable_ratio,
            min_lr_ratio=min_lr_ratio,
        )

    return make_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
    )


def maybe_run_safe_lr_finder(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    loss_fn: Any,
    cfg: dict[str, Any],
    device: torch.device,
) -> float | None:
    """Optionally probe optimal learning rate before epoch 1 if auto_lr_finder is configured."""
    train_cfg = cfg.get("train", {})
    if not bool(train_cfg.get("auto_lr_finder", False)):
        return None

    min_lr = float(train_cfg.get("lr_finder_min_lr", 1e-6))
    max_lr = float(train_cfg.get("lr_finder_max_lr", 1e-2))
    num_iter = int(train_cfg.get("lr_finder_num_iter", 70))
    max_safe_lr = float(train_cfg.get("lr_finder_max_safe_lr", 1e-3))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    effective_loss_fn = loss_fn
    if not callable(loss_fn):
        from rmr_v3.losses import compute_rmr_v3_losses
        effective_loss_fn = lambda out, tgt: compute_rmr_v3_losses(out, tgt, loss_fn)

    logger.info("[LR Finder] Probing optimal LR over %d iterations [%.1e -> %.1e]...", num_iter, min_lr, max_lr)
    finder = SafeLRFinder(model=model, optimizer=optimizer, loss_fn=effective_loss_fn, device=device)
    suggested_lr = finder.range_test(
        train_loader,
        min_lr=min_lr,
        max_lr=max_lr,
        num_iter=num_iter,
        max_safe_lr=max_safe_lr,
        grad_clip=grad_clip,
    )

    backbone_scale = float(train_cfg.get("backbone_lr_scale", cfg.get("model", {}).get("backbone_lr_scale", 0.1)))
    for group in optimizer.param_groups:
        scale = backbone_scale if "backbone" in group.get("name", "") else 1.0
        group["lr"] = suggested_lr * scale
        if "initial_lr" in group:
            group["initial_lr"] = suggested_lr * scale

    logger.info("[LR Finder] Set discovered optimal LR: %.2e (backbone: %.2e)", suggested_lr, suggested_lr * backbone_scale)
    return suggested_lr
