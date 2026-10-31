"""Checkpoint and EMA Shadow Parameter Management for RMR.

Handles atomic checkpoint saving, EMA weight tracking and evaluation context swapping,
and validation-driven early stopping.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json
from pathlib import Path
from typing import Any
import torch

from rmr_core.training import build_checkpoint, safe_torch_save
from .model import RMRv3


class EMAManager:
    """Maintains on-device float32 shadow parameters with warmup decay and BN statistics EMA."""

    def __init__(self, model: RMRv3, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.num_updates: int = 0
        self.state: dict[str, torch.Tensor] = {}
        self._float_buffers: set[str] = set()
        if self.decay > 0.0:
            for name, p in model.named_parameters():
                self.state[name] = p.detach().clone().float()
            persistent_keys = set(model.state_dict().keys())
            for name, b in model.named_buffers():
                if name in persistent_keys:
                    if b.is_floating_point():
                        self.state[name] = b.detach().clone().float()
                        self._float_buffers.add(name)
                    else:
                        self.state[name] = b.detach().clone()

    @torch.no_grad()
    def update(self, model: RMRv3) -> None:
        if not self.state or self.decay <= 0.0:
            return
        self.num_updates += 1
        d = min(self.decay, (1.0 + float(self.num_updates)) / (10.0 + float(self.num_updates)))
        for name, param in model.named_parameters():
            if name in self.state:
                self.state[name].mul_(d).add_(param.detach().float(), alpha=1.0 - d)
        for name, buf in model.named_buffers():
            if name in self.state:
                if name in self._float_buffers:
                    self.state[name].mul_(d).add_(buf.detach().float(), alpha=1.0 - d)
                else:
                    self.state[name].copy_(buf.detach())

    def restore(self, ckpt: dict[str, Any]) -> bool:
        if not self.state or "ema_model" not in ckpt:
            return False
        ema_sd = ckpt["ema_model"]
        for k, dst in self.state.items():
            if k in ema_sd:
                src = ema_sd[k].to(device=dst.device)
                dst.copy_(src.float() if dst.is_floating_point() else src.to(dtype=dst.dtype))
        fallback_updates = max(1, int(ckpt.get("epoch", 0)) * 37)
        self.num_updates = int(ckpt.get("ema_updates", fallback_updates))
        return True

    @contextmanager
    def swap_into(self, model: RMRv3, device: torch.device):
        """Temporarily swap EMA shadow weights into the model for evaluation."""
        if not self.state or self.decay <= 0.0:
            yield
            return
        live_backup = {k: v.clone() for k, v in model.state_dict().items()}
        try:
            swap_dict = {
                k: (self.state[k].to(device=device, dtype=live_backup[k].dtype) if k in self.state else live_backup[k])
                for k in live_backup
            }
            model.load_state_dict(swap_dict, strict=True)
            yield
        finally:
            model.load_state_dict(live_backup)


class CheckpointManager:
    """Encapsulates checkpoint saving (last.pt, best.pt) and early stopping."""

    def __init__(
        self,
        out_dir: Path,
        cfg: dict[str, Any],
        config_hash: str,
        patience: int = 0,
        solver_warmup_epochs: int = 5,
        solver_ramp_epochs: int = 20,
        best_mae: float = float("inf"),
        epochs_without_improvement: int = 0,
        generator: torch.Generator | None = None,
    ) -> None:
        self.out_dir = out_dir
        self.cfg = cfg
        self.config_hash = config_hash
        self.patience = patience
        self.solver_warmup_epochs = solver_warmup_epochs
        self.solver_ramp_epochs = solver_ramp_epochs
        self.best_mae = best_mae
        self.epochs_without_improvement = epochs_without_improvement
        self.generator = generator

    def save_last(
        self,
        epoch: int,
        model: RMRv3,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
        solver_strength: float,
        ema_manager: EMAManager,
    ) -> None:
        ckpt = build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=self.cfg,
            config_hash=self.config_hash,
            best_mae=self.best_mae,
            epochs_without_improvement=self.epochs_without_improvement if self.patience > 0 else 0,
            solver_strength=solver_strength,
            ema_state=ema_manager.state if ema_manager.state else None,
            extra_fields={"ema_updates": ema_manager.num_updates} if ema_manager.state else None,
            generator=self.generator,
        )
        safe_torch_save(ckpt, self.out_dir / "last.pt")

    def evaluate_and_save_best(
        self,
        epoch: int,
        cur_mae: float,
        val_metrics: dict[str, Any],
        model: RMRv3,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: torch.amp.GradScaler,
        solver_strength: float,
        ema_manager: EMAManager,
        eval_every: int,
        val_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        solver_enabled = bool(model.cfg.enable_solver)
        solver_engaged = (not solver_enabled) or (
            solver_strength >= 1.0 or epoch >= self.solver_warmup_epochs + self.solver_ramp_epochs
        )
        is_best = (cur_mae < self.best_mae) and solver_engaged

        if is_best:
            self.best_mae = cur_mae
            self.epochs_without_improvement = 0
            ckpt = build_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=self.cfg,
                config_hash=self.config_hash,
                best_mae=self.best_mae,
                epochs_without_improvement=self.epochs_without_improvement,
                solver_strength=solver_strength,
                ema_state=ema_manager.state if ema_manager.state else None,
                extra_fields={"ema_updates": ema_manager.num_updates} if ema_manager.state else None,
                generator=self.generator,
            )
            safe_torch_save(ckpt, self.out_dir / "best_val_mae.pt")
            safe_torch_save(ckpt, self.out_dir / "best_model.pt")
            eval_dir = self.out_dir / "eval_val"
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "summary.json").write_text(json.dumps(val_metrics, indent=2))
            if val_rows:
                pred_path = eval_dir / "predictions.csv"
                fieldnames = list(val_rows[0].keys())
                with pred_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(val_rows)
            return True, " >>> [NEW BEST CHECKPOINT SAVED] <<<"
        else:
            if solver_engaged and self.patience > 0:
                self.epochs_without_improvement += eval_every
            ramp_end = self.solver_warmup_epochs + self.solver_ramp_epochs
            tag = f" (Solver ramping: epoch {epoch}/{ramp_end})" if not solver_engaged else ""
            return False, tag

    @property
    def should_stop_early(self) -> bool:
        return self.patience > 0 and self.epochs_without_improvement >= self.patience
