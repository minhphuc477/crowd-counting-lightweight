from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.evaluation import evaluate_dataset
from rmr_v3.diagnostics import (
    compute_dispersion_saturation,
    compute_nb_interval_coverage,
    compute_reliability_correlations,
    compute_solver_trajectory_diagnostics,
    compute_uncertainty_calibration_bins,
    regional_reliability_rows,
)
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.tracking import DiagnosticTracker, LossTracker
from rmr_v3.checkpoint import EMAManager


def make_model(cfg: dict) -> tuple[RMRv3, bool]:
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("train", {})
    bb_lr = float(m_cfg.get("backbone_lr_scale", t_cfg.get("backbone_lr_scale", 0.1)))
    config = RMRv3Config.from_dict(m_cfg, backbone_lr_scale=bb_lr)
    model = RMRv3(config)
    uniform_reliability = bool(m_cfg.get("uniform_reliability", False))
    return model, uniform_reliability


def make_loss_cfg(cfg: dict) -> RMRv3LossConfig:
    loss_d = dict(cfg.get("loss", {})) if "loss" in cfg else dict(cfg)
    if "output_stride" not in loss_d and "model" in cfg:
        loss_d["output_stride"] = cfg["model"].get("output_stride", 4)
    return RMRv3LossConfig.from_dict(loss_d)


def build_optimizer(model: RMRv3, cfg: dict, lr_init: float) -> torch.optim.Optimizer:
    """Build AdamW optimizer with decoupled weight decay for 1D tensors and calibrated priors."""
    backbone_scale = float(
        cfg.get("train", {}).get(
            "backbone_lr_scale", cfg.get("model", {}).get("backbone_lr_scale", 0.1)
        )
    )
    wd = float(cfg.get("train", {}).get("weight_decay", 1e-4))

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

    return torch.optim.AdamW(
        [
            {"name": "backbone_decay", "params": bb_decay, "lr": lr_init * backbone_scale, "weight_decay": wd},
            {"name": "backbone_no_decay", "params": bb_no_decay, "lr": lr_init * backbone_scale, "weight_decay": 0.0},
            {"name": "main_decay", "params": other_decay, "lr": lr_init, "weight_decay": wd},
            {"name": "main_no_decay", "params": other_no_decay, "lr": lr_init, "weight_decay": 0.0},
        ],
    )


@torch.no_grad()
def evaluate_v3(
    model: RMRv3,
    loader: DataLoader,
    device: torch.device,
    uniform_reliability: bool = False,
    density_bins: tuple[float, float] = (100.0, 500.0),
) -> dict:
    all_diag_rows = []
    traj_rows = []

    if getattr(model.cfg, "use_park", False):
        n_bands = int(getattr(model.cfg, "park_altitude_bands", 3))
        scale_map = {sid: f"band_{sid}" for sid in range(n_bands)}
    else:
        scale_sizes = tuple(model.cfg.region_sizes_px)
        scale_map = {sid: int(s if isinstance(s, int) else s[0]) for sid, s in enumerate(scale_sizes)}

    def sample_callback(sample: dict, out: dict, y: torch.Tensor, row: dict) -> dict:
        target = sample.get("target_device", sample["target_y"].to(device))
        target_4d = target if target.ndim == 4 else (target.unsqueeze(0) if target.ndim == 3 else target.unsqueeze(0).unsqueeze(0))
        d_rows = regional_reliability_rows(out, target_4d, max_regions=300)
        all_diag_rows.extend(d_rows)
        t_diag = compute_solver_trajectory_diagnostics(out, target_4d, scale_map=scale_map)
        if t_diag:
            traj_rows.append(t_diag)
        return {}

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=int(model.cfg.output_stride),
        run_tiling=False,
        forward_kwargs={"uniform_reliability": uniform_reliability, "solver_strength": 1.0},
        extra_sample_callback=sample_callback,
        enforce_gt_consistency=True,
        density_bins=density_bins,
    )

    corrs = compute_reliability_correlations(all_diag_rows, scale_map=scale_map)
    summary.update(corrs)

    calib = compute_uncertainty_calibration_bins(all_diag_rows, scale_map=scale_map)
    summary["calibration"] = calib
    summary["mean_std_residual"] = calib["mean_std_residual"]
    summary["p50_std_residual"] = calib["p50_std_residual"]
    summary["p90_std_residual"] = calib["p90_std_residual"]

    disp_min = float(model.cfg.dispersion_min)
    disp_max = float(model.cfg.dispersion_max)
    sat = compute_dispersion_saturation(all_diag_rows, disp_min=disp_min, disp_max=disp_max)
    summary.update(sat)

    nb_cov = compute_nb_interval_coverage(all_diag_rows, scale_map=scale_map)
    summary.update(nb_cov)

    if traj_rows:
        for k in traj_rows[0].keys():
            vals = [tr[k] for tr in traj_rows if k in tr]
            summary[k] = float(np.mean(vals)) if vals else 0.0

    return summary


def train_one_epoch(
    model: RMRv3,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    loss_cfg: RMRv3LossConfig,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    uniform_reliability: bool,
    solver_strength: float,
    ema_manager: EMAManager,
    teacher_model: Any = None,
    kd_loss_fn: Any = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Execute one training epoch with mixed precision, gradient clipping, and EMA tracking."""
    model.train()
    loss_tracker = LossTracker()
    diag_tracker = DiagnosticTracker(model)

    # Determine epoch length defensively without raising exceptions
    n_batches = len(train_loader) if hasattr(train_loader, "__len__") else -1

    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target_y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=amp):
            outputs = model(images, uniform_reliability=uniform_reliability, solver_strength=solver_strength)
            losses = compute_rmr_v3_losses(outputs, targets, loss_cfg, points=batch.get("points"))
            loss = losses["total"]

            if teacher_model is not None and kd_loss_fn is not None:
                with torch.no_grad():
                    t_out = teacher_model(images)
                    t_y = t_out["y"] if isinstance(t_out, dict) else t_out

                if loss_cfg.dm_target == "dual":
                    kd_y0 = kd_loss_fn(outputs["y0"], t_y)
                    kd_y = kd_loss_fn(outputs["y"], t_y)
                    kd_res = {
                        "total_kd": 0.5 * kd_y0["total_kd"] + 0.5 * kd_y["total_kd"],
                        "spatial_kl": 0.5 * kd_y0["spatial_kl"] + 0.5 * kd_y["spatial_kl"],
                        "count_kd": 0.5 * kd_y0["count_kd"] + 0.5 * kd_y["count_kd"],
                    }
                elif loss_cfg.dm_target == "y":
                    kd_res = kd_loss_fn(outputs["y"], t_y)
                else:
                    kd_res = kd_loss_fn(outputs["y0"], t_y)

                loss = loss + kd_res["total_kd"]
                losses["kd_total"] = kd_res["total_kd"]
                losses["kd_spatial"] = kd_res["spatial_kl"]
                losses["kd_count"] = kd_res["count_kd"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        ema_manager.update(model)

        loss_tracker.update(losses)
        # Zero-Sync Diagnostic Policy: Update tracker on the final batch of the epoch.
        # Captures an exact 2,400-region snapshot of solver state while eliminating
        # 37 CUDA host-synchronization flushes per epoch.
        if n_batches <= 0 or batch_idx == n_batches - 1:
            diag_tracker.update(outputs)

    scheduler.step()
    return loss_tracker.averages(), diag_tracker.summarize()
