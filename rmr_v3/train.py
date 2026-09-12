from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import sys
from pathlib import Path
from typing import Any

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

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from rmr_core.data import (
    CrowdManifestDataset,
    collate_eval,
    collate_train,
    compute_manifest_density,
)
from rmr_core.evaluation import evaluate_dataset
from rmr_core.training import (
    build_checkpoint,
    get_git_info,
    load_rng_state,
    make_scheduler,
    safe_torch_save,
    seed_everything,
)

from .config import compute_config_hash, validate_resume_compatibility, validate_v3_config
from .diagnostics import (
    compute_dispersion_saturation,
    compute_nb_interval_coverage,
    compute_reliability_correlations,
    compute_solver_trajectory_diagnostics,
    compute_uncertainty_calibration_bins,
    regional_reliability_rows,
)
from .kd import DensityMapKDLoss
from .losses import RMRv3LossConfig, compute_rmr_v3_losses
from .model import RMRv3, RMRv3Config


TRAIN_LOG_FIELDNAMES: list[str] = [
    "epoch", "lr_backbone", "lr_main", "solver_strength",
    "train_total", "train_count", "train_flat_dm16", "train_allocation",
    "train_dm16", "train_dm32", "train_dm64",
    "train_cell", "train_region_nb", "train_hurdle_bce", "train_trunc_nb",
    "train_kd_total", "train_kd_spatial", "train_kd_count",
    "region_mu_mean", "region_dispersion_mean", "region_dispersion_p10", "region_dispersion_p50", "region_dispersion_p90",
    "region_weight_mean", "region_weight_std", "region_weight_min", "region_weight_max",
    "solver_weight_mean", "solver_weight_std",
    "weight_clip_low_fraction", "weight_clip_high_fraction",
    "solver_energy_before", "solver_energy_after", "solver_energy_reduction",
    "weight_mean_32", "weight_mean_64", "weight_mean_128",
    "val_mae", "val_rmse", "val_nae", "val_bias",
    "val_game0", "val_game1", "val_game2", "val_game3",
    "val_mae_sparse", "val_mae_moderate", "val_mae_dense",
    "pearson_rate_var_error", "spearman_rate_var_error", "spearman_weight_error",
    "spearman_pred_weight_error",
    "spearman_rate_var_error_32", "spearman_rate_var_error_64", "spearman_rate_var_error_128",
    "mean_std_residual",
    "coverage_50", "coverage_80", "coverage_95",
    "calib_gap_50", "calib_gap_80", "calib_gap_95",
    "dispersion_sat_low_fraction", "dispersion_sat_high_fraction",
    "solver_help_fraction", "solver_harm_fraction", "energy_monotonic_fraction",
    "mae_reg_y0", "mae_reg_y1", "mae_reg_y2",
]


class LossTracker:
    """Tracks and averages multi-task training loss components across mini-batches."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count: int = 0

    def update(self, loss_dict: dict[str, Any]) -> None:
        self.count += 1
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                val = float(v.item())
            elif isinstance(v, (float, int)):
                val = float(v)
            else:
                continue
            self.totals[k] = self.totals.get(k, 0.0) + val

    def averages(self) -> dict[str, float]:
        c = max(self.count, 1)
        return {k: v / c for k, v in self.totals.items()}


class DiagnosticTracker:
    """Collects and computes batch-level regional statistics and solver energy traces."""

    def __init__(self, model: RMRv3) -> None:
        self.model = model
        self.mu_means: list[float] = []
        self.disp_means: list[float] = []
        self.all_disps: list[torch.Tensor] = []
        self.all_pred_weights: list[torch.Tensor] = []
        self.all_solver_weights: list[torch.Tensor] = []
        self.e_befores: list[float] = []
        self.e_afters: list[float] = []
        self.w_scale_32: list[float] = []
        self.w_scale_64: list[float] = []
        self.w_scale_128: list[float] = []

    @torch.no_grad()
    def update(self, outputs: dict[str, Any]) -> None:
        self.mu_means.append(float(outputs["b_region"].mean().item()))
        self.disp_means.append(float(outputs["region_dispersion"].mean().item()))
        self.all_disps.append(outputs["region_dispersion"].detach().cpu().float().flatten())
        self.all_pred_weights.append(outputs["region_weight"].detach().cpu().float().flatten())
        self.all_solver_weights.append(outputs["solver_region_weight"].detach().cpu().float().flatten())

        et = outputs.get("energy_trace", [])
        if et:
            self.e_befores.append(float(et[0]["before"].mean().item()))
            self.e_afters.append(float(et[-1]["after"].mean().item()))

        regions = outputs["regions"]
        w = outputs["solver_region_weight"]
        m32 = regions.scale_id == 0
        m64 = regions.scale_id == 1
        m128 = regions.scale_id == 2
        if m32.any():
            self.w_scale_32.append(float(w[..., m32].mean().item()))
        if m64.any():
            self.w_scale_64.append(float(w[..., m64].mean().item()))
        if m128.any():
            self.w_scale_128.append(float(w[..., m128].mean().item()))

    def summarize(self) -> dict[str, float]:
        disps_np = torch.cat(self.all_disps).cpu().numpy() if self.all_disps else np.array([50.0])
        pred_weights_np = torch.cat(self.all_pred_weights).cpu().numpy() if self.all_pred_weights else np.array([1.0])
        solver_weights_np = torch.cat(self.all_solver_weights).cpu().numpy() if self.all_solver_weights else np.array([1.0])

        e_b = float(np.mean(self.e_befores)) if self.e_befores else 0.0
        e_a = float(np.mean(self.e_afters)) if self.e_afters else 0.0
        e_red = (e_b - e_a) / max(e_b, 1e-8)

        w_min_val = self.model.cfg.reliability_weight_min
        w_max_val = self.model.cfg.reliability_weight_max
        w_low_frac = float(np.mean(pred_weights_np <= w_min_val + 1e-4))
        w_high_frac = float(np.mean(pred_weights_np >= w_max_val - 1e-4))

        return {
            "region_mu_mean": float(np.mean(self.mu_means)) if self.mu_means else 0.0,
            "region_dispersion_mean": float(np.mean(self.disp_means)) if self.disp_means else 50.0,
            "region_dispersion_p10": float(np.percentile(disps_np, 10)),
            "region_dispersion_p50": float(np.percentile(disps_np, 50)),
            "region_dispersion_p90": float(np.percentile(disps_np, 90)),
            "region_weight_mean": float(np.mean(pred_weights_np)),
            "region_weight_std": float(np.std(pred_weights_np)),
            "region_weight_min": float(np.min(pred_weights_np)),
            "region_weight_max": float(np.max(pred_weights_np)),
            "solver_weight_mean": float(np.mean(solver_weights_np)),
            "solver_weight_std": float(np.std(solver_weights_np)),
            "weight_clip_low_fraction": w_low_frac,
            "weight_clip_high_fraction": w_high_frac,
            "solver_energy_before": e_b,
            "solver_energy_after": e_a,
            "solver_energy_reduction": e_red,
            "weight_mean_32": float(np.mean(self.w_scale_32)) if self.w_scale_32 else 1.0,
            "weight_mean_64": float(np.mean(self.w_scale_64)) if self.w_scale_64 else 1.0,
            "weight_mean_128": float(np.mean(self.w_scale_128)) if self.w_scale_128 else 1.0,
        }


class EMAManager:
    """Maintains on-device float32 shadow parameters with swap context management."""

    def __init__(self, model: RMRv3, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.state: dict[str, torch.Tensor] = {}
        if self.decay > 0.0:
            for name, p in model.named_parameters():
                self.state[name] = p.detach().clone().float()
            for name, b in model.named_buffers():
                self.state[name] = b.detach().clone().float()

    @torch.no_grad()
    def update(self, model: RMRv3) -> None:
        if not self.state or self.decay <= 0.0:
            return
        d = self.decay
        for name, param in model.named_parameters():
            if name in self.state:
                self.state[name].mul_(d).add_(param.detach().float(), alpha=1.0 - d)
        for name, buf in model.named_buffers():
            if name in self.state:
                self.state[name].copy_(buf.float())

    def restore(self, ckpt: dict[str, Any]) -> bool:
        if not self.state or "ema_model" not in ckpt:
            return False
        ema_sd = ckpt["ema_model"]
        for k in self.state:
            if k in ema_sd:
                self.state[k].copy_(ema_sd[k].to(device=self.state[k].device).float())
        return True

    @contextmanager
    def swap_into(self, model: RMRv3, device: torch.device):
        """Temporarily swap EMA shadow weights into the model for evaluation."""
        if not self.state or self.decay <= 0.0:
            yield
            return
        live_backup = {k: v.clone() for k, v in model.state_dict().items()}
        try:
            model.load_state_dict(
                {k: self.state[k].to(device=device, dtype=live_backup[k].dtype) for k in live_backup if k in self.state}
            )
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
    ) -> None:
        self.out_dir = out_dir
        self.cfg = cfg
        self.config_hash = config_hash
        self.patience = patience
        self.solver_warmup_epochs = solver_warmup_epochs
        self.solver_ramp_epochs = solver_ramp_epochs
        self.best_mae = best_mae
        self.epochs_without_improvement = epochs_without_improvement

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
    ) -> tuple[bool, str]:
        solver_enabled = getattr(model.cfg, "enable_solver", True)
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
            )
            safe_torch_save(ckpt, self.out_dir / "best_val_mae.pt")
            (self.out_dir / "eval_val").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "eval_val" / "summary.json").write_text(json.dumps(val_metrics, indent=2))
            return True, " >>> [NEW BEST CHECKPOINT SAVED] <<<"
        else:
            if solver_engaged and self.patience > 0:
                self.epochs_without_improvement += eval_every
            tag = f" (Solver ramping: epoch {epoch}/{self.solver_warmup_epochs + self.solver_ramp_epochs})" if not solver_engaged else ""
            return False, tag

    @property
    def should_stop_early(self) -> bool:
        return self.patience > 0 and self.epochs_without_improvement >= self.patience


def make_model(cfg: dict) -> tuple[RMRv3, bool]:
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("train", {})
    bb_lr = float(m_cfg.get("backbone_lr_scale", t_cfg.get("backbone_lr_scale", 0.1)))
    config = RMRv3Config.from_dict(m_cfg, backbone_lr_scale=bb_lr)
    model = RMRv3(config)
    uniform_reliability = bool(m_cfg.get("uniform_reliability", False))
    return model, uniform_reliability


def make_loss_cfg(cfg: dict) -> RMRv3LossConfig:
    return RMRv3LossConfig.from_dict(cfg.get("loss", {}))


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

    def sample_callback(sample: dict, out: dict, y: torch.Tensor, row: dict) -> dict:
        target = sample["target_y"].to(device)
        d_rows = regional_reliability_rows(out, target.unsqueeze(0))
        all_diag_rows.extend(d_rows)
        t_diag = compute_solver_trajectory_diagnostics(out, target.unsqueeze(0))
        if t_diag:
            traj_rows.append(t_diag)
        return {}

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=getattr(model.cfg, "output_stride", 4),
        run_tiling=False,
        forward_kwargs={"uniform_reliability": uniform_reliability, "solver_strength": 1.0},
        extra_sample_callback=sample_callback,
        enforce_gt_consistency=True,
        density_bins=density_bins,
    )

    corrs = compute_reliability_correlations(all_diag_rows)
    summary.update(corrs)

    calib = compute_uncertainty_calibration_bins(all_diag_rows)
    summary["calibration"] = calib
    summary["mean_std_residual"] = calib["mean_std_residual"]
    summary["p50_std_residual"] = calib["p50_std_residual"]
    summary["p90_std_residual"] = calib["p90_std_residual"]

    disp_min = float(getattr(model.cfg, "dispersion_min", 0.5))
    disp_max = float(getattr(model.cfg, "dispersion_max", 500.0))
    sat = compute_dispersion_saturation(all_diag_rows, disp_min=disp_min, disp_max=disp_max)
    summary.update(sat)

    nb_cov = compute_nb_interval_coverage(all_diag_rows)
    summary.update(nb_cov)

    if traj_rows:
        for k in traj_rows[0].keys():
            vals = [tr[k] for tr in traj_rows if k in tr]
            summary[k] = float(np.mean(vals)) if vals else 0.0

    return summary


def format_dynamic_training_banner(
    model: RMRv3,
    cfg: dict[str, Any],
    run_id: str | None,
    n_params: int,
    device: torch.device,
    epochs: int,
    bs: int,
    lr_init: float,
    out_dir: Path,
) -> str:
    """Format a dynamic, attribute-driven training banner without hardcoded version strings."""
    m_cfg = model.cfg

    # 1. Carrier & Backbone
    bb_name = m_cfg.backbone_name.split(".")[0]
    stride = getattr(m_cfg, "output_stride", 4)
    feat_w = getattr(m_cfg, "feature_width", 32)
    neck_desc = getattr(m_cfg, "neck_type", "additive")
    if getattr(m_cfg, "use_coord_attn", False):
        neck_desc += "+CoordAttn"
    carrier_line = f"Stride-{stride} | Backbone: {bb_name} | Neck: {neck_desc} (C={feat_w})"

    # 2. Geometry & Spatial Pooling
    region_sizes = list(getattr(m_cfg, "region_sizes_px", (32, 64, 128)))
    is_aniso = any(isinstance(s, (tuple, list)) and len(s) == 2 and s[0] != s[1] for s in region_sizes)
    geo_tag = "Anisotropic Perspective" if is_aniso else "Isotropic"

    reg_strs = []
    for s in region_sizes:
        if isinstance(s, (tuple, list)):
            reg_strs.append(f"({s[0]}x{s[1]})")
        else:
            reg_strs.append(str(s))
    overlap = getattr(m_cfg, "region_overlap", 0.5)
    reg_desc = f"[{', '.join(reg_strs)}] px ({geo_tag}, overlap={overlap:.0%})"

    pool_mode = getattr(m_cfg, "regional_feature_stats", "mean")
    if pool_mode == "mean_std":
        pooling_desc = "Spatial Moments (Mean+Std)"
    elif getattr(m_cfg, "native_scale_pooling", False):
        pooling_desc = "Native Multiscale Pooling"
    else:
        pooling_desc = "Spatial Average"

    # 3. Inverse Solver Engine
    if not getattr(m_cfg, "enable_solver", True):
        solver_desc = "Disabled (Direct Feedforward Carrier Head)"
    else:
        solver_parts = []
        if getattr(m_cfg, "proximal_tau", 0.0) > 0.0:
            solver_parts.append(f"Proximal RW-SIRT (tau={m_cfg.proximal_tau})")
        elif getattr(m_cfg, "solver_mode", "additive") == "multiplicative":
            solver_parts.append(f"Density-Gated RW-SIRT (rho={m_cfg.density_gate_rho})")
        else:
            solver_parts.append("Additive RW-SIRT")

        solver_parts.append(f"T={getattr(m_cfg, 'iterations', 2)}")
        solver_parts.append(f"omega={getattr(m_cfg, 'omega', 1.0)}")

        tv_lam = getattr(m_cfg, "tv_lambda", 0.0)
        if tv_lam > 0.0:
            tv_t = getattr(m_cfg, "tv_type", "laplacian").capitalize()
            solver_parts.append(f"{tv_t} TV (lambda={tv_lam})")

        w_mode = "Uniform (W=I)" if getattr(m_cfg, "uniform_reliability", False) else "Negative-Binomial (W=diag(w_R))"
        solver_parts.append(f"Weighting={w_mode}")
        solver_desc = " | ".join(solver_parts)

    # 4. Heads & Densities
    guidance_head = "Hurdle-NB (Occupancy Gated)" if getattr(m_cfg, "hurdle_head", False) else "Negative-Binomial"
    if getattr(m_cfg, "temp_softplus", False):
        density_head = "Learnable Temp Softplus"
    else:
        m0_val = getattr(m_cfg, "init_m0", 0.015763)
        density_head = f"Calibrated Softplus (m0={m0_val:.5f})"

    # 5. Supervision Target & Loss
    dm_target = cfg.get("loss", {}).get("dm_target", "y0")
    target_desc = "Post-Solver Y (End-to-End Measure Optimization)" if dm_target == "y" else "Pre-Solver Y0 (Decoupled Carrier Guidance)"
    loss_type = cfg.get("loss", {}).get("allocation_loss_type", "flat_dm16")
    supervision_desc = f"{target_desc} -> {loss_type.upper()}"

    budget_pct = (n_params / 105_000) * 100
    headroom = 105_000 - n_params
    run_label = run_id if run_id else Path(cfg.get("output_dir", "unspecified")).name

    banner = [
        "=" * 80,
        f"  RMR Training Initialized [Dynamic Architecture Engine]",
        f"  Run: {run_label} | Parameters: {n_params:,} / 105,000 budget ({budget_pct:.1f}% used, +{headroom:,} headroom)",
        f"  Carrier: {carrier_line}",
        f"  Geometry: {reg_desc} | Pooling: {pooling_desc}",
        f"  Solver: {solver_desc}",
        f"  Heads: Regional={guidance_head} | Density={density_head}",
        f"  Supervision: {supervision_desc}",
        f"  Training: Epochs: {epochs} | Batch Size: {bs} | Initial LR: {lr_init:.2e} | Device: {device}",
        f"  Output Directory: {out_dir}",
        "=" * 80,
    ]
    return "\n".join(banner)


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

    for batch in train_loader:
        images = batch["image"].to(device)
        targets = batch["target_y"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(images, uniform_reliability=uniform_reliability, solver_strength=solver_strength)
            losses = compute_rmr_v3_losses(outputs, targets, loss_cfg, points=batch.get("points"))
            loss = losses["total"]

            if teacher_model is not None and kd_loss_fn is not None:
                with torch.no_grad():
                    t_out = teacher_model(images)
                    t_y = t_out["y"] if isinstance(t_out, dict) else t_out
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
        diag_tracker.update(outputs)

    scheduler.step()
    return loss_tracker.averages(), diag_tracker.summarize()


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
    ap.add_argument("--deterministic", action="store_true", default=False, help="Enable strict determinism")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--allow-cross-commit-resume", action="store_true", default=False, help="Allow resuming checkpoint created from different git commit")
    ap.add_argument("--teacher-ckpt", default=None, help="Path to teacher checkpoint for Stage 3 Knowledge Distillation")
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
    if args.disable_early_stopping:
        cfg.setdefault("train", {})["early_stopping"] = False
        cfg.setdefault("train", {})["patience"] = 0

    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    elif args.run_id is not None:
        cfg["output_dir"] = f"runs/sha_a/{args.run_id}"
    elif "output_dir" not in cfg:
        cfg["output_dir"] = f"runs/sha_a/{Path(args.config).stem}"

    validate_v3_config(cfg)

    seed = int(cfg.get("seed", 42))
    deterministic = bool(args.deterministic or cfg.get("train", {}).get("deterministic", False))
    cfg.setdefault("train", {})["deterministic"] = deterministic
    seed_everything(seed, deterministic=deterministic)

    if "init_m0" not in cfg.get("model", {}) and "train_manifest" in cfg.get("data", {}):
        stride = int(cfg.get("model", {}).get("output_stride", 4))
        cfg.setdefault("model", {})["init_m0"] = compute_manifest_density(
            cfg["data"]["train_manifest"],
            output_stride=stride,
            data_root=cfg["data"].get("data_root"),
        )

    run_config_hash = compute_config_hash(cfg)

    resume_ckpt = None
    if args.resume:
        try:
            resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            resume_ckpt = torch.load(args.resume, map_location="cpu")
        current_commit, _ = get_git_info()
        ckpt_commit = str(resume_ckpt.get("git_commit", resume_ckpt.get("provenance", {}).get("git_commit", "unknown")))
        validate_resume_compatibility(
            resume_ckpt.get("config", {}),
            cfg,
            ckpt_hash=resume_ckpt.get("config_hash"),
            incoming_hash=run_config_hash,
            ckpt_commit=ckpt_commit,
            current_commit=current_commit,
            allow_cross_commit=bool(args.allow_cross_commit_resume),
        )

    out_dir = Path(cfg["output_dir"])
    if out_dir.exists() and not args.resume:
        existing = list(out_dir.iterdir())
        if existing:
            if not args.overwrite:
                raise RuntimeError(
                    f"Output directory '{out_dir}' already exists with artifacts: "
                    f"{[f.name for f in existing[:10]]}. Use --overwrite or --resume."
                )
            import shutil
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_ds = CrowdManifestDataset(
        cfg["data"]["train_manifest"],
        train=True,
        output_stride=cfg.get("model", {}).get("output_stride", 4),
        crop_size=cfg.get("data", {}).get("crop_size", 512),
        scale_range=tuple(cfg.get("data", {}).get("scale_range", [0.75, 1.25])),
        hflip_prob=float(cfg.get("data", {}).get("hflip_prob", 0.5)),
        brightness_jitter=float(cfg.get("data", {}).get("brightness_jitter", 0.0)),
        contrast_jitter=float(cfg.get("data", {}).get("contrast_jitter", 0.0)),
        gamma_jitter=tuple(cfg.get("data", {}).get("gamma_jitter", [1.0, 1.0])),
        random_invert_prob=float(cfg.get("data", {}).get("random_invert_prob", 0.0)),
        data_root=cfg.get("data", {}).get("data_root"),
    )
    val_manifest = cfg.get("data", {}).get("val_manifest")
    val_ds = None if not val_manifest else CrowdManifestDataset(
        val_manifest,
        train=False,
        output_stride=cfg.get("model", {}).get("output_stride", 4),
        data_root=cfg.get("data", {}).get("data_root"),
    )

    workers = int(cfg.get("train", {}).get("workers", 0))
    pin_mem = bool(cfg.get("train", {}).get("pin_memory", False))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
        collate_fn=collate_train,
        drop_last=True,
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, uniform_reliability = make_model(cfg)
    model.to(device)

    # EMA Tracking setup
    ema_decay = float(cfg.get("train", {}).get("ema_decay", cfg.get("model", {}).get("ema_decay", 0.0)))
    ema_manager = EMAManager(model, decay=ema_decay)
    if ema_manager.state:
        print(f"EMA enabled: decay={ema_decay:.4f}")

    # Teacher model for Stage 3 KD
    teacher_model = None
    kd_loss_fn = None
    teacher_ckpt_path = args.teacher_ckpt or cfg.get("train", {}).get("teacher_ckpt")
    if teacher_ckpt_path:
        tp = Path(teacher_ckpt_path)
        if tp.exists():
            try:
                try:
                    t_ckpt = torch.load(tp, map_location="cpu", weights_only=False)
                except TypeError:
                    t_ckpt = torch.load(tp, map_location="cpu")
                t_cfg = t_ckpt.get("config", {})
                teacher_model, _ = make_model(t_cfg)
                teacher_model.load_state_dict(t_ckpt["model"])
                teacher_model.switch_to_deploy()
                teacher_model.to(device).eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
                kd_loss_fn = DensityMapKDLoss(
                    lambda_spatial_kl=float(cfg.get("loss", {}).get("lambda_kd_spatial", 1.0)),
                    lambda_count_kd=float(cfg.get("loss", {}).get("lambda_kd_count", 0.1)),
                )
                print(f"[Stage 3 KD] Teacher loaded and frozen successfully.")
            except Exception as e:
                print(f"[Stage 3 KD Warning] Failed to load teacher from {tp}: {e}. Proceeding without KD.")
        else:
            print(f"[Stage 3 KD Warning] Teacher checkpoint not found at '{tp}'. Proceeding without KD.")

    epochs = int(cfg.get("train", {}).get("epochs", 1000))
    lr_init = float(cfg.get("train", {}).get("lr", 1e-4))
    bs = int(cfg.get("train", {}).get("batch_size", 8))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        "\n"
        + format_dynamic_training_banner(
            model=model,
            cfg=cfg,
            run_id=args.run_id,
            n_params=n_params,
            device=device,
            epochs=epochs,
            bs=bs,
            lr_init=lr_init,
            out_dir=out_dir,
        ),
        flush=True,
    )

    backbone_scale = float(cfg.get("train", {}).get("backbone_lr_scale", cfg.get("model", {}).get("backbone_lr_scale", 0.1)))
    backbone_params = list(model.encoder.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_init * backbone_scale},
            {"params": other_params, "lr": lr_init},
        ],
        weight_decay=float(cfg.get("train", {}).get("weight_decay", 1e-4)),
    )

    warmup_epochs = int(cfg.get("train", {}).get("warmup_epochs", 5))
    scheduler = make_scheduler(optimizer, epochs, warmup_epochs)
    amp = bool(cfg.get("train", {}).get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_cfg = make_loss_cfg(cfg)
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 500.0))
    eval_every = int(cfg.get("train", {}).get("eval_every", 10))
    density_bins = tuple(float(x) for x in cfg.get("eval", {}).get("density_bins", [100.0, 500.0]))
    patience = int(cfg.get("train", {}).get("patience", 0)) if cfg.get("train", {}).get("early_stopping", True) else 0

    solver_warmup_epochs = int(cfg.get("train", {}).get("solver_warmup_epochs", 5))
    solver_ramp_epochs = int(cfg.get("train", {}).get("solver_ramp_epochs", 20))

    start_epoch = 0
    best_mae = float("inf")
    epochs_without_improvement = 0

    if resume_ckpt is not None:
        ckpt = resume_ckpt
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "rng_state" in ckpt:
            load_rng_state(ckpt["rng_state"], strict=deterministic)
            print("Restored exact RNG states (random, numpy, torch, cuda)")
        if ema_manager.restore(ckpt):
            print(f"Restored EMA shadow state from checkpoint ({len(ema_manager.state)} tensors).")
        start_epoch = int(ckpt.get("epoch", 0))
        best_mae = float(ckpt.get("best_mae", float("inf")))
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0)) if patience > 0 else 0
        print(f"Resumed from epoch index {start_epoch} (next display: epoch {start_epoch + 1}), best MAE: {best_mae:.2f}")

    ckpt_manager = CheckpointManager(
        out_dir=out_dir,
        cfg=cfg,
        config_hash=run_config_hash,
        patience=patience,
        solver_warmup_epochs=solver_warmup_epochs,
        solver_ramp_epochs=solver_ramp_epochs,
        best_mae=best_mae,
        epochs_without_improvement=epochs_without_improvement,
    )

    log_csv = out_dir / "train_log.csv"
    if not log_csv.exists() or start_epoch == 0:
        with open(log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
    else:
        try:
            with open(log_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows_to_keep = [r for r in reader if int(r.get("epoch", 0)) <= start_epoch]
            with open(log_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows_to_keep)
        except Exception as e:
            print(f"Warning: could not filter train_log.csv on resume ({e}), proceeding with append.")

    for epoch in range(start_epoch, epochs):
        if epoch < solver_warmup_epochs:
            solver_strength = 0.0
        else:
            solver_strength = min(1.0, float(epoch - solver_warmup_epochs + 1) / max(1.0, float(solver_ramp_epochs)))
        model.set_solver_strength(solver_strength)

        cur_lr_bb = optimizer.param_groups[0]["lr"]
        cur_lr_main = optimizer.param_groups[1]["lr"]

        loss_avgs, diag_summary = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_cfg=loss_cfg,
            device=device,
            amp=amp,
            grad_clip=grad_clip,
            uniform_reliability=uniform_reliability,
            solver_strength=solver_strength,
            ema_manager=ema_manager,
            teacher_model=teacher_model,
            kd_loss_fn=kd_loss_fn,
        )

        train_allocation = loss_avgs.get("allocation", 0.0)
        train_dm16 = loss_avgs.get("dm_16", train_allocation if "dm_32" not in loss_avgs else 0.0)

        row_log: dict[str, Any] = {
            "epoch": epoch + 1,
            "lr_backbone": cur_lr_bb,
            "lr_main": cur_lr_main,
            "solver_strength": solver_strength,
            "train_total": loss_avgs.get("total", 0.0),
            "train_count": loss_avgs.get("count", 0.0),
            "train_flat_dm16": train_allocation,
            "train_allocation": train_allocation,
            "train_dm16": train_dm16,
            "train_dm32": loss_avgs.get("dm_32", 0.0),
            "train_dm64": loss_avgs.get("dm_64", 0.0),
            "train_cell": loss_avgs.get("cell", 0.0),
            "train_region_nb": loss_avgs.get("region_nb", 0.0),
            "train_hurdle_bce": loss_avgs.get("hurdle_bce", 0.0),
            "train_trunc_nb": loss_avgs.get("trunc_nb", 0.0),
            "train_kd_total": loss_avgs.get("kd_total", 0.0),
            "train_kd_spatial": loss_avgs.get("kd_spatial", 0.0),
            "train_kd_count": loss_avgs.get("kd_count", 0.0),
            **diag_summary,
        }

        # Periodic evaluation
        is_eval_epoch = (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs
        if is_eval_epoch and val_loader is not None:
            with ema_manager.swap_into(model, device):
                val_metrics = evaluate_v3(
                    model, val_loader, device,
                    uniform_reliability=uniform_reliability,
                    density_bins=density_bins,
                )

            row_log.update({
                "val_mae": float(val_metrics["MAE"]),
                "val_rmse": float(val_metrics["RMSE"]),
                "val_nae": float(val_metrics["NAE"]),
                "val_bias": float(val_metrics["Bias"]),
                "val_game0": float(val_metrics["GAME0"]),
                "val_game1": float(val_metrics["GAME1"]),
                "val_game2": float(val_metrics["GAME2"]),
                "val_game3": float(val_metrics["GAME3"]),
                "val_mae_sparse": float(val_metrics.get("mae_sparse", 0.0)),
                "val_mae_moderate": float(val_metrics.get("mae_moderate", 0.0)),
                "val_mae_dense": float(val_metrics.get("mae_dense", 0.0)),
                "pearson_rate_var_error": float(val_metrics.get("pearson_rate_var_error", 0.0)),
                "spearman_rate_var_error": float(val_metrics.get("spearman_rate_var_error", 0.0)),
                "spearman_weight_error": float(val_metrics.get("spearman_weight_error", 0.0)),
                "spearman_pred_weight_error": float(val_metrics.get("spearman_pred_weight_error", 0.0)),
                "spearman_rate_var_error_32": float(val_metrics.get("spearman_rate_var_error_32", 0.0)),
                "spearman_rate_var_error_64": float(val_metrics.get("spearman_rate_var_error_64", 0.0)),
                "spearman_rate_var_error_128": float(val_metrics.get("spearman_rate_var_error_128", 0.0)),
                "mean_std_residual": float(val_metrics.get("mean_std_residual", 0.0)),
                "coverage_50": float(val_metrics.get("coverage_50", 0.0)),
                "coverage_80": float(val_metrics.get("coverage_80", 0.0)),
                "coverage_95": float(val_metrics.get("coverage_95", 0.0)),
                "calib_gap_50": float(val_metrics.get("calib_gap_50", 0.0)),
                "calib_gap_80": float(val_metrics.get("calib_gap_80", 0.0)),
                "calib_gap_95": float(val_metrics.get("calib_gap_95", 0.0)),
                "dispersion_sat_low_fraction": float(val_metrics.get("dispersion_sat_low_fraction", 0.0)),
                "dispersion_sat_high_fraction": float(val_metrics.get("dispersion_sat_high_fraction", 0.0)),
                "solver_help_fraction": float(val_metrics.get("solver_help_fraction", 0.0)),
                "solver_harm_fraction": float(val_metrics.get("solver_harm_fraction", 0.0)),
                "energy_monotonic_fraction": float(val_metrics.get("energy_monotonic_fraction", 1.0)),
                "mae_reg_y0": float(val_metrics.get("mae_reg_y0", 0.0)),
                "mae_reg_y1": float(val_metrics.get("mae_reg_y1", 0.0)),
                "mae_reg_y2": float(val_metrics.get("mae_reg_y2", 0.0)),
            })

            cur_mae = float(val_metrics["MAE"])
            is_best, status_tag = ckpt_manager.evaluate_and_save_best(
                epoch=epoch + 1,
                cur_mae=cur_mae,
                val_metrics=val_metrics,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                solver_strength=solver_strength,
                ema_manager=ema_manager,
                eval_every=eval_every,
            )

            sparse_s = f"Sparse(<=100): {val_metrics.get('mae_sparse', 0.0):.2f}" if "mae_sparse" in val_metrics else ""
            mod_s = f"Mod(101-500): {val_metrics.get('mae_moderate', 0.0):.2f}" if "mae_moderate" in val_metrics else ""
            dense_s = f"Dense(>500): {val_metrics.get('mae_dense', 0.0):.2f}" if "mae_dense" in val_metrics else ""
            strata_str = f" | {sparse_s} | {mod_s} | {dense_s}" if sparse_s else ""

            cov_50 = f"{val_metrics.get('coverage_50', 0.0)*100:.1f}%" if "coverage_50" in val_metrics else "N/A"
            cov_80 = f"{val_metrics.get('coverage_80', 0.0)*100:.1f}%" if "coverage_80" in val_metrics else "N/A"
            cov_95 = f"{val_metrics.get('coverage_95', 0.0)*100:.1f}%" if "coverage_95" in val_metrics else "N/A"
            hurdle_str = f" | h_bce: {loss_avgs.get('hurdle_bce', 0.0):.4f}" if loss_avgs.get("hurdle_bce", 0.0) > 0 else ""

            if loss_cfg.use_multiscale_dm or loss_cfg.use_hierarchical_dm:
                alloc_repr = f"alloc: {train_allocation:.3f} (16:{train_dm16:.3f}, 32:{row_log['train_dm32']:.3f}, 64:{row_log['train_dm64']:.3f})"
            else:
                alloc_repr = f"alloc(dm16): {train_allocation:.3f}"

            print(
                f"\n{'='*92}\n"
                f"  EPOCH [{epoch+1:04d}/{epochs:04d}] PERIODIC EVALUATION (182 test samples)\n"
                f"{'-'*92}\n"
                f"  Train Loss    : {row_log['train_total']:.4f} [cnt: {row_log['train_count']:.2f}, {alloc_repr}, cell: {row_log['train_cell']:.3f}, reg_nb: {row_log['train_region_nb']:.3f}{hurdle_str}]\n"
                f"  Solver / W    : Str: {solver_strength:.2f} | E_red: {diag_summary['solver_energy_reduction']*100:.1f}% | W_pred: {row_log['region_weight_mean']:.2f} (std: {row_log['region_weight_std']:.2f}) | W_solv: {row_log['solver_weight_mean']:.2f}\n"
                f"  Val Metrics   : MAE: {cur_mae:.2f} | RMSE: {float(val_metrics['RMSE']):.2f} | NAE: {float(val_metrics['NAE']):.3f} | Bias: {float(val_metrics['Bias']):+.2f}\n"
                f"  GAME Hierarchy: G0: {float(val_metrics['GAME0']):.2f} | G1: {float(val_metrics['GAME1']):.2f} | G2: {float(val_metrics['GAME2']):.2f} | G3: {float(val_metrics['GAME3']):.2f}{strata_str}\n"
                f"  Uncertainty   : Coverage: 50%={cov_50}, 80%={cov_80}, 95%={cov_95} | Median Disp: {row_log['region_dispersion_p50']:.1f}\n"
                f"  Checkpoint    : Current Val MAE: {cur_mae:.2f} | Best Val MAE: {ckpt_manager.best_mae:.2f}{status_tag}\n"
                f"{'='*92}\n",
                flush=True,
            )
        else:
            if loss_cfg.use_multiscale_dm or loss_cfg.use_hierarchical_dm:
                alloc_repr = f"alloc: {train_allocation:.3f} (16:{train_dm16:.3f}, 32:{row_log['train_dm32']:.3f}, 64:{row_log['train_dm64']:.3f})"
            else:
                alloc_repr = f"alloc(dm16): {train_allocation:.3f}"

            hurdle_str = f" | h_bce: {loss_avgs.get('hurdle_bce', 0.0):.4f}" if loss_avgs.get("hurdle_bce", 0.0) > 0 else ""

            print(
                f"[{epoch+1:04d}/{epochs:04d}] "
                f"Loss: {row_log['train_total']:.4f} [cnt: {row_log['train_count']:.2f}, {alloc_repr}, cell: {row_log['train_cell']:.3f}, reg_nb: {row_log['train_region_nb']:.3f}{hurdle_str}] | "
                f"SolvStr: {solver_strength:.2f} | "
                f"Disp: {row_log['region_dispersion_p50']:.1f} | "
                f"W_pred: {row_log['region_weight_mean']:.2f} | W_solv: {row_log['solver_weight_mean']:.2f}",
                flush=True,
            )

        ckpt_manager.save_last(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            solver_strength=solver_strength,
            ema_manager=ema_manager,
        )

        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDNAMES, extrasaction="ignore")
            writer.writerow(row_log)

        if is_eval_epoch and val_loader is not None and ckpt_manager.should_stop_early:
            print(
                f"\n[Early Stopping] No validation MAE improvement for {ckpt_manager.epochs_without_improvement} epochs "
                f"(patience={patience}). Terminating training at epoch {epoch + 1}.",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
