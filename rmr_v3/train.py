from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

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

import numpy as np
import torch
import torch.nn.functional as F
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

from rmr_v3.config import compute_config_hash, validate_resume_compatibility, validate_v3_config
from rmr_v3.diagnostics import (
    compute_dispersion_saturation,
    compute_nb_interval_coverage,
    compute_reliability_correlations,
    compute_solver_trajectory_diagnostics,
    compute_uncertainty_calibration_bins,
    regional_reliability_rows,
)
from rmr_v3.kd import DensityMapKDLoss
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config


TRAIN_LOG_FIELDNAMES: list[str] = [
    "epoch", "lr_backbone", "lr_main", "solver_strength",
    "train_total", "train_count", "train_flat_dm16", "train_allocation",
    "train_dm16", "train_dm32", "train_dm64",
    "train_cell", "train_region_nb", "train_hurdle_bce", "train_trunc_nb",
    "train_curvature", "train_hard_bg", "train_fg_bce", "train_scale_align",
    "train_kd_total", "train_kd_spatial", "train_kd_count",
    "region_mu_mean", "region_dispersion_mean", "region_dispersion_p10", "region_dispersion_p50", "region_dispersion_p90",
    "region_weight_mean", "region_weight_std", "region_weight_min", "region_weight_max",
    "solver_weight_mean", "solver_weight_std",
    "weight_clip_low_fraction", "weight_clip_high_fraction",
    "solver_energy_before", "solver_energy_after", "solver_energy_reduction",
    "weight_mean_16", "weight_mean_32", "weight_mean_64", "weight_mean_64_32", "weight_mean_128",
    "scale_pi_16", "scale_pi_32", "scale_pi_64", "scale_pi_64_32", "scale_pi_128",
    "val_mae", "val_rmse", "val_nae", "val_bias",
    "val_game0", "val_game1", "val_game2", "val_game3",
    "val_mae_sparse", "val_mae_moderate", "val_mae_dense",
    "pearson_rate_var_error", "spearman_rate_var_error", "spearman_weight_error",
    "spearman_pred_weight_error",
    "spearman_rate_var_error_16", "spearman_rate_var_error_32", "spearman_rate_var_error_64", "spearman_rate_var_error_128",
    "mean_std_residual",
    "coverage_50", "coverage_80", "coverage_95",
    "calib_gap_50", "calib_gap_80", "calib_gap_95",
    "dispersion_sat_low_fraction", "dispersion_sat_high_fraction",
    "solver_help_fraction", "solver_harm_fraction", "energy_monotonic_fraction",
    "mae_reg_y0", "mae_reg_y1", "mae_reg_y2",
    "curvature_alpha", "effective_curvature",
]


from rmr_v3.tracking import LossTracker, DiagnosticTracker, format_dynamic_training_banner
from rmr_v3.checkpoint import EMAManager, CheckpointManager

__all__ = [
    "LossTracker",
    "DiagnosticTracker",
    "format_dynamic_training_banner",
    "EMAManager",
    "CheckpointManager",
    "make_model",
    "make_loss_cfg",
    "evaluate_v3",
    "train_one_epoch",
]


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

    scale_sizes = tuple(getattr(model.cfg, "region_sizes_px", (32, 64, 128)))
    scale_map = {sid: int(s if isinstance(s, int) else s[0]) for sid, s in enumerate(scale_sizes)}

    def sample_callback(sample: dict, out: dict, y: torch.Tensor, row: dict) -> dict:
        target = sample["target_y"].to(device)
        d_rows = regional_reliability_rows(out, target.unsqueeze(0), max_regions=300)
        all_diag_rows.extend(d_rows)
        t_diag = compute_solver_trajectory_diagnostics(out, target.unsqueeze(0), scale_map=scale_map)
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

    corrs = compute_reliability_correlations(all_diag_rows, scale_map=scale_map)
    summary.update(corrs)

    calib = compute_uncertainty_calibration_bins(all_diag_rows, scale_map=scale_map)
    summary["calibration"] = calib
    summary["mean_std_residual"] = calib["mean_std_residual"]
    summary["p50_std_residual"] = calib["p50_std_residual"]
    summary["p90_std_residual"] = calib["p90_std_residual"]

    disp_min = float(getattr(model.cfg, "dispersion_min", 0.5))
    disp_max = float(getattr(model.cfg, "dispersion_max", 500.0))
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

                if getattr(loss_cfg, "dm_target", "y0") == "dual":
                    kd_y0 = kd_loss_fn(outputs["y0"], t_y)
                    kd_y = kd_loss_fn(outputs["y"], t_y)
                    kd_res = {
                        "total_kd": 0.5 * kd_y0["total_kd"] + 0.5 * kd_y["total_kd"],
                        "spatial_kl": 0.5 * kd_y0["spatial_kl"] + 0.5 * kd_y["spatial_kl"],
                        "count_kd": 0.5 * kd_y0["count_kd"] + 0.5 * kd_y["count_kd"],
                    }
                elif getattr(loss_cfg, "dm_target", "y0") == "y":
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
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            for cand in ["last.pt", "best_val_mae.pt"]:
                if (resume_path / cand).is_file():
                    resume_path = resume_path / cand
                    break
            else:
                raise FileNotFoundError(
                    f"--resume was given directory '{args.resume}', but neither 'last.pt' nor 'best_val_mae.pt' was found inside it."
                )
        try:
            resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            resume_ckpt = torch.load(resume_path, map_location="cpu")
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
                # Prioritize EMA weights if present, falling back to raw model weights or state dict
                if "ema_model" in t_ckpt:
                    t_state = t_ckpt["ema_model"]
                    tag = "ema_model"
                elif "model" in t_ckpt:
                    t_state = t_ckpt["model"]
                    tag = "model"
                else:
                    t_state = t_ckpt
                    tag = "direct_state_dict"
                teacher_model.load_state_dict(t_state)
                teacher_model.switch_to_deploy()
                teacher_model.to(device).eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
                kd_loss_fn = DensityMapKDLoss(
                    lambda_spatial_kl=float(cfg.get("loss", {}).get("lambda_kd_spatial", 1.0)),
                    lambda_count_kd=float(cfg.get("loss", {}).get("lambda_kd_count", 0.1)),
                )
                print(f"[Stage 3 KD] Teacher loaded and frozen successfully from '{tp}' (weights: {tag}).")
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
    wd = float(cfg.get("train", {}).get("weight_decay", 1e-4))

    # Decouple parameter groups: exclude 1D tensors (biases, GroupNorm scale/shift)
    # and calibrated priors (e.g. fine_head bias b0, dispersion bias, tau) from weight decay.
    # Applying weight decay to b0 (-4.1422) pulls it toward 0, which exponentially inflates
    # baseline background density and causes massive positive bias drift in late epochs.
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

    optimizer = torch.optim.AdamW(
        [
            {"params": bb_decay, "lr": lr_init * backbone_scale, "weight_decay": wd},
            {"params": bb_no_decay, "lr": lr_init * backbone_scale, "weight_decay": 0.0},
            {"params": other_decay, "lr": lr_init, "weight_decay": wd},
            {"params": other_no_decay, "lr": lr_init, "weight_decay": 0.0},
        ],
    )

    warmup_epochs = int(cfg.get("train", {}).get("warmup_epochs", 5))
    min_lr_ratio = float(cfg.get("train", {}).get("min_lr_ratio", 0.05))
    scheduler = make_scheduler(optimizer, epochs, warmup_epochs, min_lr_ratio=min_lr_ratio)
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
            "train_curvature": loss_avgs.get("curvature", 0.0),
            "train_hard_bg": loss_avgs.get("hard_bg", 0.0),
            "train_fg_bce": loss_avgs.get("fg_bce", 0.0),
            "train_scale_align": loss_avgs.get("scale_align", 0.0),
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
                "spearman_rate_var_error_16": float(val_metrics.get("spearman_rate_var_error_16", 0.0)),
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
                "curvature_alpha": float(model.fine_head.curvature_alpha.item()) if getattr(model.fine_head, "density_curvature", False) else 0.0,
                "effective_curvature": float(F.softplus(model.fine_head.curvature_alpha).item()) if getattr(model.fine_head, "density_curvature", False) else 0.0,
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

            v11_str = ""
            if loss_avgs.get("curvature", 0.0) > 0:
                v11_str += f" | curv: {loss_avgs.get('curvature', 0.0):.4f}"
            if loss_avgs.get("hard_bg", 0.0) > 0:
                v11_str += f" | h_bg: {loss_avgs.get('hard_bg', 0.0):.4f}"
            if loss_avgs.get("fg_bce", 0.0) > 0:
                v11_str += f" | fg_bce: {loss_avgs.get('fg_bce', 0.0):.4f}"
            if loss_avgs.get("scale_align", 0.0) > 0:
                v11_str += f" | sc_aln: {loss_avgs.get('scale_align', 0.0):.4f}"

            print(
                f"\n{'='*92}\n"
                f"  EPOCH [{epoch+1:04d}/{epochs:04d}] PERIODIC EVALUATION (182 test samples)\n"
                f"{'-'*92}\n"
                f"  Train Loss    : {row_log['train_total']:.4f} [cnt: {row_log['train_count']:.2f}, {alloc_repr}, cell: {row_log['train_cell']:.3f}, reg_nb: {row_log['train_region_nb']:.3f}{hurdle_str}{v11_str}]\n"
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
            v11_str = ""
            if loss_avgs.get("curvature", 0.0) > 0:
                v11_str += f" | curv: {loss_avgs.get('curvature', 0.0):.4f}"
            if loss_avgs.get("hard_bg", 0.0) > 0:
                v11_str += f" | h_bg: {loss_avgs.get('hard_bg', 0.0):.4f}"
            if loss_avgs.get("fg_bce", 0.0) > 0:
                v11_str += f" | fg_bce: {loss_avgs.get('fg_bce', 0.0):.4f}"
            if loss_avgs.get("scale_align", 0.0) > 0:
                v11_str += f" | sc_aln: {loss_avgs.get('scale_align', 0.0):.4f}"

            print(
                f"[{epoch+1:04d}/{epochs:04d}] "
                f"Loss: {row_log['train_total']:.4f} [cnt: {row_log['train_count']:.2f}, {alloc_repr}, cell: {row_log['train_cell']:.3f}, reg_nb: {row_log['train_region_nb']:.3f}{hurdle_str}{v11_str}] | "
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
