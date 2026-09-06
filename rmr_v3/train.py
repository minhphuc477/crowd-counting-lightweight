from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

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
from rmr_core.metrics import game_physical_image, game_single, summarize_predictions
from rmr_core.training import load_rng_state, make_scheduler, save_rng_state, seed_everything

from .diagnostics import (
    compute_dispersion_saturation,
    compute_nb_interval_coverage,
    compute_reliability_correlations,
    compute_solver_trajectory_diagnostics,
    compute_uncertainty_calibration_bins,
    regional_reliability_rows,
    summarize_diagnostics,
)
from .losses import RMRv3LossConfig, compute_rmr_v3_losses
from .model import RMRv3, RMRv3Config




def make_model(cfg: dict) -> tuple[RMRv3, bool]:
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("train", {})

    output_stride = int(m_cfg.get("output_stride", 4))
    feature_width = int(m_cfg.get("feature_width", 32))
    backbone_name = str(m_cfg.get("backbone", m_cfg.get("backbone_name", "mobilenetv4_conv_small_050.e3000_r224_in1k")))
    pretrained = bool(m_cfg.get("pretrained", True))
    backbone_lr_scale = float(m_cfg.get("backbone_lr_scale", t_cfg.get("backbone_lr_scale", 0.1)))

    init_m0 = float(m_cfg.get("init_m0", 0.015763))
    region_sizes_px = tuple(int(x) for x in m_cfg.get("region_sizes_px", (32, 64, 128)))
    region_overlap = float(m_cfg.get("region_overlap", 0.5))
    include_full_image = bool(m_cfg.get("include_full_image", False))

    iterations = int(m_cfg.get("iterations", 2))
    omega = float(m_cfg.get("omega", m_cfg.get("sirt_omega", 1.0)))
    residual_clip = float(m_cfg.get("residual_clip", 0.0))

    dispersion_init = float(m_cfg.get("dispersion_init", 50.0))
    dispersion_min = float(m_cfg.get("dispersion_min", 0.5))
    dispersion_max = float(m_cfg.get("dispersion_max", 500.0))

    reliability_mode = str(m_cfg.get("reliability_mode", "nb_rate_variance"))
    reliability_rate_std_floor = float(m_cfg.get("reliability_rate_std_floor", 0.01))
    reliability_weight_min = float(m_cfg.get("reliability_weight_min", 0.25))
    reliability_weight_max = float(m_cfg.get("reliability_weight_max", 4.0))
    normalize_reliability_within_scale = bool(m_cfg.get("normalize_reliability_within_scale", True))

    detach_region_mean_in_solver = bool(m_cfg.get("detach_region_mean_in_solver", True))
    detach_reliability_in_solver = bool(m_cfg.get("detach_reliability_in_solver", True))

    uniform_reliability = bool(m_cfg.get("uniform_reliability", False))

    config = RMRv3Config(
        output_stride=output_stride,
        feature_width=feature_width,
        backbone_name=backbone_name,
        pretrained=pretrained,
        backbone_lr_scale=backbone_lr_scale,
        init_m0=init_m0,
        region_sizes_px=region_sizes_px,
        region_overlap=region_overlap,
        include_full_image=include_full_image,
        iterations=iterations,
        omega=omega,
        residual_clip=residual_clip,
        dispersion_init=dispersion_init,
        dispersion_min=dispersion_min,
        dispersion_max=dispersion_max,
        reliability_mode=reliability_mode,
        reliability_rate_std_floor=reliability_rate_std_floor,
        reliability_weight_min=reliability_weight_min,
        reliability_weight_max=reliability_weight_max,
        normalize_reliability_within_scale=normalize_reliability_within_scale,
        detach_region_mean_in_solver=detach_region_mean_in_solver,
        detach_reliability_in_solver=detach_reliability_in_solver,
    )

    model = RMRv3(config)
    return model, uniform_reliability


def make_loss_cfg(cfg: dict) -> RMRv3LossConfig:
    l_cfg = cfg.get("loss", {})
    return RMRv3LossConfig(
        lambda_count=float(l_cfg.get("lambda_count", 1.0)),
        lambda_flat_dm16=float(l_cfg.get("lambda_flat_dm16", 1.0)),
        lambda_cell=float(l_cfg.get("lambda_cell", 0.25)),
        lambda_region_nb=float(l_cfg.get("lambda_region_nb", 0.20)),
        count_loss_mode=str(l_cfg.get("count_loss_mode", "nb")),
        count_nb_dispersion=float(l_cfg.get("count_nb_dispersion", 50.0)),
        kappa_flat16=float(l_cfg.get("kappa_flat16", 20.0)),
        normalize_flat_dm16=bool(l_cfg.get("normalize_flat_dm16", True)),
        cell_beta=float(l_cfg.get("cell_beta", 1.0)),
    )


@torch.no_grad()
def evaluate_v3(
    model: RMRv3,
    loader: DataLoader,
    device: torch.device,
    uniform_reliability: bool = False,
    density_bins: tuple[float, float] = (100.0, 500.0),
) -> dict:
    model.eval()
    pred_rows = []
    all_diag_rows = []
    traj_rows = []
    output_stride = getattr(model.cfg, "output_stride", 4)

    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["target_y"].to(device)

            out = model(image, uniform_reliability=uniform_reliability, solver_strength=1.0)
            y = out["y"][0]
            pred = float(y.sum().item())
            gt = float(target.sum().item())

            row = {"gt": gt, "pred": pred}
            if "points" in sample and "height" in sample and "width" in sample:
                game_dict = game_physical_image(
                    y,
                    sample["points"],
                    image_h=sample["height"],
                    image_w=sample["width"],
                    stride=output_stride,
                    levels=(0, 1, 2, 3),
                )
                for level in range(4):
                    row[f"GAME{level}"] = game_dict[level]
            else:
                for level in range(4):
                    row[f"GAME{level}"] = game_single(y, target, level)
            pred_rows.append(row)

            d_rows = regional_reliability_rows(out, target.unsqueeze(0))
            all_diag_rows.extend(d_rows)

            t_diag = compute_solver_trajectory_diagnostics(out, target.unsqueeze(0))
            if t_diag:
                traj_rows.append(t_diag)

    summary = summarize_predictions(pred_rows)

    # Provide lowercase aliases for robustness
    summary["mae"] = summary["MAE"]
    summary["rmse"] = summary["RMSE"]
    summary["nae"] = summary["NAE"]
    summary["bias"] = summary["Bias"]

    corrs = compute_reliability_correlations(all_diag_rows)
    summary.update(corrs)

    calib = compute_uncertainty_calibration_bins(all_diag_rows)
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

    # Density-stratified MAE
    gts = np.array([r["gt"] for r in pred_rows])
    preds = np.array([r["pred"] for r in pred_rows])
    aes = np.abs(preds - gts)

    lo, hi = density_bins
    sparse_mask = gts <= lo
    mod_mask = (gts > lo) & (gts <= hi)
    dense_mask = gts > hi

    summary["mae_sparse"] = float(np.mean(aes[sparse_mask])) if np.any(sparse_mask) else 0.0
    summary["mae_moderate"] = float(np.mean(aes[mod_mask])) if np.any(mod_mask) else 0.0
    summary["mae_dense"] = float(np.mean(aes[dense_mask])) if np.any(dense_mask) else 0.0

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RMR-v3 (RW-RMR)")
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--resume", default=None, help="Resume from checkpoint path")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--disable-early-stopping", action="store_true", default=False)
    ap.add_argument("--deterministic", action="store_true", default=False, help="Enable strict determinism")
    ap.add_argument("--overwrite", action="store_true", default=False)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
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
        cfg["output_dir"] = args.output_dir

    seed = int(cfg.get("seed", 42))
    deterministic = bool(args.deterministic or cfg.get("train", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    if "init_m0" not in cfg.get("model", {}) and "train_manifest" in cfg.get("data", {}):
        stride = int(cfg.get("model", {}).get("output_stride", 4))
        cfg.setdefault("model", {})["init_m0"] = compute_manifest_density(
            cfg["data"]["train_manifest"],
            output_stride=stride,
        )

    out_dir = Path(cfg["output_dir"])
    if out_dir.exists() and not args.resume:
        existing = list(out_dir.glob("*.pt")) + list(out_dir.glob("*.csv"))
        if existing:
            if not args.overwrite:
                raise RuntimeError(
                    f"Output directory '{out_dir}' already exists with artifacts: "
                    f"{[f.name for f in existing]}. Use --overwrite or --resume."
                )
            for f in existing:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_ds = CrowdManifestDataset(
        cfg["data"]["train_manifest"],
        train=True,
        output_stride=cfg.get("model", {}).get("output_stride", 4),
        crop_size=cfg.get("data", {}).get("crop_size", 512),
        scale_range=tuple(cfg.get("data", {}).get("scale_range", [0.75, 1.25])),
    )
    val_manifest = cfg.get("data", {}).get("val_manifest")
    val_ds = None if not val_manifest else CrowdManifestDataset(
        val_manifest,
        train=False,
        output_stride=cfg.get("model", {}).get("output_stride", 4),
    )

    workers = int(cfg.get("train", {}).get("workers", 0))
    pin_mem = bool(cfg.get("train", {}).get("pin_memory", False))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("train", {}).get("batch_size", 8),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=(workers > 0),
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

    epochs = int(cfg.get("train", {}).get("epochs", 1000))
    lr_init = float(cfg.get("train", {}).get("lr", 1e-4))
    bs = int(cfg.get("train", {}).get("batch_size", 8))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    variant_name = "V3-A (Probabilistic Uniform)" if uniform_reliability else "V3-B (Reliability Weighted)"
    print(
        f"\n{'=' * 80}\n"
        f"  RMR-v3 Training Initialized\n"
        f"  Variant: {variant_name} | Parameters: {n_params:,} | Device: {device}\n"
        f"  Epochs: {epochs} | Batch Size: {bs} | Initial LR: {lr_init:.2e}\n"
        f"  Output Directory: {out_dir}\n"
        f"{'=' * 80}\n",
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

    solver_warmup_epochs = int(cfg.get("train", {}).get("solver_warmup_epochs", 5))
    solver_ramp_epochs = int(cfg.get("train", {}).get("solver_ramp_epochs", 20))

    log_csv = out_dir / "train_log.csv"
    fieldnames = [
        "epoch", "lr_backbone", "lr_main", "solver_strength",
        "train_total", "train_count", "train_flat_dm16", "train_cell", "train_region_nb",
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

    start_epoch = 0
    best_mae = float("inf")
    patience = int(cfg.get("train", {}).get("patience", 0)) if cfg.get("train", {}).get("early_stopping", True) else 0
    epochs_without_improvement = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "rng_state" in ckpt:
            load_rng_state(ckpt["rng_state"])
            print("Restored exact RNG states (random, numpy, torch, cuda)")
        # Exactly resume at the next epoch index
        start_epoch = int(ckpt.get("epoch", 0))
        best_mae = float(ckpt.get("best_mae", float("inf")))
        print(f"Resumed from epoch index {start_epoch} (next display: epoch {start_epoch + 1}), best MAE: {best_mae:.2f}")

    if not log_csv.exists() or start_epoch == 0:
        with open(log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

    for epoch in range(start_epoch, epochs):
        # Solver warmup and ramp protocol:
        # - epoch 0..solver_warmup_epochs-1: solver_strength = 0
        # - epoch solver_warmup_epochs..solver_warmup_epochs+solver_ramp_epochs-1: linear ramp 0->1
        # - epoch >= solver_warmup_epochs+solver_ramp_epochs: full solver strength 1.0
        if epoch < solver_warmup_epochs:
            solver_strength = 0.0
        else:
            solver_strength = min(1.0, float(epoch - solver_warmup_epochs + 1) / max(1.0, float(solver_ramp_epochs)))
        model.set_solver_strength(solver_strength)

        model.train()
        total_loss_accum = 0.0
        count_loss_accum = 0.0
        dm16_loss_accum = 0.0
        cell_loss_accum = 0.0
        reg_nb_loss_accum = 0.0

        mu_means = []
        disp_means = []
        all_disps = []
        all_pred_weights = []
        all_solver_weights = []
        e_befores = []
        e_afters = []

        w_scale_32 = []
        w_scale_64 = []
        w_scale_128 = []

        cur_lr_bb = optimizer.param_groups[0]["lr"]
        cur_lr_main = optimizer.param_groups[1]["lr"]

        for batch in train_loader:
            # Correct collate_train keys: "image" and "target_y"
            images = batch["image"].to(device)
            targets = batch["target_y"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(images, uniform_reliability=uniform_reliability, solver_strength=solver_strength)
                losses = compute_rmr_v3_losses(outputs, targets, loss_cfg)
                loss = losses["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss_accum += float(loss.item())
            count_loss_accum += float(losses["count"].item())
            dm16_loss_accum += float(losses["flat_dm16"].item())
            cell_loss_accum += float(losses["cell"].item())
            reg_nb_loss_accum += float(losses["region_nb"].item())

            # Log tensors
            with torch.no_grad():
                mu_means.append(float(outputs["b_region"].mean().item()))
                disp_means.append(float(outputs["region_dispersion"].mean().item()))
                all_disps.extend(outputs["region_dispersion"].float().cpu().numpy().flatten().tolist())
                all_pred_weights.extend(outputs["region_weight"].float().cpu().numpy().flatten().tolist())
                all_solver_weights.extend(outputs["solver_region_weight"].float().cpu().numpy().flatten().tolist())

                et = outputs.get("energy_trace", [])
                if et:
                    e_befores.append(float(et[0]["before"].mean().item()))
                    e_afters.append(float(et[-1]["after"].mean().item()))

                regions = outputs["regions"]
                w = outputs["solver_region_weight"]
                m32 = regions.scale_id == 0
                m64 = regions.scale_id == 1
                m128 = regions.scale_id == 2
                if m32.any():
                    w_scale_32.append(float(w[..., m32].mean().item()))
                if m64.any():
                    w_scale_64.append(float(w[..., m64].mean().item()))
                if m128.any():
                    w_scale_128.append(float(w[..., m128].mean().item()))

        scheduler.step()

        num_batches = max(1, len(train_loader))
        train_total = total_loss_accum / num_batches
        train_count = count_loss_accum / num_batches
        train_flat_dm16 = dm16_loss_accum / num_batches
        train_cell = cell_loss_accum / num_batches
        train_region_nb = reg_nb_loss_accum / num_batches

        disps_np = np.array(all_disps) if all_disps else np.array([50.0])
        pred_weights_np = np.array(all_pred_weights) if all_pred_weights else np.array([1.0])
        solver_weights_np = np.array(all_solver_weights) if all_solver_weights else np.array([1.0])

        e_b = float(np.mean(e_befores)) if e_befores else 0.0
        e_a = float(np.mean(e_afters)) if e_afters else 0.0
        e_red = (e_b - e_a) / max(e_b, 1e-8)

        w_min_val = model.cfg.reliability_weight_min
        w_max_val = model.cfg.reliability_weight_max
        w_low_frac = float(np.mean(pred_weights_np <= w_min_val + 1e-4))
        w_high_frac = float(np.mean(pred_weights_np >= w_max_val - 1e-4))

        row_log = {
            "epoch": epoch + 1,
            "lr_backbone": cur_lr_bb,
            "lr_main": cur_lr_main,
            "solver_strength": solver_strength,
            "train_total": train_total,
            "train_count": train_count,
            "train_flat_dm16": train_flat_dm16,
            "train_cell": train_cell,
            "train_region_nb": train_region_nb,
            "region_mu_mean": float(np.mean(mu_means)) if mu_means else 0.0,
            "region_dispersion_mean": float(np.mean(disp_means)) if disp_means else 50.0,
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
            "weight_mean_32": float(np.mean(w_scale_32)) if w_scale_32 else 1.0,
            "weight_mean_64": float(np.mean(w_scale_64)) if w_scale_64 else 1.0,
            "weight_mean_128": float(np.mean(w_scale_128)) if w_scale_128 else 1.0,
        }

        # Periodic evaluation
        is_eval_epoch = (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs
        if is_eval_epoch and val_loader is not None:
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
            solver_engaged = solver_strength >= 1.0 or epoch + 1 >= solver_warmup_epochs + solver_ramp_epochs
            # Guard: only update best_mae after solver ramp has fully engaged
            is_best = (cur_mae < best_mae) and solver_engaged
            if is_best:
                best_mae = cur_mae
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "rng_state": save_rng_state(),
                        "solver_strength": solver_strength,
                        "config": cfg,
                        "best_mae": best_mae,
                    },
                    out_dir / "best_val_mae.pt",
                )
                (out_dir / "eval_val").mkdir(parents=True, exist_ok=True)
                (out_dir / "eval_val" / "summary.json").write_text(json.dumps(val_metrics, indent=2))
            else:
                if solver_engaged:
                    epochs_without_improvement += eval_every

            print(
                f"[{epoch+1:04d}/{epochs:04d}] "
                f"Loss: {train_total:.4f} [cnt: {train_count:.2f}, dm16: {train_flat_dm16:.3f}, cell: {train_cell:.3f}, reg_nb: {train_region_nb:.3f}] | "
                f"SolvStr: {solver_strength:.2f} | "
                f"W_pred: {row_log['region_weight_mean']:.2f} (std: {row_log['region_weight_std']:.2f}) | "
                f"W_solv: {row_log['solver_weight_mean']:.2f} | "
                f"E_red: {e_red*100:.1f}% | "
                f"VAL MAE: {cur_mae:.2f} (Best: {best_mae:.2f})",
                flush=True,
            )
        else:
            print(
                f"[{epoch+1:04d}/{epochs:04d}] "
                f"Loss: {train_total:.4f} [cnt: {train_count:.2f}, dm16: {train_flat_dm16:.3f}, cell: {train_cell:.3f}, reg_nb: {train_region_nb:.3f}] | "
                f"SolvStr: {solver_strength:.2f} | "
                f"Disp: {row_log['region_dispersion_p50']:.1f} | "
                f"W_pred: {row_log['region_weight_mean']:.2f} | W_solv: {row_log['solver_weight_mean']:.2f}",
                flush=True,
            )

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_state": save_rng_state(),
                "solver_strength": solver_strength,
                "config": cfg,
                "best_mae": best_mae,
            },
            out_dir / "last.pt",
        )

        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row_log)

        if is_eval_epoch and val_loader is not None and patience > 0 and epochs_without_improvement >= patience:
            print(
                f"\n[Early Stopping] No validation MAE improvement for {epochs_without_improvement} epochs "
                f"(patience={patience}). Terminating training at epoch {epoch + 1}.",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
