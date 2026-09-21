from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval, collate_train, compute_manifest_density
from rmr_core.training import (
    get_git_info, load_rng_state, make_generator, make_scheduler,
    safe_torch_load, seed_everything, seed_worker,
)

from rmr_v3.checkpoint import CheckpointManager, EMAManager
from rmr_v3.config import compute_config_hash, validate_resume_compatibility, validate_v3_config
from rmr_v3.engine import (
    build_optimizer,
    evaluate_v3,
    make_loss_cfg,
    make_model,
    train_one_epoch,
)
from rmr_v3.kd import DensityMapKDLoss
from rmr_v3.reporting import TRAIN_LOG_FIELDNAMES, format_epoch_row, format_eval_block
from rmr_v3.tracking import format_dynamic_training_banner


def run_training_loop(cfg: dict[str, Any], args: Any) -> None:
    """Orchestrate dataset loading, model initialization, optimization, and training loop."""
    validate_v3_config(cfg)

    seed = int(cfg.get("seed", 42))
    deterministic = bool(
        getattr(args, "deterministic", False)
        or (not getattr(args, "non_deterministic", False) and cfg.get("train", {}).get("deterministic", True))
    )
    cfg.setdefault("train", {})["deterministic"] = deterministic
    seed_everything(seed, deterministic=deterministic)

    m_cfg = cfg.setdefault("model", {})
    eff_stride = 2 if m_cfg.get("subpixel_stride2", False) else int(m_cfg.get("output_stride", 4))
    if "init_m0" not in m_cfg and "train_manifest" in cfg.get("data", {}):
        m_cfg["init_m0"] = compute_manifest_density(
            cfg["data"]["train_manifest"],
            output_stride=eff_stride,
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
        resume_ckpt = safe_torch_load(resume_path, map_location="cpu", weights_only=True)
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
        output_stride=eff_stride,
        crop_size=cfg.get("data", {}).get("crop_size", 512),
        scale_range=tuple(cfg.get("data", {}).get("scale_range", [0.75, 1.25])),
        hflip_prob=float(cfg.get("data", {}).get("hflip_prob", 0.5)),
        brightness_jitter=float(cfg.get("data", {}).get("brightness_jitter", 0.0)),
        contrast_jitter=float(cfg.get("data", {}).get("contrast_jitter", 0.0)),
        gamma_jitter=tuple(cfg.get("data", {}).get("gamma_jitter", [1.0, 1.0])),
        random_invert_prob=float(cfg.get("data", {}).get("random_invert_prob", 0.0)),
        data_root=cfg.get("data", {}).get("data_root"),
        cache_images=bool(cfg.get("data", {}).get("cache_images", True)),
        preload=bool(cfg.get("data", {}).get("preload", False)),
    )
    val_manifest = cfg.get("data", {}).get("val_manifest")
    val_ds = None if not val_manifest else CrowdManifestDataset(
        val_manifest,
        train=False,
        output_stride=eff_stride,
        data_root=cfg.get("data", {}).get("data_root"),
        cache_images=bool(cfg.get("data", {}).get("cache_images", True)),
        preload=bool(cfg.get("data", {}).get("preload", False)),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workers = int(cfg.get("train", {}).get("workers", 0))
    pin_mem = bool(cfg.get("train", {}).get("pin_memory", device.type == "cuda"))
    gen = make_generator(seed) if deterministic else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=bool(workers > 0),
        prefetch_factor=2 if workers > 0 else None,
        collate_fn=collate_train,
        drop_last=True,
        worker_init_fn=seed_worker if deterministic else None,
        generator=gen,
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )

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
                t_ckpt = safe_torch_load(tp, map_location="cpu", weights_only=True)
                t_cfg = t_ckpt.get("config", {})
                teacher_model, _ = make_model(t_cfg)
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

    optimizer = build_optimizer(model, cfg, lr_init)
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

        cur_lr_bb = next((g["lr"] for g in optimizer.param_groups if g.get("name") == "backbone_decay"), optimizer.param_groups[0]["lr"])
        cur_lr_main = next((g["lr"] for g in optimizer.param_groups if g.get("name") == "main_decay"), optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else optimizer.param_groups[-1]["lr"])

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

        row_log: dict[str, Any] = {
            "epoch": epoch + 1,
            "lr_backbone": cur_lr_bb,
            "lr_main": cur_lr_main,
            "solver_strength": solver_strength,
            "train_total": loss_avgs.get("total", 0.0),
            "train_count": loss_avgs.get("count", 0.0),
            "train_flat_dm16": loss_avgs.get("allocation", 0.0),
            "train_allocation": loss_avgs.get("allocation", 0.0),
            "train_dm16": loss_avgs.get("dm_16", loss_avgs.get("allocation", 0.0) if "dm_32" not in loss_avgs else 0.0),
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
            "train_cell_carrier": loss_avgs.get("cell_carrier", 0.0),
            "train_cell_fine": loss_avgs.get("cell_fine", 0.0),
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

            eval_map = {
                "val_mae": "MAE", "val_rmse": "RMSE", "val_nae": "NAE", "val_bias": "Bias",
                "val_game0": "GAME0", "val_game1": "GAME1", "val_game2": "GAME2", "val_game3": "GAME3",
                "val_mae_sparse": "mae_sparse", "val_mae_moderate": "mae_moderate", "val_mae_dense": "mae_dense",
            }
            extra_keys = [
                "pearson_rate_var_error", "spearman_rate_var_error", "spearman_weight_error",
                "spearman_pred_weight_error", "spearman_rate_var_error_16", "spearman_rate_var_error_32",
                "spearman_rate_var_error_64", "spearman_rate_var_error_128", "mean_std_residual",
                "coverage_50", "coverage_80", "coverage_95", "calib_gap_50", "calib_gap_80", "calib_gap_95",
                "dispersion_sat_low_fraction", "dispersion_sat_high_fraction", "solver_help_fraction",
                "solver_harm_fraction", "energy_monotonic_fraction", "mae_reg_y0", "mae_reg_y1", "mae_reg_y2",
            ]
            for lk, vk in eval_map.items():
                row_log[lk] = float(val_metrics.get(vk, 0.0))
            for k in extra_keys:
                row_log[k] = float(val_metrics.get(k, 1.0 if k == "energy_monotonic_fraction" else 0.0))
            has_curv = getattr(model.fine_head, "density_curvature", False) and hasattr(model.fine_head, "curvature_alpha")
            row_log["curvature_alpha"] = float(model.fine_head.curvature_alpha.item()) if has_curv else 0.0
            row_log["effective_curvature"] = float(F.softplus(model.fine_head.curvature_alpha).item()) if has_curv else 0.0

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

            print(
                format_eval_block(
                    epoch=epoch,
                    epochs=epochs,
                    row_log=row_log,
                    val_metrics=val_metrics,
                    cur_mae=cur_mae,
                    best_mae=ckpt_manager.best_mae,
                    status_tag=status_tag,
                    solver_strength=solver_strength,
                    diag_summary=diag_summary,
                    loss_avgs=loss_avgs,
                    loss_cfg=loss_cfg,
                ),
                flush=True,
            )
        else:
            print(
                format_epoch_row(
                    epoch=epoch,
                    epochs=epochs,
                    row_log=row_log,
                    loss_avgs=loss_avgs,
                    loss_cfg=loss_cfg,
                    solver_strength=solver_strength,
                ),
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
