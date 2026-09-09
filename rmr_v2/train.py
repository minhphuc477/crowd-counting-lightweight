from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import sys
from pathlib import Path

# Windows cp1252 stdout encoding safety
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

import subprocess
from rmr_core.data import CrowdManifestDataset, collate_eval, collate_train, compute_manifest_density
from rmr_core.metrics import game_physical_image, game_single, summarize_predictions
from rmr_core.training import load_rng_state, make_scheduler, save_rng_state, seed_everything
from .config import compute_v2_config_hash, validate_v2_config, validate_v2_resume_compatibility
from .losses import LossConfig, compute_losses
from .model import RMRConfig, RMRCount, count_parameters


def get_git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        dirty = bool(status)
    except Exception:
        dirty = False
    return commit, dirty


def make_model(cfg: dict) -> RMRCount:
    model_cfg = cfg["model"]
    update_rule = model_cfg.get("update_rule")
    if update_rule is None:
        update_rule = "jacobian" if model_cfg.get("use_jacobian_gate", False) else "latent"

    mcfg = RMRConfig(
        output_stride=model_cfg.get("output_stride", 4),
        feature_width=model_cfg.get("feature_width", 32),
        region_sizes_px=tuple(model_cfg.get("region_sizes_px", [32, 64, 128])),
        region_overlap=model_cfg.get("region_overlap", 0.5),
        include_full_image=model_cfg.get("include_full_image", False),
        iterations=model_cfg.get("iterations", 2),
        eta_max=model_cfg.get("eta_max", 0.20),
        eta_init=model_cfg.get("eta_init", 0.05),
        residual_clip=model_cfg.get("residual_clip", 5.0),
        update_rule=update_rule,
        use_jacobian_gate=model_cfg.get("use_jacobian_gate", False),
        sirt_omega=model_cfg.get("sirt_omega", 1.0),
        learnable_sirt_omega=model_cfg.get("learnable_sirt_omega", False),
        projected_use_preconditioner=model_cfg.get("projected_use_preconditioner", False),
        detach_region_evidence=model_cfg.get("detach_region_evidence", True),
        backbone_name=model_cfg.get(
            "backbone_name",
            model_cfg.get("backbone", "mobilenetv4_conv_small_050.e3000_r224_in1k"),
        ),
        pretrained=model_cfg.get("pretrained", True),
        init_m0=float(model_cfg.get("init_m0", 0.015763)),
        backbone_lr_scale=float(
            model_cfg.get(
                "backbone_lr_scale",
                cfg["train"].get("backbone_lr_scale", 0.1),
            )
        ),
    )

    return RMRCount(mcfg, variant=model_cfg["variant"])


def make_loss_cfg(cfg: dict) -> LossConfig:
    x = cfg.get("loss", {})
    return LossConfig(
        lambda_count=x.get("lambda_count", x.get("lambda_global", 1.0)),
        lambda_flat_dm16=x.get("lambda_flat_dm16", 1.0),
        lambda_cell=x.get("lambda_cell", 0.25),
        lambda_region_map=x.get("lambda_region_map", 0.20),
        lambda_region_head=x.get("lambda_region_head", 0.20),
        lambda_deep_supervision=x.get("lambda_deep_supervision", 0.0),
        count_loss_mode=x.get("count_loss_mode", "nb"),
        nb_dispersion=x.get("nb_dispersion", 50.0),
        kappa_flat16=x.get("kappa_flat16", 20.0),
        cell_beta=x.get("cell_beta", 1.0),
        region_beta=x.get("region_beta", 0.1),
        normalize_flat_dm16=x.get("normalize_flat_dm16", True),
    )


@torch.no_grad()
def evaluate(model: RMRCount, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    output_stride = getattr(model.cfg, "output_stride", 4)
    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["target_y"].to(device)
            out = model(image)
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
            rows.append(row)
    return summarize_predictions(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--disable-early-stopping", action="store_true", default=False)
    ap.add_argument("--deterministic", action="store_true", default=False, help="Enable strict determinism")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument(
        "--allow-cross-commit-resume",
        action="store_true",
        default=False,
        help="Allow resuming checkpoint created from different git commit",
    )
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

    validate_v2_config(cfg)

    seed = int(cfg.get("seed", 42))
    deterministic = bool(args.deterministic or cfg.get("train", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    if "init_m0" not in cfg["model"] and "train_manifest" in cfg.get("data", {}):
        stride = int(cfg["model"].get("output_stride", 4))
        cfg["model"]["init_m0"] = compute_manifest_density(
            cfg["data"]["train_manifest"],
            output_stride=stride,
        )

    run_config_hash = compute_v2_config_hash(cfg)

    out_dir = Path(cfg["output_dir"])
    if out_dir.exists() and not args.resume:
        existing_artifacts = list(out_dir.glob("*.pt")) + list(out_dir.glob("*.csv"))
        if existing_artifacts:
            if not args.overwrite:
                raise RuntimeError(
                    f"Output directory '{out_dir}' already exists and contains run artifacts: "
                    f"{[f.name for f in existing_artifacts]}. "
                    f"Use --overwrite to start fresh or --resume to continue."
                )
            for f in existing_artifacts:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    d_cfg = cfg.get("data", {})
    train_ds = CrowdManifestDataset(
        d_cfg["train_manifest"],
        train=True,
        output_stride=int(cfg.get("model", {}).get("output_stride", 4)),
        crop_size=int(d_cfg.get("crop_size", 512)),
        scale_range=tuple(d_cfg.get("scale_range", [0.75, 1.25])),
        hflip_prob=float(d_cfg.get("hflip_prob", 0.5)),
        brightness_jitter=float(d_cfg.get("brightness_jitter", 0.0)),
        contrast_jitter=float(d_cfg.get("contrast_jitter", 0.0)),
        data_root=d_cfg.get("data_root"),
    )
    val_manifest = d_cfg.get("val_manifest")
    val_ds = None if not val_manifest else CrowdManifestDataset(
        val_manifest,
        train=False,
        output_stride=int(cfg.get("model", {}).get("output_stride", 4)),
        data_root=d_cfg.get("data_root"),
    )
    workers = int(cfg["train"].get("workers", 0))
    pin_mem = bool(cfg["train"].get("pin_memory", False))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"].get("batch_size", 8),
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
    model = make_model(cfg).to(device)
    epochs = int(cfg["train"].get("epochs", 1000))
    lr_init = float(cfg["train"].get("lr", 3e-4))
    bs = int(cfg["train"].get("batch_size", 8))
    print(
        f"\n{'=' * 80}\n"
        f"  RMR-Count Training Initialized\n"
        f"  Variant: {model.variant.upper()} | Parameters: {count_parameters(model):,} | Device: {device}\n"
        f"  Epochs: {epochs} | Batch Size: {bs} | Initial LR: {lr_init:.2e}\n"
        f"  Output Directory: {out_dir}\n"
        f"{'=' * 80}\n",
        flush=True,
    )

    backbone_scale = float(
        cfg["train"].get(
            "backbone_lr_scale",
            getattr(model.cfg, "backbone_lr_scale", 1.0),
        )
    )
    if backbone_scale < 1.0 and hasattr(model, "encoder") and model.cfg.backbone_name != "tiny":
        backbone_params = list(model.encoder.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        param_groups = [
            {"params": backbone_params, "lr": lr_init * backbone_scale},
            {"params": other_params, "lr": lr_init},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_init,
            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        )
    scheduler = make_scheduler(optimizer, epochs, int(cfg["train"].get("warmup_epochs", 5)))
    amp = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_cfg = make_loss_cfg(cfg)
    grad_clip = float(cfg["train"].get("grad_clip", 500.0))
    eval_every = int(cfg["train"].get("eval_every", 10))
    solver_warmup_epochs = int(cfg["train"].get("solver_warmup_epochs", 5))
    solver_ramp_epochs = int(cfg["train"].get("solver_ramp_epochs", 20))
    early_stopping = bool(cfg["train"].get("early_stopping", True))
    patience = int(cfg["train"].get("patience", 10 if early_stopping else 0))
    if patience <= 0:
        early_stopping = False
    no_improve_evals = 0
    diverge_evals = 0

    start_epoch = 0
    best_mae = float("inf")
    if args.resume:
        try:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location="cpu")
        current_commit, _ = get_git_info()
        ckpt_commit = str(ckpt.get("git_commit", ckpt.get("provenance", {}).get("git_commit", "unknown")))
        validate_v2_resume_compatibility(
            ckpt.get("config", {}),
            cfg,
            ckpt_hash=ckpt.get("config_hash"),
            incoming_hash=run_config_hash,
            ckpt_commit=ckpt_commit,
            current_commit=current_commit,
            allow_cross_commit=bool(args.allow_cross_commit_resume),
        )
        state_dict = dict(ckpt["model"])
        if getattr(model, "eta_logits", None) is None and "eta_logits" in state_dict:
            state_dict.pop("eta_logits", None)
        if getattr(model, "log_sirt_omega", None) is None and "log_sirt_omega" in state_dict:
            state_dict.pop("log_sirt_omega", None)
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "rng_state" in ckpt:
            load_rng_state(ckpt["rng_state"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", best_mae)

    log_path = out_dir / "train_log.csv"
    fieldnames = [
        "epoch", "lr", "lr_backbone", "lr_main", "solver_strength", "rmr_update_rule", "solver_step0", "eta0",
        "train_total", "train_count", "train_flat_dm16", "train_cell", "train_global", "train_region_head", "train_region_map", "train_deep",
        "grad_norm_mean", "grad_norm_max", "clip_rate",
        "residual_abs_mean", "residual_abs_max", "z_lt_minus10_frac",
        "initial_pred_count_mean", "initial_pred_count_std",
        "solver_rel_step_mean", "solver_delta_n_mean",
        "solver_signed_delta_n_mean", "solver_r_spatial_mean",
        "val_MAE", "val_RMSE", "val_NAE", "val_Bias",
    ]
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in range(start_epoch, epochs):
        model.train()

        if model.variant in {"local_refine", "learned_project", "rmr"}:
            if epoch < solver_warmup_epochs:
                solver_strength = 0.0
            else:
                solver_strength = min(
                    1.0,
                    (epoch - solver_warmup_epochs + 1) / max(1, solver_ramp_epochs),
                )
            model.set_solver_strength(solver_strength)
        else:
            solver_strength = 1.0

        sums: dict[str, float] = {
            "total": 0.0, "cell": 0.0, "count": 0.0, "global": 0.0,
            "flat_dm16": 0.0, "region_head": 0.0, "region_map": 0.0, "deep": 0.0,
        }
        n_steps = 0
        clipped = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        residual_abs_sum = 0.0
        residual_abs_max = 0.0
        z_sat_sum = 0.0
        init_count_sum = 0.0
        init_count_sq_sum = 0.0
        init_count_n = 0
        solver_rel_step_sum = 0.0
        solver_delta_n_sum = 0.0
        solver_signed_delta_n_sum = 0.0
        solver_r_spatial_sum = 0.0

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target_y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(image)
                losses = compute_losses(outputs, target, model.variant, loss_cfg)

            y0_counts = outputs["y0"].detach().sum(dim=(-3, -2, -1))
            init_count_sum += float(y0_counts.sum().item())
            init_count_sq_sum += float((y0_counts ** 2).sum().item())
            init_count_n += y0_counts.numel()

            iterates = outputs.get("iterates", [])
            if len(iterates) >= 2:
                y0_det = iterates[0].detach()
                yf_det = iterates[-1].detach()
                rel_step = float(
                    ((yf_det - y0_det).abs().sum() / (y0_det.abs().sum() + 1e-8)).item()
                )
                dy = yf_det - y0_det
                signed_dn = dy.sum(dim=(-3, -2, -1))
                abs_dn = signed_dn.abs()
                l1_dy = dy.abs().sum(dim=(-3, -2, -1))
                r_sp = torch.where(
                    l1_dy > 1e-6,
                    (1.0 - (abs_dn / (l1_dy + 1e-6))).clamp(0.0, 1.0),
                    torch.zeros_like(l1_dy),
                )
                solver_rel_step_sum += rel_step
                solver_delta_n_sum += float(abs_dn.mean().item())
                solver_signed_delta_n_sum += float(signed_dn.mean().item())
                solver_r_spatial_sum += float(r_sp.mean().item())

            residuals = outputs.get("residual_fields", [])
            if residuals:
                r_last = residuals[-1].detach()
                residual_abs_sum += float(r_last.abs().mean().item())
                residual_abs_max = max(residual_abs_max, float(r_last.abs().max().item()))
            z_last = outputs.get("z")
            if z_last is not None:
                z_sat_sum += float((z_last.detach() < -10.0).float().mean().item())

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
            clipped += int(grad_norm > grad_clip)
            grad_norm_sum += grad_norm
            grad_norm_max = max(grad_norm_max, grad_norm)
            scaler.step(optimizer)
            scaler.update()

            for k in sums:
                if k in losses:
                    sums[k] += float(losses[k].detach().item())
            n_steps += 1

        if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 1:
            lr_backbone = float(optimizer.param_groups[0]["lr"])
            lr_main = float(optimizer.param_groups[-1]["lr"])
        else:
            lr_backbone = float(optimizer.param_groups[0]["lr"])
            lr_main = lr_backbone
        current_lr = lr_main
        scheduler.step()

        if init_count_n > 0:
            init_mean = init_count_sum / init_count_n
            init_var = init_count_sq_sum / init_count_n - init_mean ** 2
            init_std = float(init_var ** 0.5) if init_var > 0 else 0.0
        else:
            init_mean = init_std = 0.0

        if (
            model.variant == "rmr"
            and getattr(model, "rmr_update_rule", None) == "projected_sirt"
        ):
            solver_step0 = float(
                model._sirt_omega(device=device).detach().cpu().item()
                * model.solver_strength
            )
        elif hasattr(model, "_eta") and getattr(model, "eta_logits", None) is not None:
            solver_step0 = float(model._eta(0).detach().cpu().item())
        else:
            solver_step0 = 0.0
        rule_name = getattr(model, "rmr_update_rule", "latent")

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "lr_backbone": lr_backbone,
            "lr_main": lr_main,
            "solver_strength": solver_strength,
            "rmr_update_rule": rule_name,
            "solver_step0": solver_step0,
            "eta0": solver_step0,
            "train_total": sums["total"] / max(1, n_steps),
            "train_count": sums["count"] / max(1, n_steps),
            "train_flat_dm16": sums["flat_dm16"] / max(1, n_steps),
            "train_cell": sums["cell"] / max(1, n_steps),
            "train_global": sums["global"] / max(1, n_steps),
            "train_region_head": sums["region_head"] / max(1, n_steps),
            "train_region_map": sums["region_map"] / max(1, n_steps),
            "train_deep": sums["deep"] / max(1, n_steps),
            "grad_norm_mean": grad_norm_sum / max(1, n_steps),
            "grad_norm_max": grad_norm_max,
            "clip_rate": clipped / max(1, n_steps),
            "residual_abs_mean": residual_abs_sum / max(1, n_steps),
            "residual_abs_max": residual_abs_max,
            "z_lt_minus10_frac": z_sat_sum / max(1, n_steps),
            "initial_pred_count_mean": init_mean,
            "initial_pred_count_std": init_std,
            "solver_rel_step_mean": solver_rel_step_sum / max(1, n_steps),
            "solver_delta_n_mean": solver_delta_n_sum / max(1, n_steps),
            "solver_signed_delta_n_mean": solver_signed_delta_n_sum / max(1, n_steps),
            "solver_r_spatial_mean": solver_r_spatial_sum / max(1, n_steps),
            "val_MAE": "",
            "val_RMSE": "",
            "val_NAE": "",
            "val_Bias": "",
        }

        do_eval = val_loader is not None and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        git_commit, git_dirty = get_git_info()
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng_state": save_rng_state(),
            "solver_strength": solver_strength,
            "best_mae": best_mae,
            "config": cfg,
            "config_hash": run_config_hash,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        }
        if do_eval:
            metrics = evaluate(model, val_loader, device)
            row.update({
                "val_MAE": metrics["MAE"],
                "val_RMSE": metrics["RMSE"],
                "val_NAE": metrics["NAE"],
                "val_Bias": metrics["Bias"],
            })
            solver_fully_ramped = (solver_strength >= 1.0) or (
                model.variant not in {"local_refine", "learned_project", "rmr"}
            )
            is_new_best = False
            if metrics["MAE"] < best_mae and solver_fully_ramped:
                best_mae = metrics["MAE"]
                state["best_mae"] = best_mae
                torch.save(state, out_dir / "best_val_mae.pt")
                is_new_best = True

            status_tag = ""
            if is_new_best:
                status_tag = " >>> [NEW BEST CHECKPOINT SAVED] <<<"
            elif not solver_fully_ramped:
                status_tag = f" (Solver ramping: epoch {epoch+1}/25)"

            print(
                f"\n{'='*92}\n"
                f"  EPOCH [{epoch+1:04d}/{epochs:04d}] PERIODIC EVALUATION (182 test samples)\n"
                f"{'-'*92}\n"
                f"  Train Loss    : {row['train_total']:.4f} [cnt: {row['train_count']:.4f}, flat: {row['train_flat_dm16']:.4f}, cell: {row['train_cell']:.4f}, reg: {row['train_region_head']:.4f}]\n"
                f"  Solver Status : Strength: {solver_strength:.2f} | Step0 (omega): {solver_step0:.2f}\n"
                f"  Val Metrics   : MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | NAE: {metrics['NAE']:.3f} | Bias: {metrics['Bias']:+.2f}\n"
                f"  Checkpoint    : Current Val MAE: {metrics['MAE']:.2f} | Best Val MAE: {best_mae:.2f}{status_tag}\n"
                f"{'='*92}\n",
                flush=True,
            )
        else:
            print(
                f"[{epoch+1:04d}/{epochs:04d}] "
                f"Loss: {row['train_total']:.4f} [cnt: {row['train_count']:.4f}, flat: {row['train_flat_dm16']:.4f}, cell: {row['train_cell']:.4f}, reg: {row['train_region_head']:.4f}] | "
                f"SolvStr: {solver_strength:.2f}",
                flush=True,
            )

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            state["best_mae"] = best_mae
            torch.save(state, out_dir / "last.pt")

        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)

        gc.collect()


if __name__ == "__main__":
    main()
