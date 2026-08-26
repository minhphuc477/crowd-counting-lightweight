"""Continuation training from best.pt using Point-Neighbor Mass Decomposition Loss.

Loads weights from runs/sha/best.pt, restarts LR at 5e-5 with a 150-epoch cosine schedule,
and optimizes the new objective:
    L = 1.0 * L_count + 1.0 * L_point + 0.25 * L_hnb + 0.10 * L_hn + 0.10 * L_route
"""
import json
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from hpc.data.common import BaseCrowdDataset
from hpc.data.sha import ShanghaiTechDataset
from hpc.losses.criterion import HPCLossCriterion
from hpc.models.hpc_lite import HPCLiteSR48
from hpc.utils.seed import seed_everything
from hpc.utils.logging import CSVLogger
from hpc.utils.checkpoint import save_checkpoint
from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics
from train import custom_collate_fn, build_optimizer_and_scheduler, build_dataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_detailed(model: nn.Module, val_dataset, device: torch.device) -> dict:
    """Run validation and compute comprehensive metrics including signed bias."""
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            img = sample["image"].unsqueeze(0).to(device)
            gt_cnt = float(sample["gt_count"])

            pred_cnt, _ = model.predict(img, pad_multiple=32)
            predictions.append(float(pred_cnt.item()))
            ground_truths.append(gt_cnt)

    preds = np.array(predictions)
    gts = np.array(ground_truths)

    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    bias = float(np.mean(preds - gts))

    counting_metrics = evaluate_counting_metrics(predictions, ground_truths)
    subgroup_metrics = evaluate_subgroup_diagnostics(predictions, ground_truths)

    results = {}
    results.update(counting_metrics)
    results.update(subgroup_metrics)
    results["bias"] = bias
    return results


def train_continuation(
    config_path: str = "configs/sha.yaml",
    init_ckpt_path: str = "runs/sha/best.pt",
    save_dir: str = "runs/sha_point_mass",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("experiment", {}).get("seed", 42))
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    train_dataset = build_dataset(cfg, is_train=True)
    val_dataset = build_dataset(cfg, is_train=False)

    t_cfg = cfg.get("training", {})
    batch_size = int(t_cfg.get("batch_size", 16))
    accum_steps = int(t_cfg.get("accum_steps", 1))
    use_amp = bool(t_cfg.get("amp", True)) and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=int(t_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # 2. Model Initialization
    m_cfg = cfg["model"]
    model = HPCLiteSR48(
        pretrained=False,
        neck_width=m_cfg.get("neck_width", 48),
        eps_d=float(m_cfg.get("eps_d", 1e-6)),
        route_temperature=float(m_cfg.get("route_temperature", 1.0)),
        pool_kernels=tuple(m_cfg.get("pool_kernels", [3, 5, 7])),
        pool_residual_mix=float(m_cfg.get("pool_residual_mix", 0.5)),
        simam_lambda=float(m_cfg.get("simam_lambda", 1e-4)),
    ).to(device)

    # Load weights from previous best checkpoint
    if os.path.exists(init_ckpt_path):
        print(f"\n{'='*60}")
        print(f"  LOADING PRETRAINED WEIGHTS FROM: {init_ckpt_path}")
        print(f"{'='*60}\n")
        ckpt = torch.load(init_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        init_best_mae = float(ckpt.get("best_mae", 76.52))
        print(f"  Successfully loaded weights! Reference baseline MAE: {init_best_mae:.2f}")
    else:
        print(f"Warning: Checkpoint {init_ckpt_path} not found. Starting from scratch.")
        init_best_mae = float("inf")

    # Initial validation sanity check
    init_eval = validate_detailed(model, val_dataset, device)
    print(f"Initial Zero-Epoch Baseline Validation: MAE = {init_eval['mae']:.2f}, RMSE = {init_eval['rmse']:.2f}, Bias = {init_eval['bias']:+.2f}")

    # 3. Loss Criterion
    l_cfg = cfg["loss"]
    criterion = HPCLossCriterion(
        block_sizes=cfg["dataset"]["hnb_blocks"],
        allocation_block=cfg["dataset"].get("allocation_block", 16),
        lambda_count=float(l_cfg.get("lambda_count", 1.0)),
        count_scale=float(l_cfg.get("count_scale", 100.0)),
        lambda_point=float(l_cfg.get("lambda_point", 1.0)),
        lambda_hnb=float(l_cfg.get("lambda_hnb", 0.25)),
        lambda_alloc=float(l_cfg.get("lambda_alloc", 0.0)),
        lambda_hn=float(l_cfg.get("lambda_hn", 0.10)),
        lambda_empty=float(l_cfg.get("lambda_empty", 0.25)),
        lambda_global=float(l_cfg.get("lambda_global", 0.10)),
        lambda_rob=float(l_cfg.get("lambda_rob", 0.05)),
        lambda_route=float(l_cfg.get("lambda_route", 0.10)),
        hard_negative_fraction=float(l_cfg.get("hard_negative_fraction", 0.10)),
        use_stratified_nb=l_cfg.get("density_stratified_nb", True),
        global_count_mode=l_cfg.get("global_count_mode", "log_smooth_l1"),
        learn_dispersion=bool(l_cfg.get("learn_dispersion", False)),
        enable_curriculum=l_cfg.get("enable_curriculum", True),
    ).to(device)

    # 4. Optimizer and LR Scheduler (150-epoch restart)
    total_epochs = int(cfg.get("schedule", {}).get("epochs", 150))
    warmup_epochs = int(cfg.get("schedule", {}).get("warmup_epochs", 10))

    opt_cfg = dict(cfg.get("optimizer", {}))
    opt_cfg.setdefault("lr", 5.0e-5)
    opt_cfg.setdefault("lr_start", 1.0e-5)
    opt_cfg.setdefault("lr_min", 1.0e-6)
    opt_cfg.setdefault("weight_decay", 1.0e-4)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, opt_cfg, total_epochs, warmup_epochs
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_clip = float(opt_cfg.get("grad_clip", 1.0))

    train_logger = CSVLogger(os.path.join(save_dir, "train.csv"))
    val_logger = CSVLogger(os.path.join(save_dir, "val.csv"))

    best_mae = init_best_mae

    print(f"\nStarting Point-Mass Continuation Training: 1 -> {total_epochs} epochs (LR={opt_cfg['lr']})...\n")

    for epoch in range(1, total_epochs + 1):
        model.train()
        criterion.train()
        epoch_losses = []
        epoch_count_maes = []
        epoch_point_losses = []

        progress = float(epoch - 1) / float(max(total_epochs - 1, 1))
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            gt_blocks = {b: batch["gt_blocks"][b].to(device) for b in batch["gt_blocks"]}
            gt_counts = batch["gt_count"].to(device)
            gt_points = batch.get("gt_points", None)

            gt_special_mask16 = batch.get("gt_special_mask16", None)
            if gt_special_mask16 is not None:
                gt_special_mask16 = gt_special_mask16.to(device)

            gt_route_q = batch.get("gt_route_q", None)
            gt_route_mask = batch.get("gt_route_mask", None)
            if gt_route_q is not None:
                gt_route_q = gt_route_q.to(device)
            if gt_route_mask is not None:
                gt_route_mask = gt_route_mask.to(device)

            img_deg = batch.get("image_degraded", None)
            degraded_mask = batch.get("has_degraded", None)
            if img_deg is not None:
                img_deg = img_deg.to(device)
            if degraded_mask is not None:
                degraded_mask = degraded_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    d_clean, aux = model(images, return_aux=True)
                    routes8 = aux["routes8"]
                    d_deg = model(img_deg) if img_deg is not None else None
                    loss, loss_dict = criterion(
                        d_clean, gt_blocks, gt_counts,
                        gt_points=gt_points,
                        gt_special_mask16=gt_special_mask16,
                        d_degraded=d_deg, degraded_mask=degraded_mask,
                        routes8=routes8,
                        gt_route_q=gt_route_q, gt_route_mask=gt_route_mask,
                        progress=progress,
                    )
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
            else:
                d_clean, aux = model(images, return_aux=True)
                routes8 = aux["routes8"]
                d_deg = model(img_deg) if img_deg is not None else None
                loss, loss_dict = criterion(
                    d_clean, gt_blocks, gt_counts,
                    gt_points=gt_points,
                    gt_special_mask16=gt_special_mask16,
                    d_degraded=d_deg, degraded_mask=degraded_mask,
                    routes8=routes8,
                    gt_route_q=gt_route_q, gt_route_mask=gt_route_mask,
                    progress=progress,
                )
                loss = loss / accum_steps
                loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            step_loss = loss.item() * accum_steps
            epoch_losses.append(step_loss)
            epoch_count_maes.append(float(loss_dict.get("batch_count_mae", 0.0)))
            epoch_point_losses.append(float(loss_dict.get("loss_point", 0.0)))

            if (step + 1) % max(len(train_loader) // 4, 1) == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch:03d}/{total_epochs:03d}] Step [{step+1:02d}/{len(train_loader):02d}] "
                    f"Loss: {step_loss:.4f} "
                    f"(Count: {loss_dict.get('loss_count', 0):.3f}, "
                    f"Point: {loss_dict.get('loss_point', 0):.3f}, "
                    f"BatchMAE: {loss_dict.get('batch_count_mae', 0):.1f}, "
                    f"Bias: {loss_dict.get('mean_signed_count_error', 0):+.1f}, "
                    f"HNB: {loss_dict.get('loss_hnb', 0):.3f})",
                    flush=True,
                )

        scheduler.step()
        epoch_time = time.time() - t0
        mean_loss = float(np.mean(epoch_losses))
        current_lr = optimizer.param_groups[0]["lr"]

        train_logger.log({
            "epoch": epoch,
            "loss": mean_loss,
            "lr": current_lr,
            "batch_mae": float(np.mean(epoch_count_maes)),
            "point_loss": float(np.mean(epoch_point_losses)),
            "time_s": round(epoch_time, 2),
        })

        # Validation every epoch
        val_metrics = validate_detailed(model, val_dataset, device)
        val_mae = val_metrics["mae"]
        val_rmse = val_metrics["rmse"]
        val_bias = val_metrics["bias"]
        val_metrics["epoch"] = epoch
        val_logger.log(val_metrics)

        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            save_checkpoint(
                os.path.join(save_dir, "best.pt"),
                model,
                epoch=epoch,
                best_mae=best_mae,
                config=cfg,
            )

        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] "
            f"Train Loss: {mean_loss:.4f} | "
            f"Val MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, Bias: {val_bias:+.2f} "
            f"{'(*** NEW BEST ***)' if is_best else ''} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s",
            flush=True,
        )

        save_checkpoint(
            os.path.join(save_dir, "last.pt"),
            model,
            epoch=epoch,
            best_mae=best_mae,
            config=cfg,
        )

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE! Best Validation MAE = {best_mae:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_continuation()
