"""Training script for Neural Tree-Pólya Crowd Counting (NTPC).

Supports the 5 decisive ablation modes:
  - R0: Multi-Scale Exact Regional Regression
  - R1: S-DCNet Deterministic Allocation
  - R2: Flat Dirichlet-Multinomial (No Hierarchy)
  - R3: Neural DTM Tree (Core Proposed Method)
  - R4: Full NTPC (R3 + Dense-Adaptive Fine 16->8)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from hpc.data.common import custom_collate_fn
from hpc.data.sampler import build_density_luminance_sampler
from hpc.data.sha import ShanghaiTechDataset
from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics
from hpc.models.hpc_lite import HPCLite
from hpc.utils.seed import seed_everything


def evaluate_model(model: nn.Module, val_dataset, device: torch.device) -> dict:
    """Evaluate model with single-scale padded inference."""
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            img = sample["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
            gt_cnt = float(sample["gt_count"])

            pred_cnt, _ = model.predict(img, pad_multiple=32)
            predictions.append(float(pred_cnt.item()))
            ground_truths.append(gt_cnt)

    counting_metrics = evaluate_counting_metrics(predictions, ground_truths)
    subgroup_metrics = evaluate_subgroup_diagnostics(predictions, ground_truths)

    results = {}
    results.update(counting_metrics)
    results.update(subgroup_metrics)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Neural Tree-Pólya Crowd Counting (NTPC) Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    seed_everything(exp_cfg.get("seed", 42))
    save_dir = exp_cfg.get("save_dir", "./runs/ntpc_experiment")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    ds_cfg = cfg["dataset"]
    train_ds = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="train_data",
        crop_size=ds_cfg["crop_size"],
        hnb_blocks=ds_cfg.get("hnb_blocks", [8, 16, 32, 64]),
        allocation_block=ds_cfg.get("allocation_block", 16),
        is_train=True,
        scale_range=tuple(ds_cfg.get("scale_range", [0.75, 2.0])),
        flip_prob=float(ds_cfg.get("flip_prob", 0.5)),
        image_mean=ds_cfg.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=ds_cfg.get("image_std", [0.5, 0.5, 0.5]),
    )

    val_ds = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="test_data",
        crop_size=ds_cfg["crop_size"],
        hnb_blocks=ds_cfg.get("hnb_blocks", [8, 16, 32, 64]),
        allocation_block=ds_cfg.get("allocation_block", 16),
        is_train=False,
        image_mean=ds_cfg.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=ds_cfg.get("image_std", [0.5, 0.5, 0.5]),
    )

    sampler = None
    if cfg.get("sampler", {}).get("weighted", False):
        s_cfg = cfg["sampler"]
        sampler, _ = build_density_luminance_sampler(
            train_ds.image_paths,
            train_ds.points_list,
            num_density_bins=s_cfg.get("density_bins", 5),
            num_luminance_bins=s_cfg.get("luminance_bins", 4),
            power=float(s_cfg.get("power", 0.5)),
        )

    batch_size = cfg["training"].get("batch_size", 16)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Model Construction (HPC-Lite S2: MobileNetV4 + Additive FPN)
    m_cfg = cfg["model"]
    model = HPCLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050"),
        pretrained=m_cfg.get("pretrained", False),
        neck_width=m_cfg.get("neck_width", 32),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_p8_context=bool(m_cfg.get("use_p8_context", True)),
        use_repblock=bool(m_cfg.get("use_repblock", False)),
        eps_d=float(m_cfg.get("eps_d", 1e-6)),
        truncate_backbone=bool(m_cfg.get("truncate_backbone", True)),
    ).to(device)

    # Optional initial checkpoint
    init_ckpt = m_cfg.get("init_checkpoint")
    if init_ckpt and os.path.exists(init_ckpt):
        print(f"Loading initial student weights from {init_ckpt}...")
        state = torch.load(init_ckpt, map_location=device)
        model_state = state.get("model_state_dict", state)
        model.load_state_dict(model_state, strict=False)

    deploy_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NTPC Deployed Parameters: {deploy_params:,} ({deploy_params / 1e6:.3f}M)")

    # 3. Loss Configuration
    loss_cfg = cfg.get("loss", {})
    ntpc_cfg = NTPCConfig(
        mode=loss_cfg.get("mode", "r4_full_ntpc"),
        root_dispersion=float(cfg.get("statistics", {}).get("root_dispersion", 50.0)),
        kappa_root64=float(loss_cfg.get("kappa_root64", 20.0)),
        kappa_64_32=float(loss_cfg.get("kappa_64_32", 20.0)),
        kappa_32_16=float(loss_cfg.get("kappa_32_16", 20.0)),
        kappa_16_8=float(loss_cfg.get("kappa_16_8", 20.0)),
        kappa_flat16=float(loss_cfg.get("kappa_flat16", 20.0)),
        dense_threshold_16=float(loss_cfg.get("dense_threshold_16", 2.0)),
        w_root_nb=float(loss_cfg.get("w_root_nb", 1.0)),
        w_root64=float(loss_cfg.get("w_root64", 1.0)),
        w_64_32=float(loss_cfg.get("w_64_32", 1.0)),
        w_32_16=float(loss_cfg.get("w_32_16", 1.0)),
        w_16_8=float(loss_cfg.get("w_16_8", 1.0)),
        w_flat_16=float(loss_cfg.get("w_flat_16", 1.0)),
        w_exact_regression=float(loss_cfg.get("w_exact_regression", 1.0)),
        w_deterministic_alloc=float(loss_cfg.get("w_deterministic_alloc", 1.0)),
    )
    criterion = NTPCLoss(ntpc_cfg).to(device)

    # 4. Optimizer & LR Scheduler
    opt_cfg = cfg["optimizer"]
    base_lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    total_epochs = cfg["schedule"]["epochs"]
    warmup_epochs = cfg["schedule"].get("warmup_epochs", 25)
    validate_every = cfg["training"].get("validate_every", 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 5. Mixed Precision & Gradient Clipping
    use_amp = cfg["training"].get("amp", True)
    init_scale = float(cfg["training"].get("init_scale", 256.0))
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale, enabled=use_amp)
    grad_clip = float(opt_cfg.get("grad_clip", 5.0))

    # 6. Training Loop & Validation Tracking
    val_csv_path = os.path.join(save_dir, "val.csv")
    with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mae", "rmse", "nae", "sparse_mae", "med_mae", "dense_mae", "lr"])

    best_mae = float("inf")
    best_epoch = 0

    print(f"Starting NTPC Training (Mode: {ntpc_cfg.mode}) for {total_epochs} epochs...")

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_root = 0.0
        running_r64 = 0.0
        running_64_32 = 0.0
        running_32_16 = 0.0
        running_16_8 = 0.0
        running_flat = 0.0
        running_det = 0.0
        running_exact = 0.0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            target_pyramid = {int(k): v.to(device, non_blocking=True) for k, v in batch["gt_blocks"].items()}
            target_pyramid["N"] = batch["gt_count"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                density_map = model(images)  # (B, 1, H/4, W/4)
                loss, logs = criterion(
                    mass=density_map,
                    target_pyramid=target_pyramid,
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_root += logs["root_nb"].item()
            running_r64 += logs["root_to_64"].item()
            running_64_32 += logs["64_to_32"].item()
            running_32_16 += logs["32_to_16"].item()
            running_16_8 += logs["16_to_8_dense"].item()
            running_flat += logs["flat_16"].item()
            running_det += logs["deterministic_alloc"].item()
            running_exact += logs["exact_regression"].item()

        scheduler.step()
        n_steps = len(train_loader)
        epoch_time = time.time() - t0
        avg_loss = running_loss / n_steps

        if epoch % validate_every == 0 or epoch == total_epochs:
            val_res = evaluate_model(model, val_ds, device)
            mae = val_res["mae"]
            rmse = val_res["rmse"]
            nae = val_res.get("nae", 0.0)
            sparse_mae = val_res.get("bin_11_100_mae", 0.0)
            med_mae = val_res.get("bin_101_1000_mae", 0.0)
            dense_mae = val_res.get("bin_gt1000_mae", 0.0)

            with open(val_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, mae, rmse, nae, sparse_mae, med_mae, dense_mae,
                    optimizer.param_groups[0]["lr"],
                ])

            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_res": val_res,
                    "config": cfg,
                }, os.path.join(save_dir, "best.pt"))

            best_tag = " (Best)" if is_best else ""
            
            # Mode-specific log printing
            if ntpc_cfg.mode == "r0_exact":
                pieces = f"ExactL1: {running_exact/n_steps:.2f}"
            elif ntpc_cfg.mode == "r1_deterministic":
                pieces = f"Root: {running_root/n_steps:.2f}, DetAlloc: {running_det/n_steps:.2f}"
            elif ntpc_cfg.mode == "r2_flat_dm":
                pieces = f"Root: {running_root/n_steps:.2f}, Flat16: {running_flat/n_steps:.2f}"
            elif ntpc_cfg.mode == "r3_tree_dtm":
                pieces = f"Root: {running_root/n_steps:.2f}, R64: {running_r64/n_steps:.2f}, 64->32: {running_64_32/n_steps:.2f}, 32->16: {running_32_16/n_steps:.2f}"
            else:
                pieces = f"Root: {running_root/n_steps:.2f}, R64: {running_r64/n_steps:.2f}, 64->32: {running_64_32/n_steps:.2f}, 32->16: {running_32_16/n_steps:.2f}, 16->8: {running_16_8/n_steps:.2f}"

            print(
                f"Epoch [{epoch:03d}/{total_epochs}] Loss: {avg_loss:.2f} ({pieces}) | "
                f"Val MAE: {mae:.2f}, RMSE: {rmse:.2f}{best_tag} | "
                f"Med: {med_mae:.2f}, Dense: {dense_mae:.2f}, Sparse: {sparse_mae:.2f} | Time: {epoch_time:.1f}s"
            )
        else:
            print(
                f"Epoch [{epoch:03d}/{total_epochs}] Loss: {avg_loss:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s"
            )

    print(f"\nNTPC Training Completed. Best Val MAE: {best_mae:.2f} (Epoch {best_epoch})")


if __name__ == "__main__":
    main()
