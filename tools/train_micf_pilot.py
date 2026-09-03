"""Unified Trainer for the 6-Model MICF Pilot Suite (B1 to B6).

Runs controlled experiments on ShanghaiTech Part A:
B1: Local Count Baseline (L1 loss on Y)
B2: Local Output + Integral Loss (L1 loss on P Y)
B3: Direct Cumulative MICF Naive (L1 loss on C, lambda_valid=0)
B4: Direct Cumulative MICF + Validity (lambda_valid=1.0)
B5: MICF-v2 Full (Directional Context + Validity)
B6: Local Count + Directional Context (Ablation Control)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.data.common import ntpc_collate_fn
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.data.sha import ShanghaiTechDataset
from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
    IntegralLossOnLocalCount,
    MICFLoss,
)
from hpc.models.micf_lite import MICFLite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MICF-v2 Pilot Suite Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test on a small subset")
    return parser.parse_args()


def build_criterion(cfg: dict) -> nn.Module:
    l_cfg = cfg.get("loss", {})
    mode = l_cfg.get("mode", "micf")

    if mode == "local_l1":
        return nn.L1Loss()
    elif mode == "local_smooth_l1":
        return nn.SmoothL1Loss(beta=float(l_cfg.get("beta_smooth", 1.0)))
    elif mode == "integral_on_local":
        return IntegralLossOnLocalCount(
            loss_type=l_cfg.get("loss_type", "smooth_l1"),
            beta_smooth=float(l_cfg.get("beta_smooth", 1.0)),
        )
    elif mode in {"micf", "micf_naive", "micf_valid", "micf_v2_full"}:
        return MICFLoss(
            field_loss=l_cfg.get("field_loss", "smooth_l1"),
            lambda_valid=float(l_cfg.get("lambda_valid", 1.0)),
            beta_smooth=float(l_cfg.get("beta_smooth", 1.0)),
        )
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> Dict[str, float]:
    model.eval()
    errors: List[float] = []
    sq_errors: List[float] = []
    violation_rates: List[float] = []

    for idx, batch in enumerate(val_loader):
        if max_samples is not None and idx >= max_samples:
            break
        img = batch["image"].to(device)
        gt_count = float(batch["gt_count"].item())

        pred_count, pred_map = model.predict(img, pad_multiple=64)
        pred_val = float(pred_count.item())

        err = pred_val - gt_count
        errors.append(abs(err))
        sq_errors.append(err * err)

        if getattr(model, "head_type", "") == "cumulative":
            y_rec = discrete_mixed_difference(pred_map)
            viol = float((y_rec < 0).float().mean().item())
            violation_rates.append(viol)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    mean_viol = float(np.mean(violation_rates)) if violation_rates else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "violation_rate": mean_viol,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg.get("experiment", {})
    m_id = exp_cfg.get("model_id", "MICF")
    save_dir = Path(exp_cfg.get("save_dir", f"./runs/pilot_micf/{m_id.lower()}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING MICF PILOT: {m_id} - {exp_cfg.get('description', '')}")
    print(f"Device: {device} | Save Dir: {save_dir}")
    print("=" * 80, flush=True)

    # 1. Dataset & DataLoader
    ds_cfg = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})
    train_dataset = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="train_data",
        crop_size=int(ds_cfg.get("crop_size", 256)),
        is_train=True,
        scale_range=tuple(aug_cfg.get("scale_range", [0.7, 1.3])),
        flip_prob=float(aug_cfg.get("flip_prob", 0.5)),
        image_mean=ds_cfg.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=ds_cfg.get("image_std", [0.5, 0.5, 0.5]),
    )
    test_dataset = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="test_data",
        is_train=False,
        image_mean=ds_cfg.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=ds_cfg.get("image_std", [0.5, 0.5, 0.5]),
    )

    t_cfg = cfg.get("training", {})
    batch_size = int(t_cfg.get("batch_size", 16))
    num_workers = int(t_cfg.get("num_workers", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ntpc_collate_fn,
        drop_last=bool(t_cfg.get("drop_last", True)),
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=ntpc_collate_fn,
    )

    # 2. Model & Loss
    m_cfg = cfg.get("model", {})
    model = MICFLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050.e3000_r224_in1k"),
        pretrained=bool(m_cfg.get("pretrained", True)),
        neck_width=int(m_cfg.get("neck_width", 32)),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_integral_context=bool(m_cfg.get("use_integral_context", False)),
        head_type=m_cfg.get("head_type", "cumulative"),
        output_stride=int(m_cfg.get("output_stride", 16)),
        eps_d=float(m_cfg.get("eps_d", 1e-8)),
    ).to(device)

    criterion = build_criterion(cfg)

    # 3. Optimizer & Scheduler
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 0.0002))
    backbone_lr_scale = float(opt_cfg.get("backbone_lr_scale", 0.1))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0001))

    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    total_epochs = args.epochs or int(cfg.get("schedule", {}).get("epochs", 100))
    if args.smoke_test:
        total_epochs = 1

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    use_amp = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_mae = float("inf")
    history: List[Dict[str, Any]] = []

    print(f"Model initialized: {m_cfg.get('head_type')} head, context={m_cfg.get('use_integral_context')}")
    print(f"Training for {total_epochs} epochs ...", flush=True)

    # 4. Training Loop
    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            points_batch = [pts.to(device) for pts in batch["gt_points"]]
            B, _, H, W = images.shape

            # Build exact count pyramid at stride 16
            pyramid = build_exact_count_pyramid(
                points_batch,
                height=H,
                width=W,
                block_sizes=(16,),
                pad_multiple=64,
                device=device,
            )
            y_target = pyramid[16]  # [B, H/16, W/16] or [B, 1, H/16, W/16]
            if y_target.ndim == 3:
                y_target = y_target.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_field = model.forward_field(images)

                if model.head_type == "cumulative":
                    c_target = cell_counts_to_cumulative_field(y_target, orientation="TL")
                    loss = criterion(pred_field, c_target)
                elif isinstance(criterion, IntegralLossOnLocalCount):
                    loss = criterion(pred_field, y_target)
                else:
                    loss = criterion(pred_field, y_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(opt_cfg.get("grad_clip", 500.0)))
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))

            if args.smoke_test and batch_idx >= 2:
                break

        scheduler.step()
        epoch_time = time.time() - t0
        mean_loss = float(np.mean(epoch_losses))

        # Evaluate periodically
        eval_every = 1 if args.smoke_test else int(t_cfg.get("evaluate_every", 2))
        if epoch % eval_every == 0 or epoch == total_epochs:
            val_res = evaluate_model(
                model,
                val_loader,
                device,
                max_samples=5 if args.smoke_test else None,
            )
            mae = val_res["mae"]
            rmse = val_res["rmse"]
            viol = val_res["violation_rate"]

            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_mae": best_mae,
                        "config": cfg,
                    },
                    save_dir / "best.pt",
                )

            log_entry = {
                "epoch": epoch,
                "loss": mean_loss,
                "mae": mae,
                "rmse": rmse,
                "violation_rate": viol,
                "best_mae": best_mae,
                "time_sec": epoch_time,
            }
            history.append(log_entry)

            print(
                f"[Epoch {epoch:3d}/{total_epochs:3d}] Loss: {mean_loss:.4f} | "
                f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | Viol: {viol*100:.2f}% | "
                f"Best: {best_mae:.2f} {'(*)' if is_best else ''} ({epoch_time:.1f}s)",
                flush=True,
            )
        else:
            print(f"[Epoch {epoch:3d}/{total_epochs:3d}] Loss: {mean_loss:.4f} ({epoch_time:.1f}s)", flush=True)

        if args.smoke_test:
            break

    # Save training history
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining completed! Results saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
