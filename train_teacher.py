"""Training Engine for TeacherLite (Teacher-Lite v2).

Supports:
- DM-Count Optimal Transport (OT) + Total Variation (TV) loss
- Multi-Scale exact mass MAE
- Pure 4-quantile density-balanced crop sampling
- Checkpoint continuation from Teacher 68.07
- Subset diagnostic monitoring (Medium, Dense, Sparse, Overall)
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from hpc.data.sha import ShanghaiTechDataset
from hpc.data.sampler import build_density_luminance_sampler, build_density_quantile_sampler
from hpc.data.common import custom_collate_fn
from hpc.teachers.teacher_lite import TeacherLite
from hpc.losses.teacher_criterion import TeacherCriterion


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def build_optimizer(model: TeacherLite, cfg: dict):
    opt_cfg = cfg.get("optimizer", {})
    train_cfg = cfg.get("train", {})
    backbone_lr = float(opt_cfg.get("backbone_lr", train_cfg.get("backbone_lr", 1.0e-5)))
    head_lr = float(opt_cfg.get("head_lr", opt_cfg.get("task_lr", train_cfg.get("head_lr", 5.0e-5))))
    weight_decay = float(opt_cfg.get("weight_decay", 1.0e-4))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ],
        betas=betas,
    )
    return optimizer


def get_lr_factor(epoch: int, total_epochs: int, warmup_epochs: int = 50, min_ratio: float = 0.01) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine_decay


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    out["image"] = batch["image"].to(device, non_blocking=True)
    out["gt_count"] = batch["gt_count"].to(device, non_blocking=True)
    out["gt_z_alloc"] = batch["gt_z_alloc"].to(device, non_blocking=True)
    out["gt_blocks"] = {
        int(k): v.to(device, non_blocking=True)
        for k, v in batch["gt_blocks"].items()
    }
    if "gt_points" in batch:
        out["gt_points"] = [
            p.to(device, non_blocking=True) if isinstance(p, torch.Tensor) else p
            for p in batch["gt_points"]
        ]
    return out


@torch.no_grad()
def evaluate_teacher(
    model: TeacherLite,
    val_dataset: ShanghaiTechDataset,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    preds_map, preds_reg, gts = [], [], []

    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        img = sample["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
        gt = float(sample["gt_count"])

        # Variable resolution evaluation with padding to multiple of 32
        _, _, h, w = img.shape
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        out = model(img)
        out_h = math.ceil(h / 4)
        out_w = math.ceil(w / 4)

        d_valid = out["density"][..., :out_h, :out_w]
        c_map = float(d_valid.sum().item())
        c_reg = float(out["count_reg"].item())

        preds_map.append(c_map)
        preds_reg.append(c_reg)
        gts.append(gt)

    preds_map = np.array(preds_map)
    preds_reg = np.array(preds_reg)
    gts = np.array(gts)

    mae_map = float(np.mean(np.abs(preds_map - gts)))
    rmse_map = float(np.sqrt(np.mean((preds_map - gts) ** 2)))
    bias_map = float(np.mean(preds_map - gts))

    mae_reg = float(np.mean(np.abs(preds_reg - gts)))
    rmse_reg = float(np.sqrt(np.mean((preds_reg - gts) ** 2)))
    bias_reg = float(np.mean(preds_reg - gts))

    sparse = gts < 100
    med = (gts >= 100) & (gts <= 1000)
    dense = gts > 1000

    mae_sparse = float(np.mean(np.abs(preds_map[sparse] - gts[sparse]))) if np.any(sparse) else 0.0
    mae_med = float(np.mean(np.abs(preds_map[med] - gts[med]))) if np.any(med) else 0.0
    mae_dense = float(np.mean(np.abs(preds_map[dense] - gts[dense]))) if np.any(dense) else 0.0

    return {
        "mae_map": mae_map,
        "rmse_map": rmse_map,
        "bias_map": bias_map,
        "mae_reg": mae_reg,
        "rmse_reg": rmse_reg,
        "bias_reg": bias_reg,
        "mae_sparse": mae_sparse,
        "mae_med": mae_med,
        "mae_dense": mae_dense,
    }


def main():
    parser = argparse.ArgumentParser(description="Train TeacherLite v2 for HPC-Lite KD")
    parser.add_argument("--config", type=str, default="configs/sha_teacher_lite_v2.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("experiment", {}).get("seed", 42))

    save_dir = cfg.get("experiment", {}).get("save_dir", "./runs/sha_teacher_lite_v2")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    ds_cfg = cfg.get("dataset", cfg.get("data", {}))
    image_mean = ds_cfg.get("image_mean", [0.485, 0.456, 0.406])
    image_std = ds_cfg.get("image_std", [0.229, 0.224, 0.225])
    crop_size = int(ds_cfg.get("crop_size", 448))

    train_ds = ShanghaiTechDataset(
        root=ds_cfg.get("root", "./data/ShanghaiTech"),
        part=ds_cfg.get("part", "part_A"),
        split="train_data",
        crop_size=crop_size,
        hnb_blocks=ds_cfg.get("hnb_blocks", [16, 32, 64]),
        allocation_block=ds_cfg.get("allocation_block", 16),
        is_train=True,
        scale_range=tuple(cfg.get("augmentation", {}).get("scale_range", [0.75, 2.0])),
        flip_prob=float(cfg.get("augmentation", {}).get("flip_prob", 0.5)),
        image_mean=image_mean,
        image_std=image_std,
    )

    val_ds = ShanghaiTechDataset(
        root=ds_cfg.get("root", "./data/ShanghaiTech"),
        part=ds_cfg.get("part", "part_A"),
        split="test_data",
        crop_size=crop_size,
        hnb_blocks=ds_cfg.get("hnb_blocks", [16, 32, 64]),
        allocation_block=ds_cfg.get("allocation_block", 16),
        is_train=False,
        image_mean=image_mean,
        image_std=image_std,
    )

    # Sampler
    sampler_cfg = cfg.get("sampler", {})
    data_cfg = cfg.get("data", {})
    use_quantile_sampler = (
        sampler_cfg.get("density_bins") == "quantile"
        or data_cfg.get("density_balanced_sampling", False)
        or sampler_cfg.get("density_balanced_sampling", False)
    )

    if use_quantile_sampler and len(train_ds) > 0:
        print("[Sampler] Building 4-quantile density-balanced sampler (25% Q0, 25% Q1, 25% Q2, 25% Q3)...")
        sampler, stats = build_density_quantile_sampler(
            train_ds.points_list,
            num_bins=int(sampler_cfg.get("num_bins", data_cfg.get("num_bins", 4))),
        )
        shuffle = False
        print(f"[Sampler] Quantile boundaries: {stats['quartile_boundaries']}, counts per bin: {stats['bin_counts']}")
    elif sampler_cfg.get("weighted", True) and len(train_ds) > 0:
        sampler, _ = build_density_luminance_sampler(
            train_ds.image_paths,
            train_ds.points_list,
            num_density_bins=sampler_cfg.get("density_bins", 5),
            num_luminance_bins=sampler_cfg.get("luminance_bins", 4),
            power=sampler_cfg.get("power", 0.5),
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_cfg = cfg.get("training", cfg.get("train", {}))
    batch_size = int(train_cfg.get("batch_size", 1))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=custom_collate_fn,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True if len(train_ds) > batch_size else False,
    )

    # Model & Loss
    m_cfg = cfg.get("model", {}).get("teacher", cfg.get("model", {}))
    model = TeacherLite(
        width=int(m_cfg.get("fpn_width", m_cfg.get("width", 96))),
        pretrained=bool(m_cfg.get("pretrained", True)),
        eps_d=float(m_cfg.get("eps_d", 1e-6)),
        use_p8_context=bool(m_cfg.get("use_p8_context", True)),
    ).to(device)

    # Load initial checkpoint if provided
    init_ckpt = train_cfg.get("init_checkpoint", m_cfg.get("init_checkpoint"))
    if init_ckpt and os.path.exists(init_ckpt):
        print(f"Loading initial TeacherLite weights from {init_ckpt}...")
        ckpt = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"TeacherLite Model Parameters: Total = {total_params:,} ({total_params/1e6:.3f}M)")

    l_cfg = cfg.get("loss", {})
    criterion = TeacherCriterion(
        lambda_map=float(l_cfg.get("lambda_map", 1.0)),
        positive_cell_weight=float(l_cfg.get("positive_cell_weight", 3.0)),
        lambda_ms=float(l_cfg.get("multiscale_mae", {}).get("weight", l_cfg.get("lambda_ms", 1.0))),
        lambda_count_map=float(l_cfg.get("count_map", l_cfg.get("lambda_count_map", 0.50))),
        lambda_count_reg=float(l_cfg.get("count_reg", l_cfg.get("lambda_count_reg", 0.25))),
        lambda_consistency=float(l_cfg.get("map_reg_consistency", {}).get("weight", l_cfg.get("lambda_consistency", 0.10))),
        lambda_hn=float(l_cfg.get("lambda_hn", 0.10)),
        hard_negative_fraction=float(l_cfg.get("hard_negative_fraction", 0.10)),
        count_scale=float(l_cfg.get("count_scale", 100.0)),
        lambda_ot=float(l_cfg.get("optimal_transport", {}).get("weight", l_cfg.get("lambda_ot", 0.0))),
        lambda_tv=float(l_cfg.get("total_variation", {}).get("weight", l_cfg.get("lambda_tv", 0.0))),
        ot_reg=float(l_cfg.get("optimal_transport", {}).get("reg", 10.0)),
        ot_iterations=int(l_cfg.get("optimal_transport", {}).get("iterations", 100)),
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    opt_cfg = cfg.get("optimizer", {})
    base_backbone_lr = float(opt_cfg.get("backbone_lr", train_cfg.get("backbone_lr", 1.0e-5)))
    base_head_lr = float(opt_cfg.get("head_lr", opt_cfg.get("task_lr", train_cfg.get("head_lr", 5.0e-5))))

    use_amp = bool(train_cfg.get("amp", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    accum_steps = int(train_cfg.get("grad_accum", train_cfg.get("accum_steps", 16)))
    grad_clip = float(opt_cfg.get("grad_clip", 5.0))

    epochs = int(cfg.get("schedule", {}).get("epochs", train_cfg.get("epochs", 400)))
    warmup_epochs = int(cfg.get("schedule", {}).get("warmup_epochs", 20))
    validate_every = int(train_cfg.get("validate_every", 5))

    val_csv_path = os.path.join(save_dir, "val.csv")
    with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "mae_map", "rmse_map", "bias_map",
            "mae_reg", "rmse_reg", "bias_reg",
            "mae_sparse", "mae_med", "mae_dense",
        ])

    best_mae = float("inf")
    best_epoch = 0

    print(f"Starting TeacherLite v2 training for {epochs} epochs (Batch size={batch_size}, Grad accum={accum_steps})...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        lr_factor = get_lr_factor(epoch - 1, epochs, warmup_epochs=warmup_epochs)
        optimizer.param_groups[0]["lr"] = base_backbone_lr * lr_factor
        optimizer.param_groups[1]["lr"] = base_head_lr * lr_factor

        running_loss = 0.0
        running_details = {}
        optimizer.zero_grad(set_to_none=True)

        for step, raw_batch in enumerate(train_loader):
            batch = move_batch(raw_batch, device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(batch["image"])
                loss, details = criterion(out, batch, crop_size=crop_size)
                loss_scaled = loss / accum_steps

            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            for k, v in details.items():
                running_details[k] = running_details.get(k, 0.0) + float(v.item() if hasattr(v, "item") else v)

        n_steps = len(train_loader)
        avg_loss = running_loss / n_steps
        epoch_time = time.time() - t0

        if epoch % validate_every == 0 or epoch == epochs:
            val_res = evaluate_teacher(model, val_ds, device)

            with open(val_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    val_res["mae_map"],
                    val_res["rmse_map"],
                    val_res["bias_map"],
                    val_res["mae_reg"],
                    val_res["rmse_reg"],
                    val_res["bias_reg"],
                    val_res["mae_sparse"],
                    val_res["mae_med"],
                    val_res["mae_dense"],
                ])

            is_best = val_res["mae_map"] < best_mae
            if is_best:
                best_mae = val_res["mae_map"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_res": val_res,
                    "config": cfg,
                }, os.path.join(save_dir, "best.pt"))

            best_tag = " (Best ⭐)" if is_best else ""
            print(
                f"Epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.4f} | "
                f"Overall MAE: {val_res['mae_map']:.2f}, RMSE: {val_res['rmse_map']:.2f}{best_tag} | "
                f"Med: {val_res['mae_med']:.2f}, Dense: {val_res['mae_dense']:.2f}, Sparse: {val_res['mae_sparse']:.2f} | "
                f"LR: {optimizer.param_groups[1]['lr']:.2e} | Time: {epoch_time:.1f}s"
            )
        else:
            print(f"Epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[1]['lr']:.2e} | Time: {epoch_time:.1f}s")

    print(f"\nTeacherLite v2 Training Completed. Best Overall Val MAE: {best_mae:.2f} (Epoch {best_epoch})")


if __name__ == "__main__":
    main()
