"""Knowledge Distillation Training Engine for HPC-Lite Student.

Trains student under combined GT supervision + Multi-Level Teacher KD.
Teacher is strictly frozen. All KD projectors are training-time only.
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
from hpc.data.sampler import build_density_luminance_sampler
from hpc.data.common import custom_collate_fn
from hpc.models.hpc_lite import HPCLiteSR48, HPCLite
from hpc.teachers.teacher_lite import TeacherLite
from hpc.losses.criterion import HPCLossCriterion
from hpc.losses.kd import MultiLevelKDLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_lr_factor(epoch: int, total_epochs: int, warmup_epochs: int = 15, min_ratio: float = 0.05) -> float:
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
    if "image_degraded" in batch:
        out["image_degraded"] = batch["image_degraded"].to(device, non_blocking=True)
        if "has_degraded" in batch:
            out["degraded_mask"] = batch["has_degraded"].to(device, non_blocking=True)
        elif "degraded_mask" in batch:
            out["degraded_mask"] = batch["degraded_mask"].to(device, non_blocking=True)
    return out


@torch.no_grad()
def evaluate_student(
    model: nn.Module,
    val_dataset: ShanghaiTechDataset,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []

    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        img = sample["image"].unsqueeze(0).to(device)
        gt = float(sample["gt_count"])

        cnt, _ = model.predict(img, pad_multiple=32)
        preds.append(float(cnt.item()))
        gts.append(gt)

    preds = np.array(preds)
    gts = np.array(gts)

    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    bias = float(np.mean(preds - gts))

    sparse = gts < 100
    med = (gts >= 100) & (gts <= 1000)
    dense = gts > 1000

    mae_sparse = float(np.mean(np.abs(preds[sparse] - gts[sparse]))) if np.any(sparse) else 0.0
    mae_med = float(np.mean(np.abs(preds[med] - gts[med]))) if np.any(med) else 0.0
    mae_dense = float(np.mean(np.abs(preds[dense] - gts[dense]))) if np.any(dense) else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mae_sparse": mae_sparse,
        "mae_med": mae_med,
        "mae_dense": mae_dense,
    }


def main():
    parser = argparse.ArgumentParser(description="Train HPC-Lite Student with Teacher KD")
    parser.add_argument("--config", type=str, default="configs/sha_kd_quick.yaml", help="Path to KD config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("experiment", {}).get("seed", 42))

    save_dir = cfg.get("experiment", {}).get("save_dir", "./runs/sha_kd_quick_from_7649")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    ds_cfg = cfg["dataset"]
    image_mean = ds_cfg.get("image_mean", [0.485, 0.456, 0.406])
    image_std = ds_cfg.get("image_std", [0.229, 0.224, 0.225])

    train_ds = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="train_data",
        crop_size=ds_cfg.get("crop_size", 448),
        hnb_blocks=ds_cfg.get("hnb_blocks", [16, 32, 64]),
        allocation_block=ds_cfg.get("allocation_block", 16),
        is_train=True,
        scale_range=tuple(cfg.get("augmentation", {}).get("scale_range", [0.75, 2.0])),
        flip_prob=float(cfg.get("augmentation", {}).get("flip_prob", 0.5)),
        image_mean=image_mean,
        image_std=image_std,
    )

    val_ds = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="test_data",
        crop_size=ds_cfg.get("crop_size", 448),
        hnb_blocks=ds_cfg.get("hnb_blocks", [16, 32, 64]),
        allocation_block=ds_cfg.get("allocation_block", 16),
        is_train=False,
        image_mean=image_mean,
        image_std=image_std,
    )

    # Sampler
    sampler_cfg = cfg.get("sampler", {})
    if sampler_cfg.get("weighted", True) and len(train_ds) > 0:
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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"].get("batch_size", 16),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=custom_collate_fn,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True if len(train_ds) > cfg["training"].get("batch_size", 16) else False,
    )

    # 1. Student Model
    st_cfg = cfg["student"]
    s_name = st_cfg.get("name", "hpc_lite_sr48")
    if s_name == "hpc_lite_sr48":
        student = HPCLiteSR48(pretrained=True, neck_width=st_cfg.get("neck_width", 48)).to(device)
    else:
        student = HPCLite(backbone_name=st_cfg.get("backbone", "mobilenetv4_conv_small_050"), pretrained=True, neck_width=st_cfg.get("neck_width", 32)).to(device)

    # Load initial checkpoint if provided
    init_ckpt = st_cfg.get("init_checkpoint")
    if init_ckpt and os.path.exists(init_ckpt):
        print(f"Loading initial student weights from {init_ckpt}...")
        ckpt = torch.load(init_ckpt, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)

    deployed_params = sum(p.numel() for p in student.parameters())
    print(f"Student Deployed Parameters: {deployed_params:,} ({deployed_params/1e6:.3f}M)")

    # 2. Teacher Model (Frozen)
    t_cfg = cfg["teacher"]
    teacher = TeacherLite(
        width=t_cfg.get("width", 96),
        pretrained=False,
        use_p8_context=bool(t_cfg.get("use_p8_context", True)),
    ).to(device)
    t_ckpt_path = t_cfg.get("checkpoint")
    if t_ckpt_path and os.path.exists(t_ckpt_path):
        print(f"Loading TeacherLite weights from {t_ckpt_path}...")
        t_ckpt = torch.load(t_ckpt_path, map_location=device)
        teacher.load_state_dict(t_ckpt["model_state_dict"] if "model_state_dict" in t_ckpt else t_ckpt)
    else:
        print(f"WARNING: Teacher checkpoint {t_ckpt_path} not found. Running with initialized weights.")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 3. Ground Truth Loss Criterion
    gt_l_cfg = cfg["student_ground_truth_loss"]
    gt_criterion = HPCLossCriterion(
        block_sizes=ds_cfg.get("hnb_blocks", [16, 32, 64]),
        allocation_block=ds_cfg.get("allocation_block", 16),
        lambda_count=float(gt_l_cfg.get("lambda_count", 1.0)),
        count_scale=float(gt_l_cfg.get("count_scale", 100.0)),
        lambda_point=float(gt_l_cfg.get("lambda_point", 0.0)),
        lambda_ms_mae=float(gt_l_cfg.get("lambda_ms_mae", 0.0)),
        lambda_hnb=float(gt_l_cfg.get("lambda_hnb", 0.35)),
        lambda_alloc=float(gt_l_cfg.get("lambda_alloc", 0.15)),
        lambda_hn=float(gt_l_cfg.get("lambda_hn", 0.10)),
        lambda_empty=float(gt_l_cfg.get("lambda_empty", 0.25)),
        lambda_global=float(gt_l_cfg.get("lambda_global", 0.10)),
        lambda_rob=float(gt_l_cfg.get("lambda_rob", 0.05)),
        lambda_route=float(gt_l_cfg.get("lambda_route", 0.10)),
        hard_negative_fraction=float(gt_l_cfg.get("hard_negative_fraction", 0.10)),
        use_stratified_nb=bool(gt_l_cfg.get("density_stratified_nb", True)),
        global_count_mode=gt_l_cfg.get("global_count_mode", "log_smooth_l1"),
        learn_dispersion=bool(gt_l_cfg.get("learn_dispersion", False)),
        enable_curriculum=bool(gt_l_cfg.get("enable_curriculum", False)),
    ).to(device)

    # 4. KD Criterion
    kd_cfg = cfg["kd"]
    kd_criterion = MultiLevelKDLoss(
        student_channels=kd_cfg.get("student_channels", {"p4": 48, "p8": 48, "p16": 48}),
        teacher_channels=kd_cfg.get("teacher_channels", {"p4": 96, "p8": 96, "p16": 96}),
        kd_dim=int(kd_cfg.get("kd_dim", 64)),
        lambda_feat=float(kd_cfg.get("lambda_feat", 0.10)),
        lambda_energy=float(kd_cfg.get("lambda_energy", 0.05)),
        lambda_relation=float(kd_cfg.get("lambda_relation", 0.05)),
        lambda_map=float(kd_cfg.get("lambda_map", 0.20)),
        lambda_count=float(kd_cfg.get("lambda_count", 0.05)),
        reliability_gate=bool(kd_cfg.get("reliability_gate", True)),
        ramp_start=float(kd_cfg.get("ramp_start", 0.05)),
        ramp_full=float(kd_cfg.get("ramp_full", 0.20)),
    ).to(device)

    # 5. Optimizer
    student_params = [p for p in student.parameters() if p.requires_grad]
    kd_proj_params = [p for p in kd_criterion.parameters() if p.requires_grad]

    base_s_lr = float(cfg["optimizer"].get("student_lr", 2.0e-5))
    base_kd_lr = float(cfg["optimizer"].get("kd_projector_lr", 1.0e-4))
    weight_decay = float(cfg["optimizer"].get("weight_decay", 1.0e-4))

    optimizer = torch.optim.AdamW(
        [
            {"params": student_params, "lr": base_s_lr, "weight_decay": weight_decay},
            {"params": kd_proj_params, "lr": base_kd_lr, "weight_decay": weight_decay},
        ]
    )

    use_amp = bool(cfg["training"].get("amp", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(cfg["schedule"].get("epochs", 300))
    warmup_epochs = int(cfg["schedule"].get("warmup_epochs", 15))
    min_ratio = float(cfg["schedule"].get("min_ratio", 0.05))
    validate_every = int(cfg["training"].get("validate_every", 5))
    grad_clip = float(cfg["optimizer"].get("grad_clip", 5.0))

    val_csv_path = os.path.join(save_dir, "val.csv")
    with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mae", "rmse", "bias", "mae_sparse", "mae_med", "mae_dense", "lr_student", "lr_kd"])

    best_mae = float("inf")
    best_epoch = 0

    s_mean = torch.tensor(image_mean, device=device).view(1, 3, 1, 1)
    s_std = torch.tensor(image_std, device=device).view(1, 3, 1, 1)
    t_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    t_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print(f"Starting KD Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        student.train()
        teacher.eval()

        progress = float(epoch - 1) / float(max(1, epochs - 1))
        lr_factor = get_lr_factor(epoch - 1, epochs, warmup_epochs=warmup_epochs, min_ratio=min_ratio)
        optimizer.param_groups[0]["lr"] = base_s_lr * lr_factor
        optimizer.param_groups[1]["lr"] = base_kd_lr * lr_factor

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, raw_batch in enumerate(train_loader):
            batch = move_batch(raw_batch, device)
            images = batch["image"]

            # Convert Student-normalized image to Teacher-normalized image
            images_raw = images * s_std + s_mean
            images_teacher = (images_raw - t_mean) / t_std

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    teacher_out = teacher(images_teacher)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                d_s, aux_s = student(images, return_aux=True)
                student_out = {
                    "density": d_s,
                    "p4": aux_s["p4"],
                    "p8": aux_s["p8"],
                    "p16": aux_s["p16"],
                }

                # GT loss
                gt_loss, gt_details = gt_criterion(
                    d_map=d_s,
                    gt_block_counts=batch["gt_blocks"],
                    gt_counts=batch["gt_count"],
                    gt_z_alloc=batch["gt_z_alloc"],
                    routes8=aux_s.get("routes8"),
                    progress=progress,
                )

                # KD loss
                kd_loss, kd_details = kd_criterion(
                    student_out=student_out,
                    teacher_out=teacher_out,
                    gt_count=batch["gt_count"],
                    progress=progress,
                )

                total_loss = gt_loss + kd_loss

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += total_loss.item()

        n_steps = len(train_loader)
        avg_loss = running_loss / n_steps
        epoch_time = time.time() - t0

        if epoch % validate_every == 0 or epoch == epochs:
            val_res = evaluate_student(student, val_ds, device)

            with open(val_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    val_res["mae"],
                    val_res["rmse"],
                    val_res["bias"],
                    val_res["mae_sparse"],
                    val_res["mae_med"],
                    val_res["mae_dense"],
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                ])

            is_best = val_res["mae"] < best_mae
            if is_best:
                best_mae = val_res["mae"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "kd_state_dict": kd_criterion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_res": val_res,
                    "config": cfg,
                }, os.path.join(save_dir, "best.pt"))

            best_tag = " (Best)" if is_best else ""
            print(
                f"Epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.4f} | "
                f"Val MAE: {val_res['mae']:.2f}, RMSE: {val_res['rmse']:.2f}{best_tag} | "
                f"Med: {val_res['mae_med']:.2f}, Dense: {val_res['mae_dense']:.2f}, Sparse: {val_res['mae_sparse']:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s"
            )
        else:
            print(f"Epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s")

    print(f"\nKD Training Completed. Best Val MAE: {best_mae:.2f} (Epoch {best_epoch})")


if __name__ == "__main__":
    main()
