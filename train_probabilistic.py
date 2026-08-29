"""Training script for HPC-Lite: Adaptive Hierarchical Probabilistic Count Tree."""

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
from hpc.losses.count_tree import CountTreeConfig
from hpc.losses.hpc_adaptive import AdaptiveHPCLoss, HPCLossConfig
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
    parser = argparse.ArgumentParser(description="Train HPC-Lite Probabilistic Hierarchy Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    seed_everything(exp_cfg.get("seed", 42))
    save_dir = exp_cfg.get("save_dir", "./runs/sha_hpc_lite_probabilistic")
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

    # 2. Model Construction
    m_cfg = cfg["model"]
    model = HPCLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050"),
        pretrained=m_cfg.get("pretrained", True),
        neck_width=m_cfg.get("neck_width", 32),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_p8_context=bool(m_cfg.get("use_p8_context", True)),
        use_repblock=bool(m_cfg.get("use_repblock", False)),
        eps_d=float(m_cfg.get("eps_d", 1e-6)),
        truncate_backbone=bool(m_cfg.get("truncate_backbone", True)),
    ).to(device)

    # Optional initial checkpoint loading
    init_ckpt = m_cfg.get("init_checkpoint")
    if init_ckpt and os.path.exists(init_ckpt):
        print(f"Loading initial student weights from {init_ckpt} (strict=False)...")
        ckpt = torch.load(init_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        # If loading from non-repblock checkpoint, transfer head_dw weights to rbr_dense
        if model.use_repblock and "head_dw.weight" in state_dict and model.head_refine is not None:
            model.head_refine.rbr_dense[0].weight.data.copy_(state_dict["head_dw.weight"])
            print("  Transferred pre-trained head_dw weights to RepDWBlock dense branch!")

    p_count = sum(p.numel() for p in model.parameters())
    print(f"HPC-Lite Deployed Parameters: {p_count:,} ({p_count/1e6:.3f}M)")

    # 3. Probabilistic Hierarchy Loss Criterion
    pt_cfg = cfg.get("prob_tree", {})
    w_cfg = pt_cfg.get("weights", {})
    hz_cfg = cfg.get("hard_zero", {})
    lc_cfg = cfg.get("local_contrast", {})
    stat_cfg = cfg.get("statistics", {})

    loss_config = HPCLossConfig(
        tree=CountTreeConfig(
            root_dispersion=float(stat_cfg.get("root_dispersion", 50.0)),
            kappa_root64=float(pt_cfg.get("kappa_root64", 20.0)),
            kappa_64_32=float(pt_cfg.get("kappa_64_32", 20.0)),
            kappa_32_16=float(pt_cfg.get("kappa_32_16", 20.0)),
            kappa_16_8=float(pt_cfg.get("kappa_16_8", 20.0)),
            kappa_flat16=float(pt_cfg.get("kappa_flat16", 20.0)),
            dense_threshold_16=int(stat_cfg.get("dense_threshold_16", 4)),
            use_dirichlet_multinomial=(pt_cfg.get("distribution", "dirichlet_multinomial") == "dirichlet_multinomial"),
            w_root_nb=float(w_cfg.get("root_nb", 1.0)),
            w_root64=float(w_cfg.get("root_to_64", 1.0)),
            w_64_32=float(w_cfg.get("64_to_32", 1.0)),
            w_32_16=float(w_cfg.get("32_to_16", 1.0)),
            w_16_8=float(w_cfg.get("16_to_8_dense", 1.0)),
            w_flat_16=float(w_cfg.get("flat_16", 0.0)),
            w_indep_nb=float(w_cfg.get("indep_nb", 0.0)),
        ),
        hard_zero_weight=float(hz_cfg.get("weight", 0.10)) if hz_cfg.get("enabled", True) else 0.0,
        local_contrast_weight=float(lc_cfg.get("weight", 0.05)) if lc_cfg.get("enabled", True) else 0.0,
        exact_count_weight=float(cfg.get("exact_count", {}).get("weight", 0.0)),
        hard_zero_top_fraction=float(hz_cfg.get("top_fraction", 0.10)),
        local_low_threshold=int(stat_cfg.get("local_t1", 1)),
        local_dense_threshold=int(stat_cfg.get("local_t2", 4)),
        local_max_samples=int(lc_cfg.get("max_samples", 256)),
        local_temperature=float(lc_cfg.get("temperature", 0.10)),
    )

    criterion = AdaptiveHPCLoss(loss_config, feature_dim=32).to(device)

    # 4. Optimizer & Schedule
    opt_cfg = cfg.get("optimizer", {})
    wd = float(opt_cfg.get("weight_decay", 1.0e-4))
    grad_clip = float(opt_cfg.get("grad_clip", 5.0))

    if "differential_lr" in opt_cfg and opt_cfg["differential_lr"]:
        diff_lr = opt_cfg["differential_lr"]
        lr_existing = float(diff_lr.get("existing", opt_cfg.get("lr", 1.0e-4)))
        lr_new = float(diff_lr.get("new_training_modules", opt_cfg.get("lr", 1.0e-4)))
        param_groups = [
            {"params": list(model.parameters()), "lr": lr_existing},
            {"params": list(criterion.local_contrast.projector.parameters()), "lr": lr_new},
        ]
    else:
        unified_lr = float(opt_cfg.get("lr", 1.0e-4))
        all_params = list(model.parameters()) + list(criterion.local_contrast.projector.parameters())
        param_groups = [{"params": all_params, "lr": unified_lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    total_epochs = cfg.get("schedule", {}).get("epochs", 300)
    warmup_epochs = cfg.get("schedule", {}).get("warmup_epochs", 15)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs and warmup_epochs > 0:
            return (epoch + 1) / float(warmup_epochs)
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        t = min(t, 1.0)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = bool(cfg.get("training", {}).get("amp", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(cfg.get("training", {}).get("init_scale", 256.0)),
        enabled=use_amp,
    )
    validate_every = int(cfg.get("training", {}).get("validate_every", 5))

    val_csv_path = os.path.join(save_dir, "val.csv")
    with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mae", "rmse", "nae", "sparse_mae", "med_mae", "dense_mae", "lr"])

    best_mae = float("inf")
    best_epoch = 0

    print(f"Starting HPC-Lite Probabilistic Hierarchy Training for {total_epochs} epochs...")

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_root = 0.0
        running_root64 = 0.0
        running_64_32 = 0.0
        running_32_16 = 0.0
        running_16_8 = 0.0
        running_hz = 0.0
        running_lc = 0.0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            target_pyramid = {int(k): v.to(device, non_blocking=True) for k, v in batch["gt_blocks"].items()}
            target_pyramid["N"] = batch["gt_count"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                density_map, aux = model(images, return_aux=True)  # (B, 1, H/4, W/4)
                loss, logs = criterion(
                    mass=density_map,
                    p4=aux["p4"],
                    target_pyramid=target_pyramid,
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                all_params = list(model.parameters()) + list(criterion.local_contrast.projector.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                all_params = list(model.parameters()) + list(criterion.local_contrast.projector.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_root += logs["root_nb"].item()
            running_root64 += logs["root_to_64"].item()
            running_64_32 += logs["64_to_32"].item()
            running_32_16 += logs["32_to_16"].item()
            running_16_8 += logs["16_to_8_dense"].item()
            running_hz += logs["hard_zero"].item()
            running_lc += logs["local_contrast"].item()

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
            print(
                f"Epoch [{epoch:03d}/{total_epochs}] Loss: {avg_loss:.2f} "
                f"(Root: {running_root/n_steps:.2f}, R64: {running_root64/n_steps:.2f}, 64->32: {running_64_32/n_steps:.2f}, 32->16: {running_32_16/n_steps:.2f}, 16->8: {running_16_8/n_steps:.2f}, HZ: {running_hz/n_steps:.3f}, LC: {running_lc/n_steps:.2f}) | "
                f"Val MAE: {mae:.2f}, RMSE: {rmse:.2f}{best_tag} | "
                f"Med: {med_mae:.2f}, Dense: {dense_mae:.2f}, Sparse: {sparse_mae:.2f} | Time: {epoch_time:.1f}s"
            )
        else:
            print(
                f"Epoch [{epoch:03d}/{total_epochs}] Loss: {avg_loss:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s"
            )

    print(f"\nHPC-Lite Probabilistic Hierarchy Training Completed. Best Val MAE: {best_mae:.2f} (Epoch {best_epoch})")


if __name__ == "__main__":
    main()
