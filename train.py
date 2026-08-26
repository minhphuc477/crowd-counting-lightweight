import os
import sys
import time
import math
import argparse
from typing import Optional
import yaml
import json
import numpy as np

# Configure cache directories on F: disk (avoid C: drive)
_base_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache"))
os.environ.setdefault("HF_HOME", os.path.join(_base_cache, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_base_cache, "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from hpc.utils.seed import seed_everything
from hpc.utils.logging import CSVLogger
from hpc.utils.checkpoint import build_checkpoint_state, save_checkpoint, load_checkpoint
from hpc.models.hpc_lite import HPCLiteSR48
from hpc.losses.criterion import HPCLossCriterion
from hpc.data.sha import ShanghaiTechDataset
from hpc.data.qnrf import UCFQNRFDataset
from hpc.data.nwpu import NWPUDataset
from hpc.data.sampler import build_density_luminance_sampler
from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics
from tools.profile_model import profile_model_efficiency


def build_optimizer_and_scheduler(
    model: nn.Module,
    criterion: nn.Module,
    opt_cfg: dict,
    total_steps: int,
    warmup_steps: int,
):
    """Build AdamW optimizer with separate parameter groups and cosine warm-up scheduler."""
    backbone_params = []
    head_neck_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_neck_params.append(param)
            
    dispersion_params = [p for p in criterion.parameters() if p.requires_grad]
    
    param_groups = [
        {"params": backbone_params, "lr": float(opt_cfg.get("backbone_lr", 2.5e-5))},
        {"params": head_neck_params, "lr": float(opt_cfg.get("head_lr", 1.0e-4))},
    ]
    if dispersion_params:
        param_groups.append(
            {"params": dispersion_params, "lr": float(opt_cfg.get("dispersion_lr", 1.0e-4))}
        )
        
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(opt_cfg.get("weight_decay", 1.0e-4)),
    )
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(float(step + 1) / max(warmup_steps, 1), 1e-4)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def custom_collate_fn(batch):
    """Collate function supporting dictionary of hierarchical blocks."""
    images = torch.stack([item["image"] for item in batch])
    gt_counts = torch.stack([item["gt_count"] for item in batch])
    gt_z_allocs = torch.stack([item["gt_z_alloc"] for item in batch])
    
    # Collect block counts per scale
    scales = list(batch[0]["gt_blocks"].keys())
    gt_blocks = {
        b: torch.stack([item["gt_blocks"][b] for item in batch])
        for b in scales
    }
    
    res = {
        "image": images,
        "gt_counts": gt_counts,
        "gt_z_alloc": gt_z_allocs,
        "gt_blocks": gt_blocks,
        "img_paths": [item["img_path"] for item in batch],
    }
    
    # Optional degraded image view with a per-sample validity mask.
    # has_degraded=True only when the dataset actually drew a random augmentation;
    # otherwise the placeholder is a clean clone and must NOT contribute to L_rob.
    if "image_degraded" in batch[0]:
        degraded = []
        has_degraded_flags = []
        for item in batch:
            deg = item.get("image_degraded")
            degraded.append(deg if deg is not None else item["image"])
            has_degraded_flags.append(bool(item.get("has_degraded", False)))
        res["image_degraded"] = torch.stack(degraded)
        res["has_degraded"] = torch.tensor(has_degraded_flags, dtype=torch.bool)

    return res


def build_dataset(cfg: dict, is_train: bool):
    """Build dataset based on configuration."""
    ds_cfg = cfg["dataset"]
    ds_name = ds_cfg["name"].lower()
    root = ds_cfg["root"]
    crop_size = ds_cfg["crop_size"]
    hnb_blocks = ds_cfg["hnb_blocks"]
    alloc_block = ds_cfg.get("allocation_block", 16)
    scale_range = tuple(ds_cfg.get("scale_range", [0.75, 2.0]))
    flip_prob = float(ds_cfg.get("flip_prob", 0.5))
    
    rob_cfg = cfg.get("robustness", {})
    second_view_prob = float(rob_cfg.get("second_view_prob", 0.30)) if is_train else 0.0
    
    if "sha" in ds_name or "shb" in ds_name:
        part = ds_cfg.get("part", "part_A" if "sha" in ds_name else "part_B")
        split = "train_data" if is_train else "test_data"
        return ShanghaiTechDataset(
            root=root,
            part=part,
            split=split,
            crop_size=crop_size,
            hnb_blocks=hnb_blocks,
            allocation_block=alloc_block,
            is_train=is_train,
            scale_range=scale_range,
            flip_prob=flip_prob,
            second_view_prob=second_view_prob,
            photometric_cfg=rob_cfg,
        )
    elif "qnrf" in ds_name:
        split = "Train" if is_train else "Test"
        return UCFQNRFDataset(
            root=root,
            split=split,
            crop_size=crop_size,
            hnb_blocks=hnb_blocks,
            allocation_block=alloc_block,
            is_train=is_train,
            scale_range=scale_range,
            flip_prob=flip_prob,
            second_view_prob=second_view_prob,
            photometric_cfg=rob_cfg,
        )
    elif "nwpu" in ds_name:
        split = "train" if is_train else "val"
        return NWPUDataset(
            root=root,
            split=split,
            crop_size=crop_size,
            hnb_blocks=hnb_blocks,
            allocation_block=alloc_block,
            is_train=is_train,
            scale_range=scale_range,
            flip_prob=flip_prob,
            second_view_prob=second_view_prob,
            photometric_cfg=rob_cfg,
        )
    else:
        raise ValueError(f"Unsupported dataset {ds_name}")


def validate(model: nn.Module, val_dataset, device: torch.device) -> dict:
    """Run validation with single-scale padded inference."""
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


def train_hpc_lite(config_path: str, resume: Optional[str] = None):
    """Main training routine.

    Args:
        config_path: Path to the YAML experiment config.
        resume: Optional path to a checkpoint .pt file.  When supplied, model
            weights, NB dispersion parameters, optimizer / scheduler / AMP
            scaler states, best_mae, and start_epoch are all restored.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    exp_cfg = cfg["experiment"]
    seed_everything(exp_cfg.get("seed", 42))
    save_dir = exp_cfg.get("save_dir", "./runs/default")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config copy
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, indent=2)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Datasets and Loaders
    train_dataset = build_dataset(cfg, is_train=True)
    val_dataset = build_dataset(cfg, is_train=False)
    
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 4)
    accum_steps = train_cfg.get("accum_steps", 2)
    num_workers = train_cfg.get("num_workers", 2)
    use_amp = train_cfg.get("amp", True) and (device.type == "cuda")
    
    sampler_cfg = cfg.get("sampler", {})
    if sampler_cfg.get("weighted", True) and len(train_dataset) > 0:
        sampler, stats = build_density_luminance_sampler(
            train_dataset.image_paths,
            train_dataset.points_list,
            num_density_bins=sampler_cfg.get("density_bins", 5),
            num_luminance_bins=sampler_cfg.get("luminance_bins", 4),
            power=sampler_cfg.get("power", 0.5),
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
        stats = {}
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True if len(train_dataset) > batch_size else False,
    )
    
    # 2. Model Initialization
    m_cfg = cfg["model"]
    model = HPCLiteSR48(
        pretrained=m_cfg.get("pretrained", True),
        neck_width=m_cfg.get("neck_width", 48),
        eps_d=float(m_cfg.get("eps_d", 1e-6)),
        route_temperature=float(m_cfg.get("route_temperature", 1.0)),
        pool_kernels=tuple(m_cfg.get("pool_kernels", [3, 5, 7])),
        pool_residual_mix=float(m_cfg.get("pool_residual_mix", 0.5)),
        simam_lambda=float(m_cfg.get("simam_lambda", 1e-4)),
    ).to(device)

    # 3. Loss Criterion
    l_cfg = cfg["loss"]
    criterion = HPCLossCriterion(
        block_sizes=cfg["dataset"]["hnb_blocks"],
        allocation_block=cfg["dataset"].get("allocation_block", 16),
        lambda_hnb=float(l_cfg.get("lambda_hnb", 1.0)),
        lambda_alloc=float(l_cfg.get("lambda_alloc", 0.5)),
        lambda_hn=float(l_cfg.get("lambda_hn", 0.25)),
        lambda_empty=float(l_cfg.get("lambda_empty", 0.5)),
        lambda_global=float(l_cfg.get("lambda_global", 0.5)),
        lambda_direct=float(l_cfg.get("lambda_direct", 0.5)),
        lambda_special=float(l_cfg.get("lambda_special", 0.25)),
        lambda_rob=float(l_cfg.get("lambda_rob", 0.1)),
        lambda_kd=float(l_cfg.get("lambda_kd", 0.0)),
        hard_negative_fraction=float(l_cfg.get("hard_negative_fraction", 0.10)),
        use_stratified_nb=l_cfg.get("density_stratified_nb", True),
        global_count_mode=l_cfg.get("global_count_mode", "log_smooth_l1"),
        special_alloc_beta=float(l_cfg.get("special_alloc_beta", 1.0)),
        enable_curriculum=l_cfg.get("enable_curriculum", True),
    ).to(device)
    
    # 4. Optimizer and LR Scheduler
    total_epochs = cfg.get("schedule", {}).get("epochs", 300)
    total_steps = max(len(train_loader) * total_epochs, 1)
    warmup_frac = cfg.get("schedule", {}).get("warmup_fraction", 0.05)
    warmup_steps = int(total_steps * warmup_frac)
    
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, criterion, cfg.get("optimizer", {}), total_steps, warmup_steps
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    # 5. Loggers — opened before resume so CSVLogger picks up existing header
    train_logger = CSVLogger(os.path.join(save_dir, "train.csv"))
    val_logger = CSVLogger(os.path.join(save_dir, "val.csv"))

    # Model Profile (run once; skip on resume to avoid overwriting on every restart)
    profile_path = os.path.join(save_dir, "profile.json")
    if not os.path.exists(profile_path):
        profile = profile_model_efficiency(
            model, input_resolution=cfg["dataset"]["crop_size"], device_name=str(device)
        )
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Model parameters: Deployed = {profile['params_deploy']:,}, Total = {profile['params_total']:,}")
    else:
        with open(profile_path) as f:
            profile = json.load(f)
        print(f"Model parameters (from profile): Deployed = {profile['params_deploy']:,}, Total = {profile['params_total']:,}")

    best_mae = float("inf")
    start_epoch = 1
    grad_clip = float(cfg.get("optimizer", {}).get("grad_clip", 1.0))

    # -----------------------------------------------------------------------
    # 6. Checkpoint Resume
    # -----------------------------------------------------------------------
    if resume is not None:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        print(f"\n{'='*60}")
        print(f"  RESUMING from: {resume}")
        print(f"{'='*60}\n")
        ckpt = load_checkpoint(
            resume,
            model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            scaler=scaler,
            device=device,
            strict=True,
        )
        # Restore training state metadata
        best_mae = float(ckpt.get("best_mae", float("inf")))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"  Resumed at epoch {start_epoch}, best_mae so far = {best_mae:.4f}")
    else:
        # Fresh start: apply data-driven head bias if configured
        if m_cfg.get("data_driven_head_bias", True):
            mean_crop = stats.get("mean_count", 50.0) * 0.5
            model.init_head_bias_from_data(
                mean_crop_count=mean_crop,
                crop_size=cfg["dataset"]["crop_size"],
                output_stride=cfg["dataset"].get("output_stride", 4),
            )

    # global_step must reflect steps already taken in previous runs so the LR
    # schedule and curriculum progress factor continue correctly.
    global_step = (start_epoch - 1) * len(train_loader)

    print(f"Starting training from epoch {start_epoch}/{total_epochs} "
          f"(global_step={global_step})...")
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        criterion.train()
        epoch_losses = []

        t0 = time.time()
        for step, batch in enumerate(train_loader):
            progress = float(global_step) / float(max(total_steps, 1))
            images = batch["image"].to(device)
            gt_blocks = {b: batch["gt_blocks"][b].to(device) for b in batch["gt_blocks"]}
            gt_z_alloc = batch["gt_z_alloc"].to(device)
            gt_counts = batch["gt_count"].to(device)

            # Special block masks for SR48 large/border emphasis
            gt_special_mask16 = None
            if "gt_special_mask16" in batch:
                gt_special_mask16 = batch["gt_special_mask16"].to(device)

            img_deg = batch.get("image_degraded", None)
            degraded_mask = batch.get("has_degraded", None)
            if img_deg is not None:
                img_deg = img_deg.to(device)
            if degraded_mask is not None:
                degraded_mask = degraded_mask.to(device)

            # Forward pass under AMP
            if use_amp:
                with torch.amp.autocast("cuda"):
                    d_clean = model(images)
                    d_deg = model(img_deg) if img_deg is not None else None
                    loss, loss_dict = criterion(
                        d_clean, gt_blocks, gt_z_alloc, gt_counts,
                        gt_special_mask16=gt_special_mask16,
                        d_degraded=d_deg, degraded_mask=degraded_mask,
                        progress=progress,
                    )
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
            else:
                d_clean = model(images)
                d_deg = model(img_deg) if img_deg is not None else None
                loss, loss_dict = criterion(
                    d_clean, gt_blocks, gt_z_alloc, gt_counts,
                    gt_special_mask16=gt_special_mask16,
                    d_degraded=d_deg, degraded_mask=degraded_mask,
                    progress=progress,
                )
                loss = loss / accum_steps
                loss.backward()
                
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    torch.nn.utils.clip_grad_norm_(criterion.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    torch.nn.utils.clip_grad_norm_(criterion.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            global_step += 1
            epoch_losses.append(loss.item() * accum_steps)
            
            # Print batch progress periodically
            if (step + 1) % max(len(train_loader) // 4, 1) == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch:03d}/{total_epochs:03d}] Step [{step+1:02d}/{len(train_loader):02d}] "
                    f"Loss: {loss.item() * accum_steps:.4f} (HNB: {loss_dict.get('loss_hnb', 0):.3f}, "
                    f"Alloc: {loss_dict.get('loss_alloc', 0):.3f}, HN: {loss_dict.get('loss_hn', 0):.3f})",
                    flush=True,
                )
            
        epoch_time = time.time() - t0
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        
        # Log train epoch
        current_lr = optimizer.param_groups[1]["lr"]
        train_logger.log({"epoch": epoch, "loss": mean_loss, "lr": current_lr, "time_s": round(epoch_time, 2)})
        
        # Run validation
        if len(val_dataset) > 0 and (epoch % 5 == 0 or epoch == total_epochs):
            val_metrics = validate(model, val_dataset, device)
            val_metrics["epoch"] = epoch
            val_logger.log(val_metrics)
            
            cur_mae = val_metrics.get("mae", float("inf"))
            is_best = cur_mae < best_mae
            if is_best:
                best_mae = cur_mae
                
            print(
                f"Epoch [{epoch:03d}/{total_epochs:03d}] Loss: {mean_loss:.4f} | "
                f"Val MAE: {cur_mae:.2f}, RMSE: {val_metrics.get('rmse', 0):.2f} "
                f"{'(Best)' if is_best else ''} | Time: {epoch_time:.1f}s"
            )
            
            save_checkpoint(
                build_checkpoint_state(
                    model,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_mae=best_mae,
                    config=cfg,
                ),
                save_dir=save_dir,
                filename="last.pt",
                is_best=is_best,
            )
        else:
            print(f"Epoch [{epoch:03d}/{total_epochs:03d}] Loss: {mean_loss:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            save_checkpoint(
                build_checkpoint_state(
                    model,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_mae=best_mae,
                    config=cfg,
                ),
                save_dir=save_dir,
                filename="last.pt",
                is_best=False,
            )
            
    print(f"Training completed. Best Val MAE: {best_mae:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to a checkpoint .pt file to resume from. "
            "Restores model weights, NB dispersion, optimizer, scheduler, "
            "AMP scaler, best_mae, and start epoch."
        ),
    )
    args = parser.parse_args()
    train_hpc_lite(args.config, resume=args.resume)
