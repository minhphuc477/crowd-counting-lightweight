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
    opt_cfg: dict,
    total_epochs: int,
    warmup_epochs: int = 50,
):
    """Build AdamW + warmup-cosine scheduler.

    Experiment-1 specification:
    - Uniform LR=1e-4 for all trainable model params (no backbone LR penalty).
    - No weight decay on bias / norm / 1-D params.
    - Dispersion frozen => criterion has no trainable params => not included.
    - 50-epoch warmup: 1e-5 -> 1e-4.
    - Cosine decay: 1e-4 -> 1e-6 over remaining epochs.
    """
    lr_max = float(opt_cfg.get("lr", 1.0e-4))
    lr_start = float(opt_cfg.get("lr_start", 1.0e-5))
    lr_min = float(opt_cfg.get("lr_min", 1.0e-6))
    wd = float(opt_cfg.get("weight_decay", 1.0e-4))

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if (
            p.ndim == 1
            or name.endswith(".bias")
            or "bn" in lname
            or "norm" in lname
        ):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params,    "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr_max)

    def lr_lambda(epoch: int) -> float:
        # epoch is 0-indexed here (LambdaLR calls with last_epoch)
        if epoch < warmup_epochs:
            alpha = (epoch + 1) / max(warmup_epochs, 1)
            lr = lr_start + alpha * (lr_max - lr_start)
            return lr / lr_max
        t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        t = min(t, 1.0)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))
        return lr / lr_max

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler





def custom_collate_fn(batch):
    """Collate function supporting the full SR48 training schema.

    Keys in each sample (from BaseCrowdDataset.__getitem__ train path):
        image, image_degraded, has_degraded,
        gt_blocks (dict), gt_z_alloc, gt_count,
        gt_large_mask16, gt_true_border_mask16, gt_special_mask16,
        gt_route_q, gt_route_mask, has_gt, img_path
    """
    images = torch.stack([s["image"] for s in batch])

    # Hierarchical block counts: dict[int -> Tensor]
    scales = list(batch[0]["gt_blocks"].keys())
    gt_blocks = {b: torch.stack([s["gt_blocks"][b] for s in batch]) for b in scales}

    gt_z_alloc = torch.stack([s["gt_z_alloc"] for s in batch])
    gt_count   = torch.stack([s["gt_count"]   for s in batch])   # (B,) float

    res = {
        "image":      images,
        "gt_blocks":  gt_blocks,
        "gt_z_alloc": gt_z_alloc,
        "gt_count":   gt_count,          # singular — matches training loop
        "img_path":  [s["img_path"] for s in batch],
    }

    # Special block masks (always present in train samples)
    for key in ("gt_large_mask16", "gt_true_border_mask16", "gt_special_mask16"):
        if key in batch[0]:
            res[key] = torch.stack([s[key] for s in batch])

    # SSER routing supervision targets (always present in train samples)
    if "gt_route_q" in batch[0]:
        res["gt_route_q"]    = torch.stack([s["gt_route_q"]    for s in batch])
        res["gt_route_mask"] = torch.stack([s["gt_route_mask"] for s in batch])

    # Photometric second view (always present in train samples)
    if "image_degraded" in batch[0]:
        res["image_degraded"] = torch.stack([s["image_degraded"] for s in batch])
        res["has_degraded"]   = torch.stack([s["has_degraded"]   for s in batch])

    # Per-crop annotated points for PointMassDecompositionLoss
    if "gt_points" in batch[0]:
        res["gt_points"] = [s["gt_points"] for s in batch]

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
        lambda_count=float(l_cfg.get("lambda_count", 1.0)),
        count_scale=float(l_cfg.get("count_scale", 100.0)),
        lambda_point=float(l_cfg.get("lambda_point", 1.0)),
        lambda_hnb=float(l_cfg.get("lambda_hnb", 0.25)),
        lambda_alloc=float(l_cfg.get("lambda_alloc", 0.0)),
        lambda_hn=float(l_cfg.get("lambda_hn", 0.10)),
        lambda_empty=float(l_cfg.get("lambda_empty", 0.25)),
        lambda_global=float(l_cfg.get("lambda_global", 0.10)),
        lambda_rob=float(l_cfg.get("lambda_rob", 0.05)),
        lambda_kd=float(l_cfg.get("lambda_kd", 0.0)),
        lambda_route=float(l_cfg.get("lambda_route", 0.1)),
        hard_negative_fraction=float(l_cfg.get("hard_negative_fraction", 0.10)),
        use_stratified_nb=l_cfg.get("density_stratified_nb", True),
        global_count_mode=l_cfg.get("global_count_mode", "log_smooth_l1"),
        learn_dispersion=bool(l_cfg.get("learn_dispersion", False)),
        enable_curriculum=l_cfg.get("enable_curriculum", True),
    ).to(device)

    # 4. Optimizer and LR Scheduler (epoch-based, uniform LR, no-decay grouping)
    total_epochs = cfg.get("schedule", {}).get("epochs", 2000)
    sched_cfg = cfg.get("schedule", {})
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 50))

    opt_cfg_full = dict(cfg.get("optimizer", {}))
    opt_cfg_full.setdefault("lr", 1.0e-4)
    opt_cfg_full.setdefault("lr_start", 1.0e-5)
    opt_cfg_full.setdefault("lr_min", 1.0e-6)
    opt_cfg_full.setdefault("weight_decay", 1.0e-4)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, opt_cfg_full, total_epochs, warmup_epochs
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

    # global_step used only for counting; curriculum is now epoch-based
    global_step = (start_epoch - 1) * len(train_loader)
    total_steps = max(len(train_loader) * total_epochs, 1)

    print(f"Starting training from epoch {start_epoch}/{total_epochs} "
          f"(global_step={global_step})...")
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        criterion.train()
        epoch_losses = []
        epoch_count_maes = []

        # Curriculum progress: epoch-based fraction
        progress = float(epoch - 1) / float(max(total_epochs - 1, 1))

        t0 = time.time()
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            gt_blocks = {b: batch["gt_blocks"][b].to(device) for b in batch["gt_blocks"]}
            gt_z_alloc = batch["gt_z_alloc"].to(device)
            gt_counts = batch["gt_count"].to(device)

            # Special block masks for large/border emphasis
            gt_special_mask16 = None
            if "gt_special_mask16" in batch:
                gt_special_mask16 = batch["gt_special_mask16"].to(device)

            # SSER routing supervision targets
            gt_route_q    = batch["gt_route_q"].to(device)    if "gt_route_q"    in batch else None
            gt_route_mask = batch["gt_route_mask"].to(device) if "gt_route_mask" in batch else None

            img_deg = batch.get("image_degraded", None)
            degraded_mask = batch.get("has_degraded", None)
            if img_deg is not None:
                img_deg = img_deg.to(device)
            if degraded_mask is not None:
                degraded_mask = degraded_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            gt_points = batch.get("gt_points", None)

            # Forward pass — return_aux=True to get routes8 for routing loss
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

            global_step += 1
            step_loss = loss.item() * accum_steps
            epoch_losses.append(step_loss)
            epoch_count_maes.append(float(loss_dict.get("batch_count_mae", 0.0)))

            # Print batch progress with count diagnostics
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

        # Epoch-level LR step (epoch-based schedule)
        scheduler.step()

        epoch_time = time.time() - t0
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        mean_count_mae = float(np.mean(epoch_count_maes)) if epoch_count_maes else 0.0

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
