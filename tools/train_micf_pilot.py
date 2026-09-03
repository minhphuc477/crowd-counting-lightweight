"""Unified Trainer for the 6-Model MICF Pilot Suite (B1 to B6).

Runs controlled experiments on ShanghaiTech Part A:
B1: Local Count Baseline (SmoothL1 on Y)
B2: Local Output + Integral Loss (SmoothL1 on P(Y_hat) vs P(Y))
B3: Direct Cumulative MICF Naive (SmoothL1 on C_hat vs C, lambda_valid=0)
B4: Direct Cumulative MICF + Validity (lambda_valid=1.0)
B5: MICF-v2 Full (4-dir Directional Context + Validity)
B6: Local Count + Directional Context (Ablation Control)

Key training decisions (from design doc):
- Orientation-balanced augmentation (sec.28): random H+V flips in the training loop
  after dataset crop, updating gt_points and regenerating Y and C targets.
  This balances origin bias from single-corner TL cumulative supervision.
- LR warmup (sec.39): linear warmup for warmup_epochs, then cosine decay.
- Gradient clipping: 5.0 (design doc sec.39).
- Evaluation diagnostics (sec.21/22): corner-delta count gap, negative mass ratio,
  violation magnitude, in addition to MAE/RMSE.
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
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Orientation-balanced augmentation (design doc section 28)
# ---------------------------------------------------------------------------

def orientation_balanced_flip(
    images: torch.Tensor,
    points_batch: List[torch.Tensor],
    vflip_prob: float = 0.5,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Apply independent random vertical flips in-place on a training batch.

    The dataset already applies random horizontal flip at crop time.
    This function adds the orthogonal vertical flip so both H and V axes
    are independently balanced, making each point's total contribution
    equal across all four cumulative orientations (design doc sec.9):
        (H-x)(W-y) + (H-x)y + x(W-y) + xy = HW.

    Args:
        images: [B, 3, H, W] crop batch.
        points_batch: list of B tensors, each [N_i, 2] in (x, y) pixel coords.
        vflip_prob: probability of vertical flip per sample.

    Returns:
        (images_out, points_batch_out) — new tensors, same shapes.
    """
    B, _, H, W = images.shape
    images_out = images.clone()
    points_out = []

    for i in range(B):
        pts = points_batch[i].clone()
        if torch.rand(()).item() < vflip_prob:
            # Flip image vertically: row r -> H-1-r
            images_out[i] = torch.flip(images_out[i], dims=[-2])
            # Flip point y-coordinates: y -> (H-1) - y
            if pts.numel() > 0:
                pts[:, 1] = float(H - 1) - pts[:, 1]
        points_out.append(pts)

    return images_out, points_out


# ---------------------------------------------------------------------------
# LR warmup + cosine decay (design doc section 39)
# ---------------------------------------------------------------------------

def get_lr_scale(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup then cosine annealing.

    Returns a scalar multiplier in [0, 1] for the base LR.
    epoch is 1-indexed (epoch=1 is first epoch).
    """
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    # Cosine decay from warmup_epochs to total_epochs
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Criterion builder
# ---------------------------------------------------------------------------

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
            lambda_local_recon=float(l_cfg.get("lambda_local_recon", 0.0)),
            beta_smooth=float(l_cfg.get("beta_smooth", 1.0)),
            normalize_by=l_cfg.get("normalize_by", "none"),
            norm_eps=float(l_cfg.get("norm_eps", 1.0)),
        )
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


# ---------------------------------------------------------------------------
# Evaluation with rich MICF diagnostics (design doc sections 21, 22)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
    crop_size: int = 256,
) -> Dict[str, float]:
    """Evaluate model under Regime A (crop-level) and Regime B (full image).

    - Regime A (mae_crop): isolates representation geometry on fixed 256x256 crops
      (matching training crop size, avoiding receptive-field mismatch).
    - Regime B (mae_full): evaluates full scenes using hierarchical tile composition
      for cumulative models (Section 30 of design doc).
    - Measure Diagnostics (Section 22):
      - violation_rate: fraction of cells where Delta_xy C_hat < 0 (f_-)
      - violation_magnitude: mean(ReLU(-Delta_xy C_hat)) over full grid (V)
      - neg_mass_ratio: negative mass fraction r_-
    """
    model.eval()
    crop_errors: List[float] = []
    full_errors: List[float] = []
    full_direct_errors: List[float] = []
    full_tiled_errors: List[float] = []
    full_sq_errors: List[float] = []
    violation_rates: List[float] = []
    violation_magnitudes: List[float] = []
    neg_mass_ratios: List[float] = []

    is_cumulative = getattr(model, "head_type", "") in {"cumulative", "integrated_local"}

    for idx, batch in enumerate(val_loader):
        if max_samples is not None and idx >= max_samples:
            break
        img = batch["image"].to(device)
        gt_count = float(batch["gt_count"].item())
        gt_pts = batch["gt_points"][0]
        if isinstance(gt_pts, np.ndarray):
            gt_pts = torch.from_numpy(gt_pts).float()
        gt_pts = gt_pts.to(device)

        _, _, H, W = img.shape

        # -------------------------------------------------------------
        # Regime A: Fixed 256x256 central crop evaluation
        # -------------------------------------------------------------
        top = max(0, (H - crop_size) // 2)
        left = max(0, (W - crop_size) // 2)
        crop_h = min(H, crop_size)
        crop_w = min(W, crop_size)
        crop_img = img[:, :, top:top + crop_h, left:left + crop_w]

        if gt_pts.numel() > 0:
            px = gt_pts[:, 0]
            py = gt_pts[:, 1]
            in_crop = (px >= left) & (px < left + crop_w) & (py >= top) & (py < top + crop_h)
            gt_crop_count = float(in_crop.sum().item())
        else:
            gt_crop_count = 0.0

        pred_crop_count, pred_crop_map = model.predict(crop_img, pad_multiple=64)
        crop_errors.append(abs(float(pred_crop_count.item()) - gt_crop_count))

        # -------------------------------------------------------------
        # Regime B: Full-image evaluation (Direct and Tiled)
        # -------------------------------------------------------------
        pred_full_direct, _ = model.predict(img, pad_multiple=64)
        pred_full_tiled, _ = model.predict_tiled(img, tile_size=crop_size)

        err_direct = float(pred_full_direct.item()) - gt_count
        err_tiled = float(pred_full_tiled.item()) - gt_count
        full_direct_errors.append(abs(err_direct))
        full_tiled_errors.append(abs(err_tiled))

        pred_full_count = pred_full_tiled if is_cumulative else pred_full_direct
        pred_full_val = float(pred_full_count.item())
        err_full = pred_full_val - gt_count
        full_errors.append(abs(err_full))
        full_sq_errors.append(err_full * err_full)

        if is_cumulative:
            # Measure diagnostics on the crop field
            y_rec = discrete_mixed_difference(pred_crop_map)
            viol_rate = float((y_rec < 0).float().mean().item())
            violation_rates.append(viol_rate)

            # V: canonical mean(ReLU(-Y)) over the grid (Section 22)
            viol_mag = float(F.relu(-y_rec).mean().item())
            violation_magnitudes.append(viol_mag)

            # r_-: negative mass ratio
            neg_mass = float((-y_rec).clamp(min=0).sum().item())
            total_abs = float(y_rec.abs().sum().item()) + 1e-6
            neg_mass_ratios.append(neg_mass / total_abs)

    mae_crop = float(np.mean(crop_errors)) if crop_errors else 0.0
    mae_full = float(np.mean(full_errors)) if full_errors else 0.0
    mae_full_direct = float(np.mean(full_direct_errors)) if full_direct_errors else 0.0
    mae_full_tiled = float(np.mean(full_tiled_errors)) if full_tiled_errors else 0.0
    rmse_full = float(np.sqrt(np.mean(full_sq_errors))) if full_sq_errors else 0.0

    metrics = {
        "mae_crop": mae_crop,
        "mae_full": mae_full,
        "mae_full_direct": mae_full_direct,
        "mae_full_tiled": mae_full_tiled,
        "mae": mae_crop,      # Primary decision metric is Regime A (isolated geometry)
        "rmse": rmse_full,
    }
    if is_cumulative:
        metrics["violation_rate"] = float(np.mean(violation_rates)) if violation_rates else 0.0
        metrics["violation_magnitude"] = float(np.mean(violation_magnitudes)) if violation_magnitudes else 0.0
        metrics["neg_mass_ratio"] = float(np.mean(neg_mass_ratios)) if neg_mass_ratios else 0.0
    else:
        metrics["violation_rate"] = 0.0
        metrics["violation_magnitude"] = 0.0
        metrics["neg_mass_ratio"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MICF-v2 Pilot Suite Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test on a small subset")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg.get("experiment", {})
    m_id = exp_cfg.get("model_id", "MICF")
    save_dir = Path(exp_cfg.get("save_dir", f"./runs/pilot_micf/{m_id.lower()}"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    seed = args.seed or exp_cfg.get("seed", 42)
    set_seed(seed)

    print("=" * 80)
    print(f"STARTING MICF PILOT: {m_id} - {exp_cfg.get('description', '')}")
    print(f"Device: {device} | Seed: {seed} | Save Dir: {save_dir}")
    print("=" * 80, flush=True)

    # ------------------------------------------------------------------
    # 1. Dataset & DataLoader
    # ------------------------------------------------------------------
    ds_cfg = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})
    train_dataset = ShanghaiTechDataset(
        root=ds_cfg["root"],
        part=ds_cfg.get("part", "part_A"),
        split="train_data",
        crop_size=int(ds_cfg.get("crop_size", 256)),
        is_train=True,
        scale_range=tuple(aug_cfg.get("scale_range", [0.7, 1.3])),
        flip_prob=float(aug_cfg.get("flip_prob", 0.5)),   # horizontal flip in dataset
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
    # Orientation balancing: independent vertical flip prob (horizontal done in dataset)
    vflip_prob = float(aug_cfg.get("vflip_prob", 0.5))

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

    # ------------------------------------------------------------------
    # 2. Model & Loss
    # ------------------------------------------------------------------
    m_cfg = cfg.get("model", {})
    model = MICFLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050.e3000_r224_in1k"),
        pretrained=bool(m_cfg.get("pretrained", True)),
        neck_width=int(m_cfg.get("neck_width", 32)),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_integral_context=bool(m_cfg.get("use_integral_context", False)),
        context_type=str(m_cfg.get("context_type", "directional")),
        head_type=m_cfg.get("head_type", "cumulative"),
        output_stride=int(m_cfg.get("output_stride", 16)),
        eps_d=float(m_cfg.get("eps_d", 1e-8)),
        extent_aware=bool(m_cfg.get("extent_aware", False)),
    ).to(device)

    criterion = build_criterion(cfg)

    # ------------------------------------------------------------------
    # 3. Optimizer & LR schedule (warmup + cosine, design doc sec.39)
    # ------------------------------------------------------------------
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1e-4))
    backbone_lr_scale = float(opt_cfg.get("backbone_lr_scale", 0.1))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    grad_clip = float(opt_cfg.get("grad_clip", 5.0))   # design doc: 5.0

    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    # Store initial LRs BEFORE the epoch loop (fixes P0 catastrophic warmup decay)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    sched_cfg = cfg.get("schedule", {})
    total_epochs = args.epochs or int(sched_cfg.get("epochs", 1000))
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 25))
    if args.smoke_test:
        total_epochs = 1
        warmup_epochs = 0

    use_amp = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_mae = float("inf")
    history: List[Dict[str, Any]] = []

    is_cumulative = model.head_type in {"cumulative", "integrated_local"}
    print(
        f"Model: {m_cfg.get('head_type')} head | context={m_cfg.get('use_integral_context')} | "
        f"Training {total_epochs} epochs (warmup={warmup_epochs}) | grad_clip={grad_clip}"
    )
    print(f"Orientation balancing: hflip_prob={aug_cfg.get('flip_prob', 0.5)} (dataset) "
          f"vflip_prob={vflip_prob} (training loop)", flush=True)

    # ------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------
    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        t0 = time.time()

        # Apply warmup + cosine LR scaling from stored initial_lr
        lr_scale = get_lr_scale(epoch, warmup_epochs, total_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lr_scale

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            points_batch = [pts.to(device) for pts in batch["gt_points"]]
            B, _, H, W = images.shape

            # Orientation-balanced augmentation: independent vertical flip (sec.28)
            # Horizontal flip is already handled by the dataset's geom_transform.
            images, points_batch = orientation_balanced_flip(images, points_batch, vflip_prob)

            # Rebuild exact count pyramid from (possibly vertically flipped) points
            pyramid = build_exact_count_pyramid(
                points_batch,
                height=H,
                width=W,
                block_sizes=(16,),
                pad_multiple=64,
                device=device,
            )
            y_target = pyramid[16]         # [B, H/16, W/16]
            if y_target.ndim == 3:
                y_target = y_target.unsqueeze(1)   # -> [B, 1, H/16, W/16]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_field = model.forward_field(images)

                if is_cumulative:
                    # Build cumulative target from the (possibly flipped) Y
                    c_target = cell_counts_to_cumulative_field(y_target, orientation="TL")
                    loss = criterion(pred_field, c_target, target_y=y_target)
                elif isinstance(criterion, IntegralLossOnLocalCount):
                    loss = criterion(pred_field, y_target)
                else:
                    loss = criterion(pred_field, y_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))

            if args.smoke_test and batch_idx >= 2:
                break

        epoch_time = time.time() - t0
        mean_loss = float(np.mean(epoch_losses))

        # ------------------------------------------------------------------
        # 5. Evaluation + logging (Regime A: Crop, Regime B: Full)
        # ------------------------------------------------------------------
        eval_every = 1 if args.smoke_test else int(t_cfg.get("evaluate_every", 5))
        if epoch % eval_every == 0 or epoch == total_epochs:
            val_res = evaluate_model(
                model,
                val_loader,
                device,
                max_samples=5 if args.smoke_test else None,
                crop_size=int(ds_cfg.get("crop_size", 256)),
            )
            mae_crop = val_res["mae_crop"]
            mae_full = val_res["mae_full"]
            rmse_full = val_res["rmse"]
            viol = val_res["violation_rate"]

            is_best = mae_crop < best_mae
            if is_best:
                best_mae = mae_crop
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_mae": best_mae,
                        "mae_crop": mae_crop,
                        "mae_full": mae_full,
                        "config": cfg,
                        "seed": seed,
                    },
                    save_dir / "best.pt",
                )

            log_entry = {
                "epoch": epoch,
                "loss": mean_loss,
                "lr": optimizer.param_groups[-1]["lr"],
                "mae_crop": mae_crop,
                "mae_full": mae_full,
                "mae": mae_crop,
                "rmse": rmse_full,
                "violation_rate": viol,
                "best_mae": best_mae,
                "time_sec": epoch_time,
            }
            # Add MICF-specific diagnostics if present
            for k in ("violation_magnitude", "neg_mass_ratio"):
                if k in val_res:
                    log_entry[k] = val_res[k]
            history.append(log_entry)

            # Format extra MICF diagnostics for display
            diag_str = ""
            if is_cumulative and not args.smoke_test:
                diag_str = (
                    f" | ViolMag: {val_res.get('violation_magnitude', 0):.3f}"
                    f" | NegMass: {val_res.get('neg_mass_ratio', 0)*100:.1f}%"
                )

            print(
                f"[Epoch {epoch:4d}/{total_epochs}] Loss: {mean_loss:.4f} | "
                f"MAE_crop: {mae_crop:.2f} | MAE_full: {mae_full:.2f} | "
                f"Viol: {viol*100:.2f}%{diag_str} | Best_crop: {best_mae:.2f} "
                f"{'(*)' if is_best else ''} ({epoch_time:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"[Epoch {epoch:4d}/{total_epochs}] Loss: {mean_loss:.4f} "
                f"lr={optimizer.param_groups[-1]['lr']:.2e} ({epoch_time:.1f}s)",
                flush=True,
            )

        if args.smoke_test:
            break

    # ------------------------------------------------------------------
    # 6. Save final history
    # ------------------------------------------------------------------
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining completed. Results saved to {save_dir}", flush=True)
    print(f"Best MAE: {best_mae:.2f}", flush=True)


if __name__ == "__main__":
    main()
