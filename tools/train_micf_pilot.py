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
from hpc.losses.ps_fh_cmicf import PSFHCMICFLoss
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
    elif mode == "ps_fh_cmicf":
        m_cfg = cfg.get("model", {})
        k = int(m_cfg.get("finite_horizon", 4))
        return PSFHCMICFLoss(
            k=k,
            precondition_alpha=float(l_cfg.get("precondition_alpha", 0.5)),
            precondition_sv_floor=float(l_cfg.get("precondition_sv_floor", 1e-8)),
            lambda_sobolev=float(l_cfg.get("lambda_sobolev", 1.0)),
            sobolev_beta=float(l_cfg.get("sobolev_beta", 1.0)),
            lambda_count=float(l_cfg.get("lambda_count", 1.0)),
            al_rho=float(l_cfg.get("al_rho", 1.0)),
            al_dual_init=float(l_cfg.get("al_dual_init", 0.0)),
            al_dual_max=float(l_cfg.get("al_dual_max", 100.0)),
            al_update_mode=str(l_cfg.get("al_update_mode", "dual_ascent")),
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

        stride = getattr(model, "output_stride", 16)
        fh = getattr(model, "finite_horizon", None)
        req_horizon = (stride * fh) if fh is not None else stride
        eval_pad = math.lcm(64, req_horizon)

        pred_crop_count, pred_crop_map = model.predict(crop_img, pad_multiple=eval_pad)
        crop_errors.append(abs(float(pred_crop_count.item()) - gt_crop_count))

        # -------------------------------------------------------------
        # Regime B: Full-image evaluation (Direct and Tiled)
        # -------------------------------------------------------------
        pred_full_direct, _ = model.predict(img, pad_multiple=eval_pad)
        pred_full_tiled, _ = model.predict_tiled(
            img, tile_size=crop_size, halo=max(64, eval_pad)
        )

        pred_d = float(pred_full_direct.item())
        pred_t = float(pred_full_tiled.item())
        err_direct = pred_d - gt_count
        err_tiled = pred_t - gt_count
        full_direct_errors.append(abs(err_direct))
        full_tiled_errors.append(abs(err_tiled))
        full_direct_sq_errors.append(err_direct * err_direct)
        full_tiled_sq_errors.append(err_tiled * err_tiled)
        direct_tiled_discrepancies.append(abs(pred_d - pred_t))

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

            # NVR: canonical negative variation ratio Q / P
            neg_mass = float((-y_rec).clamp(min=0).sum().item())
            pos_mass = float((y_rec).clamp(min=0).sum().item())
            neg_mass_ratios.append(neg_mass / max(pos_mass, 1e-12))

    mae_crop = float(np.mean(crop_errors)) if crop_errors else 0.0
    mae_full_direct = float(np.mean(full_direct_errors)) if full_direct_errors else 0.0
    rmse_full_direct = float(np.sqrt(np.mean(full_direct_sq_errors))) if full_direct_sq_errors else 0.0
    mae_full_tiled = float(np.mean(full_tiled_errors)) if full_tiled_errors else 0.0
    rmse_full_tiled = float(np.sqrt(np.mean(full_tiled_sq_errors))) if full_tiled_sq_errors else 0.0
    mean_abs_discrepancy = float(np.mean(direct_tiled_discrepancies)) if direct_tiled_discrepancies else 0.0

    metrics = {
        "mae_crop": mae_crop,
        "mae_full_direct": mae_full_direct,
        "rmse_full_direct": rmse_full_direct,
        "mae_full_tiled": mae_full_tiled,
        "rmse_full_tiled": rmse_full_tiled,
        "mean_abs_direct_tiled_discrepancy": mean_abs_discrepancy,
        "direct_tiled_discrepancy": mean_abs_discrepancy,
        "mae_direct_tiled_difference": abs(mae_full_direct - mae_full_tiled),
        "mae": mae_crop,
        "mae_full": mae_full_tiled if is_cumulative else mae_full_direct,
        "rmse": rmse_full_tiled if is_cumulative else rmse_full_direct,
    }
    if is_cumulative:
        metrics["violation_rate"] = float(np.mean(violation_rates)) if violation_rates else 0.0
        metrics["violation_magnitude"] = float(np.mean(violation_magnitudes)) if violation_magnitudes else 0.0
        metrics["nvr"] = float(np.mean(neg_mass_ratios)) if neg_mass_ratios else 0.0
        metrics["neg_mass_ratio"] = metrics["nvr"]
    else:
        metrics["violation_rate"] = 0.0
        metrics["violation_magnitude"] = 0.0
        metrics["nvr"] = 0.0
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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from")
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
    num_workers = int(t_cfg.get("num_workers", 0))
    pin_memory = bool(t_cfg.get("pin_memory", False))
    # Orientation balancing: independent vertical flip prob (horizontal done in dataset)
    vflip_prob = float(aug_cfg.get("vflip_prob", 0.5))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ntpc_collate_fn,
        drop_last=bool(t_cfg.get("drop_last", True)),
        pin_memory=pin_memory,
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
        finite_horizon=m_cfg.get("finite_horizon", None),
        fh_strict_local=bool(m_cfg.get("fh_strict_local", False)),
        fh_local_norm=str(m_cfg.get("fh_local_norm", "group")),
    ).to(device)

    criterion = build_criterion(cfg).to(device)
    if hasattr(criterion, "preconditioner"):
        p = criterion.preconditioner
        print(
            f"PS-FH preconditioner | K={p.k} | alpha={p.alpha:.2f} | "
            f"kappa(T)={p.prefix_condition_number:.3f} | kappa_eff={p.quadratic_condition_number:.3f}",
            flush=True,
        )

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
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=float(t_cfg.get("scaler_init_scale", 128.0)),
    )

    start_epoch = 1
    best_mae = float("inf")
    history: List[Dict[str, Any]] = []
    last_grad_shares: Dict[str, float] = {}

    is_cumulative = model.head_type in {"cumulative", "integrated_local"}
    is_ps_fh = isinstance(criterion, PSFHCMICFLoss)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Loading checkpoint for resume from {resume_path}...", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if is_ps_fh and "criterion_state" in ckpt:
            criterion.load_state_dict(ckpt["criterion_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_mae = float(ckpt.get("best_mae", float("inf")))

        # Load existing history if available
        hist_path = save_dir / "history.json"
        if hist_path.is_file():
            try:
                with open(hist_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing history: {e}")
        print(
            f"Successfully resumed from epoch {ckpt.get('epoch')} -> starting epoch {start_epoch}, "
            f"best_mae={best_mae:.2f}",
            flush=True,
        )

    print(
        f"Model: {m_cfg.get('head_type')} head | context={m_cfg.get('use_integral_context')} | "
        f"Training {start_epoch}..{total_epochs} epochs (warmup={warmup_epochs}) | grad_clip={grad_clip}"
    )
    print(f"Orientation balancing: hflip_prob={aug_cfg.get('flip_prob', 0.5)} (dataset) "
          f"vflip_prob={vflip_prob} (training loop)", flush=True)

    # ------------------------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_ps_components: Dict[str, List[float]] = {}
        epoch_c_blocks: List[torch.Tensor] = []
        epoch_grad_norms_before: List[float] = []
        epoch_grad_norms_after: List[float] = []
        epoch_clip_triggers: List[float] = []
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
            out_s = model.output_stride
            fh = getattr(model, "finite_horizon", None)
            req_horizon = (out_s * fh) if fh is not None else out_s
            target_pad = math.lcm(64, req_horizon)
            pyramid = build_exact_count_pyramid(
                points_batch,
                height=H,
                width=W,
                block_sizes=(out_s,),
                pad_multiple=target_pad,
                device=device,
            )
            y_target = pyramid[out_s]         # [B, H/out_s, W/out_s]
            if y_target.ndim == 3:
                y_target = y_target.unsqueeze(1)   # -> [B, 1, H/out_s, W/out_s]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if is_ps_fh:
                    pred_field, aux = model.forward_field_with_aux(images)
                    c_target = cell_counts_to_cumulative_field(y_target, orientation="TL")
                    loss, comp = criterion(
                        pred_c=pred_field,
                        target_c=c_target,
                        target_y=y_target,
                        pred_c_blocks=aux["c_blocks"],
                        return_components=True,
                    )
                    for k_c, v_c in comp.items():
                        epoch_ps_components.setdefault(k_c, []).append(v_c)
                    epoch_c_blocks.append(-aux["y_blocks"].detach())
                elif is_cumulative:
                    pred_field = model.forward_field(images)
                    # Build cumulative target from the (possibly flipped) Y
                    c_target = cell_counts_to_cumulative_field(y_target, orientation="TL")
                    loss = criterion(pred_field, c_target, target_y=y_target)
                elif isinstance(criterion, IntegralLossOnLocalCount):
                    pred_field = model.forward_field(images)
                    loss = criterion(pred_field, y_target)
                else:
                    pred_field = model.forward_field(images)
                    loss = criterion(pred_field, y_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            gn_before_val = float(grad_norm_before.item())
            if math.isfinite(gn_before_val):
                gn_after_val = min(gn_before_val, grad_clip)
                clip_trig = 1.0 if gn_before_val > grad_clip else 0.0
                epoch_grad_norms_before.append(gn_before_val)
                epoch_grad_norms_after.append(gn_after_val)
                epoch_clip_triggers.append(clip_trig)
            else:
                epoch_clip_triggers.append(1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))

            # Gradient diagnostics every 25 epochs on batch 0
            if is_ps_fh and (epoch % 25 == 0 or epoch == 1) and batch_idx == 0:
                grad_diag = criterion.compute_gradient_diagnostics(
                    pred_c=pred_field,
                    target_c=c_target,
                    target_y=y_target,
                    pred_c_blocks=aux["c_blocks"],
                )
                g_pc = float(grad_diag["grad_pc"])
                g_sob = float(grad_diag["grad_sobolev"])
                g_count = float(grad_diag["grad_count"])
                g_al = float(grad_diag["grad_al"])
                g_sum = g_pc + g_sob + g_count + g_al
                denom = max(g_sum, 1e-12)
                share_pc = (g_pc / denom) * 100.0
                share_sob = (g_sob / denom) * 100.0
                share_count = (g_count / denom) * 100.0
                share_al = (g_al / denom) * 100.0
                last_grad_shares = {
                    "grad_share_pc": share_pc,
                    "grad_share_sobolev": share_sob,
                    "grad_share_count": share_count,
                    "grad_share_al": share_al,
                }
                print(
                    f"  [Epoch {epoch:4d} Gradient Diagnostics] "
                    f"grad_pc: {g_pc:.4e} ({share_pc:.1f}%) | "
                    f"grad_sobolev: {g_sob:.4e} ({share_sob:.1f}%) | "
                    f"grad_count: {g_count:.4e} ({share_count:.1f}%) | "
                    f"grad_al: {g_al:.4e} ({share_al:.1f}%) | "
                    f"Shares: PC {share_pc:.1f}% | Sob {share_sob:.1f}% | "
                    f"Count {share_count:.1f}% | AL {share_al:.1f}%",
                    flush=True,
                )

            if args.smoke_test and batch_idx >= 2:
                break

        epoch_time = time.time() - t0
        mean_loss = float(np.mean(epoch_losses))

        # Augmented Lagrangian dual update per epoch (phase-wise multipliers lambda_uv)
        if is_ps_fh and hasattr(criterion, "update_dual") and epoch_c_blocks:
            all_c = torch.cat(epoch_c_blocks, dim=0)
            new_lambda = criterion.update_dual(all_c)

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
            mae_full_direct = val_res["mae_full_direct"]
            rmse_full_direct = val_res["rmse_full_direct"]
            mae_full_tiled = val_res["mae_full_tiled"]
            rmse_full_tiled = val_res["rmse_full_tiled"]
            mean_abs_discrepancy = val_res["mean_abs_direct_tiled_discrepancy"]
            mae_direct_tiled_diff = val_res["mae_direct_tiled_difference"]
            viol = val_res["violation_rate"]
            nvr = val_res.get("nvr", val_res.get("neg_mass_ratio", 0.0))

            is_best = mae_crop < best_mae
            if is_best:
                best_mae = mae_crop
                ckpt_data = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_mae": best_mae,
                    "mae_crop": mae_crop,
                    "mae_full_direct": mae_full_direct,
                    "rmse_full_direct": rmse_full_direct,
                    "mae_full_tiled": mae_full_tiled,
                    "rmse_full_tiled": rmse_full_tiled,
                    "mean_abs_direct_tiled_discrepancy": mean_abs_discrepancy,
                    "mae_direct_tiled_difference": mae_direct_tiled_diff,
                    "direct_tiled_gap": mean_abs_discrepancy,
                    "mae_full": val_res.get("mae_full"),
                    "config": cfg,
                    "seed": seed,
                }
                if is_ps_fh:
                    ckpt_data["criterion_state"] = criterion.state_dict()
                    p = criterion.preconditioner
                    ckpt_data["preconditioner"] = {
                        "k": p.k,
                        "alpha": p.alpha,
                        "min_singular_value": p.min_singular_value,
                        "max_singular_value": p.max_singular_value,
                        "prefix_condition_number": p.prefix_condition_number,
                        "quadratic_condition_number": p.quadratic_condition_number,
                    }
                torch.save(ckpt_data, save_dir / "best.pt")

            log_entry = {
                "epoch": epoch,
                "loss": mean_loss,
                "lr": optimizer.param_groups[-1]["lr"],
                "mae_crop": mae_crop,
                "mae_full_direct": mae_full_direct,
                "rmse_full_direct": rmse_full_direct,
                "mae_full_tiled": mae_full_tiled,
                "rmse_full_tiled": rmse_full_tiled,
                "mean_abs_direct_tiled_discrepancy": mean_abs_discrepancy,
                "mae_direct_tiled_difference": mae_direct_tiled_diff,
                "direct_tiled_gap": mean_abs_discrepancy,
                "mae": mae_crop,
                "mae_full": val_res.get("mae_full"),
                "rmse": rmse_full_tiled if is_cumulative else rmse_full_direct,
                "violation_rate": viol,
                "nvr": nvr,
                "negative_variation_ratio": nvr,
                "best_mae": best_mae,
                "time_sec": epoch_time,
                "grad_norm_before_clip": float(np.mean(epoch_grad_norms_before)) if epoch_grad_norms_before else 0.0,
                "grad_norm_after_clip": float(np.mean(epoch_grad_norms_after)) if epoch_grad_norms_after else 0.0,
                "clip_trigger_rate": float(np.mean(epoch_clip_triggers)) if epoch_clip_triggers else 0.0,
                "clip_triggered": bool(np.any(epoch_clip_triggers)) if epoch_clip_triggers else False,
            }
            if is_ps_fh:
                for k_ps in [
                    "ps_pc_loss",
                    "ps_sobolev_loss",
                    "sobolev_pos_loss",
                    "sobolev_zero_loss",
                    "ps_count_loss",
                    "ps_violation_magnitude",
                    "ps_constraint",
                    "ps_dual_lambda",
                    "ps_dual_lambda_mean",
                    "ps_dual_lambda_max",
                    "ps_dual_lambda_terminal",
                    "ps_al_rho",
                    "ps_aug_lagrangian",
                    "positive_cell_fraction",
                    "zero_cell_fraction",
                ]:
                    if k_ps in epoch_ps_components:
                        log_entry[k_ps] = float(np.mean(epoch_ps_components[k_ps]))

            # Add MICF-specific diagnostics if present
            for k in ("violation_magnitude", "neg_mass_ratio", "nvr"):
                if k in val_res:
                    log_entry[k] = val_res[k]
            if "neg_mass_ratio" in val_res:
                log_entry["negative_mass_ratio"] = val_res["neg_mass_ratio"]
            if last_grad_shares:
                log_entry.update(last_grad_shares)

            history.append(log_entry)

            # Format extra MICF diagnostics for display
            diag_str = ""
            if is_cumulative and not args.smoke_test:
                diag_str = (
                    f" | ViolMag: {val_res.get('violation_magnitude', 0):.3f}"
                    f" | NVR: {nvr*100:.2f}%"
                )
            if is_ps_fh:
                dual_max = float(criterion.al_lambda.max().item())
                dual_term = float(criterion.al_lambda[0, 0, -1, -1].item())
                diag_str += f" | DualLamMax: {dual_max:.3f} | DualLam(3,3): {dual_term:.3f}"

            print(
                f"[Epoch {epoch:4d}/{total_epochs}] Loss: {mean_loss:.4f} | "
                f"Crop: {mae_crop:.2f} | Direct: {mae_full_direct:.2f} | Tiled: {mae_full_tiled:.2f} | "
                f"Discrep: {mean_abs_discrepancy:.2f} | Viol: {viol*100:.2f}%{diag_str} | Best_crop: {best_mae:.2f} "
                f"{'(*)' if is_best else ''} ({epoch_time:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"[Epoch {epoch:4d}/{total_epochs}] Loss: {mean_loss:.4f} "
                f"lr={optimizer.param_groups[-1]['lr']:.2e} ({epoch_time:.1f}s)",
                flush=True,
            )

        if epoch % 10 == 0 or epoch == total_epochs:
            last_ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_mae": best_mae,
                "config": cfg,
                "seed": seed,
            }
            if is_ps_fh:
                last_ckpt["criterion_state"] = criterion.state_dict()
            torch.save(last_ckpt, save_dir / "last.pt")

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
