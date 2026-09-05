from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import sys
from pathlib import Path

# Windows cp1252 stdout encoding safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import CrowdManifestDataset, collate_eval, collate_train
from .losses import LossConfig, compute_losses
from .metrics import game_single, summarize_predictions
from .model import RMRConfig, RMRCount, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(cfg: dict) -> RMRCount:
    model_cfg = cfg["model"]

    # Backward compatibility:
    # old configs did not contain update_rule.
    # They must continue to mean the historical latent implementation.
    update_rule = model_cfg.get("update_rule")
    if update_rule is None:
        update_rule = (
            "jacobian"
            if model_cfg.get("use_jacobian_gate", False)
            else "latent"
        )

    mcfg = RMRConfig(
        output_stride=model_cfg.get(
            "output_stride",
            4,
        ),
        feature_width=model_cfg.get(
            "feature_width",
            32,
        ),
        region_sizes_px=tuple(
            model_cfg.get(
                "region_sizes_px",
                [32, 64, 128],
            )
        ),
        region_overlap=model_cfg.get(
            "region_overlap",
            0.5,
        ),
        include_full_image=model_cfg.get(
            "include_full_image",
            False,
        ),
        iterations=model_cfg.get(
            "iterations",
            2,
        ),
        eta_max=model_cfg.get(
            "eta_max",
            0.20,
        ),
        eta_init=model_cfg.get(
            "eta_init",
            0.05,
        ),
        residual_clip=model_cfg.get(
            "residual_clip",
            5.0,
        ),
        update_rule=update_rule,
        use_jacobian_gate=model_cfg.get(
            "use_jacobian_gate",
            False,
        ),
        sirt_omega=model_cfg.get(
            "sirt_omega",
            1.0,
        ),
        learnable_sirt_omega=model_cfg.get(
            "learnable_sirt_omega",
            False,
        ),
        projected_use_preconditioner=model_cfg.get(
            "projected_use_preconditioner",
            False,
        ),
        detach_region_evidence=model_cfg.get(
            "detach_region_evidence",
            True,
        ),
    )

    return RMRCount(
        mcfg,
        variant=model_cfg["variant"],
    )


def make_loss_cfg(cfg: dict) -> LossConfig:
    x = cfg.get("loss", {})
    return LossConfig(
        lambda_global=x.get("lambda_global", 0.10),
        lambda_region_map=x.get("lambda_region_map", 0.20),
        lambda_region_head=x.get("lambda_region_head", 0.20),
        lambda_deep_supervision=x.get("lambda_deep_supervision", 0.0),
        cell_beta=x.get("cell_beta", 1.0),
        # P1 fix: rate magnitudes are ~0.001–0.1, so beta=2.0 puts all residuals in the
        # quadratic regime (|Δρ| ≪ 2), giving gradient ≈ Δρ/2 ≈ 0.001–0.05.
        # After lambda_region=0.2, regional branch contributes only ~0.002 to grad.
        # beta=0.1 keeps non-trivial residuals in the linear regime (L1-like),
        # giving gradient ≈ sign(Δρ) and stronger regional supervision.
        region_beta=x.get("region_beta", 0.1),
    )


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup: int):
    def fn(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-3, (epoch + 1) / max(1, warmup))
        p = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


@torch.no_grad()
def evaluate(model: RMRCount, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["target_y"].to(device)
            out = model(image)
            y = out["y"][0]
            pred = float(y.sum().item())
            gt = float(target.sum().item())
            row = {"gt": gt, "pred": pred}
            for level in range(4):
                row[f"GAME{level}"] = game_single(y, target, level)
            rows.append(row)
    return summarize_predictions(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = args.lr
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.eval_every is not None:
        cfg.setdefault("train", {})["eval_every"] = args.eval_every
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_ds = CrowdManifestDataset(
        cfg["data"]["train_manifest"],
        train=True,
        output_stride=cfg["model"].get("output_stride", 4),
        crop_size=cfg["data"].get("crop_size", 512),
        scale_range=tuple(cfg["data"].get("scale_range", [0.75, 1.25])),
    )
    val_manifest = cfg["data"].get("val_manifest")
    val_ds = None if not val_manifest else CrowdManifestDataset(
        val_manifest,
        train=False,
        output_stride=cfg["model"].get("output_stride", 4),
    )
    workers = int(cfg["train"].get("workers", 0))
    pin_mem = bool(cfg["train"].get("pin_memory", False))  # False by default for host-RAM safety
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"].get("batch_size", 8),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=(workers > 0),
        collate_fn=collate_train,
        drop_last=True,
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )

    model = make_model(cfg).to(device)
    epochs = int(cfg["train"].get("epochs", 1000))
    lr_init = float(cfg["train"].get("lr", 3e-4))
    bs = int(cfg["train"].get("batch_size", 8))
    print(
        f"\n{'=' * 80}\n"
        f"  RMR-Count Training Initialized\n"
        f"  Variant: {model.variant.upper()} | Parameters: {count_parameters(model):,} | Device: {device}\n"
        f"  Epochs: {epochs} | Batch Size: {bs} | Initial LR: {lr_init:.2e}\n"
        f"  Output Directory: {out_dir}\n"
        f"{'=' * 80}\n",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_init,
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = make_scheduler(optimizer, epochs, int(cfg["train"].get("warmup_epochs", 5)))
    amp = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_cfg = make_loss_cfg(cfg)
    grad_clip = float(cfg["train"].get("grad_clip", 5.0))
    eval_every = int(cfg["train"].get("eval_every", 10))
    solver_warmup_epochs = int(cfg["train"].get("solver_warmup_epochs", 5))
    solver_ramp_epochs = int(cfg["train"].get("solver_ramp_epochs", 20))

    start_epoch = 0
    best_mae = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", best_mae)

    # P1: expanded logging with loss components, grad norm, initial count, solver diagnostics
    log_path = out_dir / "train_log.csv"
    fieldnames = [
        "epoch", "lr", "solver_strength", "rmr_update_rule", "solver_step0", "eta0",
        "train_total", "train_cell", "train_global", "train_region_head", "train_region_map", "train_deep",
        "grad_norm_mean", "grad_norm_max", "clip_rate",
        "residual_abs_mean", "residual_abs_max", "z_lt_minus10_frac",
        "initial_pred_count_mean", "initial_pred_count_std",
        "solver_rel_step_mean", "solver_delta_n_mean",
        "solver_signed_delta_n_mean", "solver_r_spatial_mean",
        "val_MAE", "val_RMSE", "val_NAE", "val_Bias",
    ]
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in range(start_epoch, epochs):
        model.train()

        # Stabilization protocol:
        # - first solver_warmup_epochs: direct prediction/regional head learn while solver is off
        # - next solver_ramp_epochs: linearly ramp reconciliation/refinement strength to 1
        if model.variant in {"local_refine", "learned_project", "rmr"}:
            if epoch < solver_warmup_epochs:
                solver_strength = 0.0
            else:
                solver_strength = min(
                    1.0,
                    (epoch - solver_warmup_epochs + 1) / max(1, solver_ramp_epochs),
                )
            model.set_solver_strength(solver_strength)
        else:
            solver_strength = 1.0  # non-iterative variants: strength always 1 for guard below

        sums: dict[str, float] = {
            "total": 0.0, "cell": 0.0, "global": 0.0,
            "region_head": 0.0, "region_map": 0.0, "deep": 0.0,
        }
        n_steps = 0
        clipped = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        residual_abs_sum = 0.0
        residual_abs_max = 0.0
        z_sat_sum = 0.0
        init_count_sum = 0.0
        init_count_sq_sum = 0.0
        init_count_n = 0
        # Diagnostic: solver effective step & spatial redistribution
        # Detects if sigma(z)*r term is effectively zero due to z ≈ -4.6 init.
        solver_rel_step_sum = 0.0   # sum of relative L1 mass change
        solver_delta_n_sum = 0.0    # sum of |sum(Y_final) - sum(Y0)| per sample
        solver_signed_delta_n_sum = 0.0  # sum of (sum(Y_final) - sum(Y0)) per sample
        solver_r_spatial_sum = 0.0       # sum of R_spatial = 1 - |sum dy| / (sum |dy| + eps)


        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target_y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(image)
                losses = compute_losses(outputs, target, model.variant, loss_cfg)

            # P1: track initial count distribution per batch
            y0_counts = outputs["y0"].detach().sum(dim=(-3, -2, -1))  # [B]
            init_count_sum += float(y0_counts.sum().item())
            init_count_sq_sum += float((y0_counts ** 2).sum().item())
            init_count_n += y0_counts.numel()

            # Diagnostic: solver effective step & spatial redistribution
            iterates = outputs.get("iterates", [])
            if len(iterates) >= 2:
                y0_det = iterates[0].detach()
                yf_det = iterates[-1].detach()
                rel_step = float(
                    ((yf_det - y0_det).abs().sum() / (y0_det.abs().sum() + 1e-8)).item()
                )
                dy = yf_det - y0_det
                # Per-sample signed count change: sum_p dy_p
                signed_dn = dy.sum(dim=(-3, -2, -1))  # [B]
                abs_dn = signed_dn.abs()               # [B]
                # Per-sample absolute mass moved: sum_p |dy_p|
                l1_dy = dy.abs().sum(dim=(-3, -2, -1)) # [B]
                # Spatial redistribution ratio:
                # R_spatial = 1 - |sum dy| / (sum |dy| + eps)
                # 0 = pure global count shifting (unipolar)
                # 1 = pure zero-sum spatial mass redistribution
                # If solver didn't move mass (l1_dy <= 1e-6), R_spatial = 0.0
                r_sp = torch.where(
                    l1_dy > 1e-6,
                    (1.0 - (abs_dn / (l1_dy + 1e-6))).clamp(0.0, 1.0),
                    torch.zeros_like(l1_dy),
                )
                solver_rel_step_sum += rel_step
                solver_delta_n_sum += float(abs_dn.mean().item())
                solver_signed_delta_n_sum += float(signed_dn.mean().item())
                solver_r_spatial_sum += float(r_sp.mean().item())

            residuals = outputs.get("residual_fields", [])
            if residuals:
                r_last = residuals[-1].detach()
                residual_abs_sum += float(r_last.abs().mean().item())
                residual_abs_max = max(residual_abs_max, float(r_last.abs().max().item()))
            z_last = outputs.get("z")
            if z_last is not None:
                z_sat_sum += float((z_last.detach() < -10.0).float().mean().item())

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            # P1: track grad_norm mean/max, not just clip rate
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
            clipped += int(grad_norm > grad_clip)
            grad_norm_sum += grad_norm
            grad_norm_max = max(grad_norm_max, grad_norm)
            scaler.step(optimizer)
            scaler.update()

            for k in sums:
                if k in losses:
                    sums[k] += float(losses[k].detach().item())
            n_steps += 1
        # P1: record LR before stepping scheduler so logged value corresponds to current epoch
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        # P1: compute initial count distribution stats
        if init_count_n > 0:
            init_mean = init_count_sum / init_count_n
            init_var = init_count_sq_sum / init_count_n - init_mean ** 2
            init_std = float(init_var ** 0.5) if init_var > 0 else 0.0
        else:
            init_mean = init_std = 0.0

        if (
            model.variant == "rmr"
            and getattr(model, "rmr_update_rule", None) == "projected_sirt"
        ):
            solver_step0 = float(
                model._sirt_omega().detach().cpu().item()
                * model.solver_strength
            )
        elif hasattr(model, "_eta"):
            solver_step0 = float(
                model._eta(0).detach().cpu().item()
            )
        else:
            solver_step0 = 0.0
        rule_name = getattr(model, "rmr_update_rule", "latent")

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "solver_strength": solver_strength,
            "rmr_update_rule": rule_name,
            "solver_step0": solver_step0,
            "eta0": solver_step0,
            "train_total": sums["total"] / max(1, n_steps),
            "train_cell": sums["cell"] / max(1, n_steps),
            "train_global": sums["global"] / max(1, n_steps),
            "train_region_head": sums["region_head"] / max(1, n_steps),
            "train_region_map": sums["region_map"] / max(1, n_steps),
            "train_deep": sums["deep"] / max(1, n_steps),
            "grad_norm_mean": grad_norm_sum / max(1, n_steps),
            "grad_norm_max": grad_norm_max,
            "clip_rate": clipped / max(1, n_steps),
            "residual_abs_mean": residual_abs_sum / max(1, n_steps),
            "residual_abs_max": residual_abs_max,
            "z_lt_minus10_frac": z_sat_sum / max(1, n_steps),
            "initial_pred_count_mean": init_mean,
            "initial_pred_count_std": init_std,
            "solver_rel_step_mean": solver_rel_step_sum / max(1, n_steps),
            "solver_delta_n_mean": solver_delta_n_sum / max(1, n_steps),
            "solver_signed_delta_n_mean": solver_signed_delta_n_sum / max(1, n_steps),
            "solver_r_spatial_mean": solver_r_spatial_sum / max(1, n_steps),
            "val_MAE": "",
            "val_RMSE": "",
            "val_NAE": "",
            "val_Bias": "",
        }

        do_eval = val_loader is not None and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "solver_strength": solver_strength,   # P0 fix: save for provenance/audit
            "best_mae": best_mae,
            "config": cfg,
        }
        if do_eval:
            metrics = evaluate(model, val_loader, device)
            row.update({
                "val_MAE": metrics["MAE"],
                "val_RMSE": metrics["RMSE"],
                "val_NAE": metrics["NAE"],
                "val_Bias": metrics["Bias"],
            })
            # P0 fix: checkpoint guard — only update best_val_mae.pt when solver_strength == 1.0.
            # This prevents selecting checkpoints whose val metric cannot be reproduced by
            # standalone eval.py (which always loads with solver_strength = 1.0).
            solver_fully_ramped = (solver_strength >= 1.0) or (
                model.variant not in {"local_refine", "learned_project", "rmr"}
            )
            is_new_best = False
            if metrics["MAE"] < best_mae and solver_fully_ramped:
                best_mae = metrics["MAE"]
                state["best_mae"] = best_mae
                torch.save(state, out_dir / "best_val_mae.pt")
                is_new_best = True

        # Formatted readable epoch log (pure ASCII for cross-platform compatibility)
        if model.variant in {"local_refine", "learned_project", "rmr"}:
            sgn = "+" if row["solver_signed_delta_n_mean"] >= 0 else ""
            rule_str = f" [{rule_name}]" if model.variant == "rmr" else ""
            solver_str = (
                f"Solver{rule_str}: {solver_strength:4.2f} "
                f"(step: {row['solver_rel_step_mean']:.4f}, "
                f"dN: {sgn}{row['solver_signed_delta_n_mean']:.1f}, "
                f"|dN|: {row['solver_delta_n_mean']:.1f}, "
                f"R_sp: {row['solver_r_spatial_mean']:.3f})"
            )
        else:
            solver_str = "Solver: N/A"

        clip_str = f" | clip: {row['clip_rate']*100:.1f}%" if row['clip_rate'] > 0 else ""
        print(
            f"[{epoch+1:03d}/{epochs:03d}] "
            f"Loss: {row['train_total']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Grad: {row['grad_norm_mean']:.2f} (max: {row['grad_norm_max']:.2f}) | "
            f"{solver_str} | "
            f"InitCnt: {init_mean:5.1f}{clip_str}",
            flush=True,
        )

        if do_eval:
            if is_new_best:
                best_tag = " * NEW BEST"
            elif not math.isinf(best_mae):
                best_tag = f" (Best: {best_mae:.2f})"
            else:
                best_tag = " (warmup)"
            print(
                f"       --> [VAL @ Epoch {epoch+1:03d}] "
                f"MAE: {metrics['MAE']:.2f} | "
                f"RMSE: {metrics['RMSE']:.2f} | "
                f"NAE: {metrics['NAE']:.3f} | "
                f"Bias: {metrics['Bias']:+.2f}{best_tag}\n",
                flush=True,
            )
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            state["best_mae"] = best_mae
            torch.save(state, out_dir / "last.pt")

        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)

        # Explicit garbage collection to prevent host heap fragmentation across long runs
        gc.collect()


if __name__ == "__main__":
    main()
