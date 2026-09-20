from __future__ import annotations

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval, collate_train
from rmr_core.operators import regional_sum
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.losses import negative_binomial_nll_mean_dispersion
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    scale_balanced_regional_nb_nll,
    truncated_nb_nll_loss,
)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt")
    print(f"Loading checkpoint: {ckpt_path} on {device}")
    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True)
    model.train()  # ensure gradients can flow

    output_stride = 4

    # Load a representative batch of training data (dense & medium images)
    train_ds = CrowdManifestDataset("data/sha_a_train_all.jsonl", train=True, crop_size=512, output_stride=output_stride)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_train)

    batch = next(iter(loader))
    imgs = batch["image"].to(device)
    target_ys = batch["target_y"].to(device)
    points = [p.to(device) for p in batch["points"]] if "points" in batch else None

    print(f"\nEvaluating Loss Breakdown & Gradient Norms on Batch (B=4, shapes: {imgs.shape})...")

    cfg = RMRv3LossConfig(
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.5,
        lambda_region_nb=0.2,
        lambda_hurdle=0.1,
        lambda_trunc_nb=0.2,
        count_loss_mode="nb",
        count_nb_dispersion=50.0,
        cell_loss_mode="mass_weighted",
        cell_beta=1.0,
        cell_mass_weight_eps=0.001,
        cell_mass_weight_alpha=2.0,
        cell_mass_weight_gamma=1.25,
        lambda_scale_align=0.05,
        lambda_curvature=0.5,
        curvature_gate_threshold=0.08,
        curvature_gate_kernel=5,
        curvature_gate_mode="hard",
        curvature_gate_scale=0.02,
        lambda_hard_bg=0.15,
        hard_bg_ratio=0.05,
        output_stride=4,
        dm_target="dual",
    )

    out = model(imgs)
    losses = compute_rmr_v3_losses(out, target_ys, cfg, points=points)

    print("\n" + "="*80)
    print(f"{'Loss Term':<20} {'Weight':<10} {'Raw Loss Value':<18} {'Weighted Contribution':<22}")
    print("-" * 80)
    weights = {
        "count": cfg.lambda_count,
        "flat_dm16": cfg.lambda_flat_dm16,
        "cell": cfg.lambda_cell,
        "region_nb": cfg.lambda_region_nb,
        "hurdle_bce": cfg.lambda_hurdle,
        "trunc_nb": cfg.lambda_trunc_nb,
        "scale_align": cfg.lambda_scale_align,
        "curvature": cfg.lambda_curvature,
        "hard_bg": cfg.lambda_hard_bg,
        "total": 1.0,
    }

    for k in ["count", "flat_dm16", "cell", "region_nb", "hurdle_bce", "trunc_nb", "scale_align", "curvature", "hard_bg", "total"]:
        if k in losses:
            val = float(losses[k].item())
            w = weights.get(k, 1.0)
            weighted = w * val if k != "total" else val
            print(f"{k:<20} {w:<10.2f} {val:<18.4f} {weighted:<22.4f}")

    print("="*80)

    # Gradient Norms per Loss Term on Regional Head Parameters
    print("\n" + "="*80)
    print("GRADIENT NORMS ON REGIONAL HEAD PARAMETERS (mean_head, trunk)")
    print(f"{'Loss Term':<15} | {'mean_head GradNorm':<20} | {'trunk GradNorm':<18} | {'Total Head GradNorm':<20}")
    print("-" * 80)

    terms_to_test = ["region_nb", "trunc_nb", "total"]

    for term in terms_to_test:
        model.zero_grad()
        loss_val = losses[term] if term == "total" else weights[term] * losses[term]
        loss_val.backward(retain_graph=True)

        mean_head_grad = model.region_head.mean_head.weight.grad
        trunk_grads = [p.grad for p in model.region_head.trunk.parameters() if p.grad is not None]

        norm_mean_head = mean_head_grad.norm().item() if mean_head_grad is not None else 0.0
        norm_trunk = torch.stack([g.norm() for g in trunk_grads]).norm().item() if trunk_grads else 0.0
        norm_total = (norm_mean_head**2 + norm_trunk**2)**0.5

        print(f"{term:<15} | {norm_mean_head:<20.6f} | {norm_trunk:<18.6f} | {norm_total:<20.6f}")

    print("="*80)

    # Scale-by-scale gradient analysis for regional NB loss
    print("\n" + "="*80)
    print("SCALE-BY-SCALE GRADIENT ANALYSIS (32px vs 64px vs 128px)")
    print("-" * 80)

    # Compute per-scale NB loss gradient with respect to predicted mean_region
    mu_region = out.b_region.clone().detach().requires_grad_(True)
    disp_region = out.region_dispersion.clone().detach()
    grid_h, grid_w = target_ys.shape[-2:]
    reg = model._regions(grid_h, grid_w, device, stride=output_stride)
    target_region = regional_sum(target_ys.float(), reg.boxes, out_dtype=torch.float32)

    # 1. Unweighted NB NLL per scale
    per_region_nll = negative_binomial_nll_mean_dispersion(
        target_region, mu_region, dispersion=disp_region, reduction="none"
    )

    scale_names = {0: "32px", 1: "64px", 2: "128px"}
    for sid in [0, 1, 2]:
        mask = (reg.scale_id == sid)
        loss_s = per_region_nll[..., mask].mean()
        
        if mu_region.grad is not None:
            mu_region.grad.zero_()
        loss_s.backward(retain_graph=True)

        grad_s = mu_region.grad[..., mask]
        grad_norm = grad_s.norm().item()
        grad_mean_abs = grad_s.abs().mean().item()
        mean_mu = mu_region[..., mask].mean().item()
        mean_gt = target_region[..., mask].mean().item()

        print(f"Scale {scale_names[sid]} (N_boxes={mask.sum()}):")
        print(f"  Mean GT: {mean_gt:.2f} | Mean Pred Mu: {mean_mu:.2f}")
        print(f"  Loss Value: {loss_s.item():.4f}")
        print(f"  Grad dL/d(mu) Norm: {grad_norm:.6f} | Mean |dL/d(mu)|: {grad_mean_abs:.6f}")


if __name__ == "__main__":
    main()
