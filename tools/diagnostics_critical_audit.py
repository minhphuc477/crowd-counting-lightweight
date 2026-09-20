from __future__ import annotations

import json
import math
import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.operators import regional_sum
from rmr_v3.eval import load_model_from_ckpt
from rmr_v3.solver import unrolled_sirt_solver


@torch.no_grad()
def run_diagnostics(ckpt_path: str, device_str: str = "cuda:0") -> dict:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nEvaluating Checkpoint: {ckpt_path}\nDevice: {device}\n{'='*70}")

    model, uniform_rel, cfg_dict, provenance = load_model_from_ckpt(Path(ckpt_path), device=device, use_ema=True)
    model.eval()

    output_stride = int(model.cfg.output_stride)
    val_ds = CrowdManifestDataset(
        "data/sha_a_test.jsonl",
        train=False,
        output_stride=output_stride,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    # Lists for metrics
    sparse_err, med_err, dense_err = [], [], []
    sparse_y0_err, med_y0_err, dense_y0_err = [], [], []
    sparse_oracle_err, med_oracle_err, dense_oracle_err = [], [], []
    
    # Regional head direct errors (scale 32, 64, 128)
    reg_err_32, reg_err_64, reg_err_128, reg_err_avg = [], [], [], []

    # Iteration mass traces (averaged by category)
    traces_sparse, traces_med, traces_dense = [], [], []
    
    # Detailed image records
    image_records = []

    for idx, batch in enumerate(val_loader):
        item = batch[0]
        image = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)
        gt_count = float(target_y.sum().item())
        img_id = item["id"]

        # Forward pass through model
        out = model(image, uniform_reliability=False, solver_strength=1.0)
        
        y_final = out.y
        y0 = out.y0
        pred_count = float(y_final.sum().item())
        y0_count = float(y0.sum().item())
        
        # Iteration trace
        iter_counts = [float(it.sum().item()) for it in out.iterates]
        
        # Compute error
        abs_err = abs(pred_count - gt_count)
        abs_err_y0 = abs(y0_count - gt_count)

        # Categorize
        if gt_count < 100.0:
            cat = "sparse"
            sparse_err.append(abs_err)
            sparse_y0_err.append(abs_err_y0)
            traces_sparse.append(iter_counts)
        elif gt_count <= 500.0:
            cat = "medium"
            med_err.append(abs_err)
            med_y0_err.append(abs_err_y0)
            traces_med.append(iter_counts)
        else:
            cat = "dense"
            dense_err.append(abs_err)
            dense_y0_err.append(abs_err_y0)
            traces_dense.append(iter_counts)

        # -------------------------------------------------------------
        # Test 3A: Direct Regional Head Error & Simple Multi-scale Avg
        # -------------------------------------------------------------
        b_pred = out.b_region.float()
        grid_h, grid_w = target_y.shape[-2:]
        reg_target = model._regions(grid_h, grid_w, device, stride=output_stride)
        b_gt = regional_sum(target_y.unsqueeze(0).float(), reg_target.boxes, out_dtype=torch.float32)

        # For each scale: compute total implied image count from regional sum
        scale_counts = {}
        for s_idx, sid in enumerate(torch.unique(reg_target.scale_id)):
            s_mask = reg_target.scale_id == sid
            boxes_s = reg_target.boxes[s_mask]
            box_area = reg_target.area[s_mask].float()
            b_pred_s = b_pred[:, :, s_mask] if b_pred.shape[-1] == reg_target.boxes.shape[0] else None
            if b_pred_s is not None:
                mean_cov = float(box_area.sum().item()) / float(grid_h * grid_w)
                implied_count_s = float(b_pred_s.sum().item()) / max(mean_cov, 1e-4)
                scale_counts[int(sid.item())] = implied_count_s

        if len(scale_counts) >= 3:
            c32, c64, c128 = scale_counts[0], scale_counts[1], scale_counts[2]
            c_avg = (c32 + c64 + c128) / 3.0
            reg_err_32.append(abs(c32 - gt_count))
            reg_err_64.append(abs(c64 - gt_count))
            reg_err_128.append(abs(c128 - gt_count))
            reg_err_avg.append(abs(c_avg - gt_count))

        # -------------------------------------------------------------
        # Test 3B: Oracle q_m Upper Bound
        # Replace b_solver with b_gt, run solver
        # -------------------------------------------------------------
        with torch.no_grad():
            oracle_solver_out = unrolled_sirt_solver(
                y0=out.y0,
                b_solver=b_gt,
                weight_solver=out.solver_region_weight,
                regions=reg_target,
                iterations=model.cfg.iterations,
                omega=model.cfg.omega,
                solver_strength=1.0,
                residual_clip=model.cfg.residual_clip,
                eps=model.cfg.eps,
                solver_mode=model.cfg.solver_mode,
                density_gate_rho=model.cfg.density_gate_rho,
                density_gate_floor=model.cfg.density_gate_floor,
                proximal_tau=model.cfg.proximal_tau,
                proximal_mode=model.cfg.proximal_mode,
                proximal_mu=model.cfg.proximal_mu,
                tv_lambda=model.cfg.tv_lambda,
                tv_type=model.cfg.tv_type,
                tv_eps_c=model.cfg.tv_eps_c,
                laplace_kernel=model._laplace_kernel,
                adjoint_mode=model.cfg.adjoint_mode,
                b_variance=None,
                morozov_gamma=model.cfg.morozov_gamma,
                use_barzilai_borwein=model.cfg.use_barzilai_borwein,
                bb_clamp_min=model.cfg.bb_clamp_min,
                bb_clamp_max=model.cfg.bb_clamp_max,
                output_stride=output_stride,
                use_anscombe=model.cfg.use_anscombe_sirt,
                anscombe_c=model.cfg.anscombe_c,
                adaptive_tau=model.cfg.adaptive_tau,
                adaptive_tau_rho0=model.cfg.adaptive_tau_rho0,
                anisotropic_diffusion=model.cfg.anisotropic_diffusion,
                pm_kappa=model.cfg.pm_kappa,
                density_gated_anscombe=model.cfg.density_gated_anscombe,
                anscombe_tau_dense=model.cfg.anscombe_tau_dense,
                area_normalized_adjoint=model.cfg.area_normalized_adjoint,
            )
            y_oracle = oracle_solver_out["y"]
            oracle_count = float(y_oracle.sum().item())
            oracle_err = abs(oracle_count - gt_count)
            if cat == "sparse":
                sparse_oracle_err.append(oracle_err)
            elif cat == "medium":
                med_oracle_err.append(oracle_err)
            else:
                dense_oracle_err.append(oracle_err)

        image_records.append({
            "id": img_id,
            "category": cat,
            "gt": gt_count,
            "y0": y0_count,
            "pred": pred_count,
            "oracle": oracle_count,
            "err": abs_err,
            "bias": pred_count - gt_count,
            "iterates": iter_counts,
        })

    # Summary Statistics
    total_mae = np.mean(sparse_err + med_err + dense_err)
    total_y0_mae = np.mean(sparse_y0_err + med_y0_err + dense_y0_err)
    total_oracle_mae = np.mean(sparse_oracle_err + med_oracle_err + dense_oracle_err)
    
    sparse_mae = np.mean(sparse_err)
    med_mae = np.mean(med_err)
    dense_mae = np.mean(dense_err)

    sparse_y0 = np.mean(sparse_y0_err)
    med_y0 = np.mean(med_y0_err)
    dense_y0 = np.mean(dense_y0_err)

    sparse_oracle = np.mean(sparse_oracle_err)
    med_oracle = np.mean(med_oracle_err)
    dense_oracle = np.mean(dense_oracle_err)

    print(f"\n--- [1] Overall Accuracy & Comparison (Total samples: 182) ---")
    print(f"Overall MAE:      {total_mae:.2f} (Sparse: {sparse_mae:.2f} | Med: {med_mae:.2f} | Dense: {dense_mae:.2f})")
    print(f"No-Solver y0 MAE: {total_y0_mae:.2f} (Sparse: {sparse_y0:.2f} | Med: {med_y0:.2f} | Dense: {dense_y0:.2f})")
    print(f"Oracle q_m MAE:   {total_oracle_mae:.2f} (Sparse: {sparse_oracle:.2f} | Med: {med_oracle:.2f} | Dense: {dense_oracle:.2f})")

    if reg_err_32:
        print(f"\n--- [1B] Direct Regional Head Implied Count MAE ---")
        print(f"Scale 32px:  MAE = {np.mean(reg_err_32):.2f}")
        print(f"Scale 64px:  MAE = {np.mean(reg_err_64):.2f}")
        print(f"Scale 128px: MAE = {np.mean(reg_err_128):.2f}")
        print(f"Simple Avg (32+64+128): MAE = {np.mean(reg_err_avg):.2f}")

    # Iteration mass traces
    print(f"\n--- [2] Mass Budget Trace Across Solver Iterations (Normalized to GT = 100%) ---")
    def mean_trace(traces, cat_name):
        arr = np.array(traces)
        return np.mean(arr, axis=0)

    trace_s = mean_trace(traces_sparse, "sparse")
    trace_m = mean_trace(traces_med, "medium")
    trace_d = mean_trace(traces_dense, "dense")

    print(f"Iter:         " + "  ".join([f"T={t:<6}" for t in range(len(trace_d))]))
    print(f"Dense Mass:   " + "  ".join([f"{v:8.2f}" for v in trace_d]))
    print(f"Med Mass:     " + "  ".join([f"{v:8.2f}" for v in trace_m]))
    print(f"Sparse Mass:  " + "  ".join([f"{v:8.2f}" for v in trace_s]))

    # Scatter and Outlier Analysis on Dense Images
    dense_records = [r for r in image_records if r["category"] == "dense"]
    dense_records.sort(key=lambda x: x["err"], reverse=True)
    total_dense_err_sum = sum(r["err"] for r in dense_records)
    top5_err_sum = sum(r["err"] for r in dense_records[:5])
    top10_err_sum = sum(r["err"] for r in dense_records[:10])

    print(f"\n--- [3] Dense Outlier Dominance (47 Dense Images) ---")
    print(f"Total Dense Error Sum: {total_dense_err_sum:.1f}")
    print(f"Top 5 Dense Outliers Error:  {top5_err_sum:.1f} ({top5_err_sum / total_dense_err_sum * 100:.1f}% of total dense error!)")
    print(f"Top 10 Dense Outliers Error: {top10_err_sum:.1f} ({top10_err_sum / total_dense_err_sum * 100:.1f}% of total dense error!)")
    print("\nTop 5 Worst Dense Images:")
    for r in dense_records[:5]:
        print(f"  Image {r['id']:<10} GT: {r['gt']:<7.1f} y0: {r['y0']:<7.1f} Pred: {r['pred']:<7.1f} Oracle: {r['oracle']:<7.1f} Err: {r['err']:<7.1f} Bias: {r['bias']:<7.1f}")

    # Regression slope
    gts = np.array([r["gt"] for r in image_records])
    preds = np.array([r["pred"] for r in image_records])
    slope, intercept = np.polyfit(gts, preds, 1)
    corr = np.corrcoef(gts, preds)[0, 1]
    net_bias = np.mean(preds - gts)
    print(f"\n--- [4] Global Prediction Regression ---")
    print(f"Fit: Pred = {slope:.4f} * GT + {intercept:.2f} (R^2 = {corr**2:.4f}, Net Bias = {net_bias:.2f})")

    return {
        "total_mae": total_mae,
        "sparse_mae": sparse_mae,
        "med_mae": med_mae,
        "dense_mae": dense_mae,
        "total_y0_mae": total_y0_mae,
        "total_oracle_mae": total_oracle_mae,
        "sparse_oracle": sparse_oracle,
        "med_oracle": med_oracle,
        "dense_oracle": dense_oracle,
        "dense_records": dense_records,
    }


if __name__ == "__main__":
    ckpts = [
        "runs/sha_a/rmr_v29_step0_v19_anchor/best_val_mae.pt",
        "runs/sha_a/rmr_v29_h2_subpixel2/best_val_mae.pt",
    ]
    for c in ckpts:
        if Path(c).exists():
            run_diagnostics(c)
        else:
            print(f"Checkpoint {c} not found!")
