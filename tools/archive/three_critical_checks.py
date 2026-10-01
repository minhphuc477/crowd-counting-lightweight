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
def run_three_critical_checks():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt")
    print(f"\n{'='*70}\nRunning Three Critical Checks on Original v19 Checkpoint\nPath: {ckpt_path}\nDevice: {device}\n{'='*70}")

    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True)
    model.eval()

    output_stride = 4

    # =========================================================================
    # CHECK 1: REGIONAL LEVEL ERROR & LEARNED DISPERSION r VS GT REGIONAL COUNT
    # =========================================================================
    print("\n>>> [CHECK 1] Regional Error vs GT b_m per scale on Test Set (182 images)...")
    val_ds = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, output_stride=output_stride)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    reg_gt_all = {0: [], 1: [], 2: []}      # 32px, 64px, 128px
    reg_pred_all = {0: [], 1: [], 2: []}
    reg_disp_all = {0: [], 1: [], 2: []}    # learned dispersion r

    # Also track mass trace on Oracle vs Real for dense images
    dense_real_traces = []
    dense_oracle_traces = []
    
    # Store per-image records for outlier analysis
    test_records = []

    for idx, batch in enumerate(val_loader):
        item = batch[0]
        img = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)
        gt_count = float(target_y.sum().item())
        img_id = item["id"]
        h, w = item["height"], item["width"]

        out = model(img)
        pred_count = float(out.y.sum().item())
        y0_count = float(out.y0.sum().item())

        grid_h, grid_w = target_y.shape[-2:]
        reg = model._regions(grid_h, grid_w, device, stride=output_stride)
        b_gt = regional_sum(target_y.unsqueeze(0).float(), reg.boxes, out_dtype=torch.float32) # [1, 1, M]
        b_pred = out.b_region.float() # [1, 1, M]
        r_disp = out.region_dispersion.float() if hasattr(out, "region_dispersion") else None

        for sid in [0, 1, 2]:
            mask = (reg.scale_id == sid)
            gt_s = b_gt[0, 0, mask].cpu().numpy()
            pred_s = b_pred[0, 0, mask].cpu().numpy()
            reg_gt_all[sid].extend(gt_s)
            reg_pred_all[sid].extend(pred_s)
            if r_disp is not None:
                r_s = r_disp[0, 0, mask].cpu().numpy()
                reg_disp_all[sid].extend(r_s)

        # Oracle solver run
        oracle_solver_out = unrolled_sirt_solver(
            y0=out.y0,
            b_solver=b_gt,
            weight_solver=out.solver_region_weight,
            regions=reg,
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
        )

        real_trace = [float(it.sum().item()) for it in out.iterates]
        oracle_trace = [float(it.sum().item()) for it in oracle_solver_out["iterates"]]

        if gt_count > 500.0:
            dense_real_traces.append(real_trace)
            dense_oracle_traces.append(oracle_trace)

        oracle_count = float(oracle_solver_out["y"].sum().item())

        test_records.append({
            "id": img_id,
            "gt": gt_count,
            "pred": pred_count,
            "y0": y0_count,
            "oracle": oracle_count,
            "err": abs(pred_count - gt_count),
            "bias": pred_count - gt_count,
            "h": h,
            "w": w,
            "real_trace": real_trace,
            "oracle_trace": oracle_trace,
        })

    scale_names = {0: "32px", 1: "64px", 2: "128px"}
    for sid in [0, 1, 2]:
        gts = np.array(reg_gt_all[sid])
        preds = np.array(reg_pred_all[sid])
        mae_reg = np.mean(np.abs(preds - gts))
        mre_reg = np.mean(np.abs(preds - gts) / (gts + 1.0))
        # Regression slope on regional level
        slope_reg, int_reg = np.polyfit(gts, preds, 1)
        r2_reg = np.corrcoef(gts, preds)[0, 1] ** 2

        # Check behavior on high-density regions (gt >= 10 for 32px, >= 40 for 64px, >= 100 for 128px)
        thresh = 10.0 * (4 ** sid)
        dense_mask = gts >= thresh
        if dense_mask.any():
            dense_slope, dense_int = np.polyfit(gts[dense_mask], preds[dense_mask], 1)
            mean_gt_d = np.mean(gts[dense_mask])
            mean_pred_d = np.mean(preds[dense_mask])
            dense_bias = mean_pred_d - mean_gt_d
        else:
            dense_slope, dense_int, dense_bias = 0, 0, 0

        print(f"\n--- Regional Evidence for Scale {scale_names[sid]} (Total Regions: {len(gts)}) ---")
        print(f"  Overall MAE: {mae_reg:.3f} | MRE: {mre_reg:.3f}")
        print(f"  Linear Fit: Pred_m = {slope_reg:.4f} * GT_m + {int_reg:.3f} (R^2 = {r2_reg:.4f})")
        print(f"  Dense Regions (GT >= {thresh:.0f}, N={dense_mask.sum()}): Mean GT = {mean_gt_d:.1f}, Mean Pred = {mean_pred_d:.1f} (Bias = {dense_bias:.1f}, Slope = {dense_slope:.4f})")

        if reg_disp_all[sid]:
            disps = np.array(reg_disp_all[sid])
            print(f"  Learned r (dispersion) Overall Mean: {np.mean(disps):.2f}")
            if dense_mask.any():
                print(f"  Learned r on Dense Regions:          {np.mean(disps[dense_mask]):.2f}")
                print(f"  Learned r on Sparse Regions (GT<1):  {np.mean(disps[gts < 1.0]):.2f}")

    # =========================================================================
    # CHECK 1B: ORACLE VS REAL MASS TRACE (TESTING THE "MASS SINK" HYPOTHESIS)
    # =========================================================================
    print("\n>>> [CHECK 1B] Mass Trace: Real Evidence vs Oracle Evidence on 47 Dense Images...")
    mean_real_trace = np.mean(np.array(dense_real_traces), axis=0)
    mean_oracle_trace = np.mean(np.array(dense_oracle_traces), axis=0)
    mean_dense_gt = np.mean([r["gt"] for r in test_records if r["gt"] > 500.0])

    print(f"Dense Images Mean GT Count: {mean_dense_gt:.1f}")
    print(f"Iteration:       " + "  ".join([f"T={t:<6}" for t in range(len(mean_real_trace))]))
    print(f"Real Evidence:   " + "  ".join([f"{v:8.2f}" for v in mean_real_trace]))
    print(f"Oracle Evidence: " + "  ".join([f"{v:8.2f}" for v in mean_oracle_trace]))

    # =========================================================================
    # CHECK 2: TRAIN SET VS TEST SET ERROR ON DENSE (300 TRAIN IMAGES)
    # =========================================================================
    print("\n>>> [CHECK 2] Evaluating on Training Set (300 Train Images) to Test Optimization vs Generalization...")
    train_ds = CrowdManifestDataset("data/sha_a_train_all.jsonl", train=False, output_stride=output_stride)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    train_gts, train_preds = [], []
    train_sparse, train_med, train_dense = [], [], []

    for idx, batch in enumerate(train_loader):
        item = batch[0]
        img = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)
        gt_count = float(target_y.sum().item())
        out = model(img)
        pred_count = float(out.y.sum().item())

        err = abs(pred_count - gt_count)
        train_gts.append(gt_count)
        train_preds.append(pred_count)

        if gt_count < 100.0:
            train_sparse.append(err)
        elif gt_count <= 500.0:
            train_med.append(err)
        else:
            train_dense.append(err)

    train_gts = np.array(train_gts)
    train_preds = np.array(train_preds)
    t_slope, t_int = np.polyfit(train_gts, train_preds, 1)
    t_r2 = np.corrcoef(train_gts, train_preds)[0, 1] ** 2

    print(f"Train Overall MAE: {np.mean(np.abs(train_preds - train_gts)):.2f}")
    print(f"  Train Sparse MAE (N={len(train_sparse)}): {np.mean(train_sparse):.2f}")
    print(f"  Train Medium MAE (N={len(train_med)}): {np.mean(train_med):.2f}")
    print(f"  Train Dense MAE  (N={len(train_dense)}): {np.mean(train_dense):.2f}")
    print(f"Train Fit: Pred = {t_slope:.4f} * GT + {t_int:.2f} (R^2 = {t_r2:.4f}, Net Bias = {np.mean(train_preds - train_gts):.2f})")

    # =========================================================================
    # CHECK 3: DEEP FORENSIC ON TOP 10 WORST TEST IMAGES
    # =========================================================================
    print("\n>>> [CHECK 3] Forensic Analysis of Top 10 Worst Test Images...")
    test_records.sort(key=lambda x: x["err"], reverse=True)
    
    print(f"{'Image ID':<10} {'GT':<8} {'Pred':<8} {'y0':<8} {'Oracle':<8} {'Err':<8} {'Bias':<8} {'H x W':<12} {'Density (heads/Mpx)':<20}")
    for r in test_records[:10]:
        mpx = (r['h'] * r['w']) / 1e6
        density_mpx = r['gt'] / max(mpx, 1e-4)
        dims = f"{r['h']}x{r['w']}"
        print(f"{r['id']:<10} {r['gt']:<8.1f} {r['pred']:<8.1f} {r['y0']:<8.1f} {r['oracle']:<8.1f} {r['err']:<8.1f} {r['bias']:<8.1f} {dims:<12} {density_mpx:<20.1f}")


if __name__ == "__main__":
    run_three_critical_checks()
