from __future__ import annotations

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
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    topk_hard_background_loss,
)


@torch.no_grad()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt")
    print(f"Loading checkpoint: {ckpt_path} on {device}")
    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True)
    model.eval()

    output_stride = 4

    test_ds = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, output_stride=output_stride)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    medium_records = []
    all_raw_empty_logits = []
    all_hard_bg_losses = {}

    for idx, batch in enumerate(test_loader):
        item = batch[0]
        img = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)
        gt_count = float(target_y.sum().item())
        img_id = item["id"]

        out = model(img)
        pred_count = float(out.y.sum().item())
        y0_count = float(out.y0.sum().item())

        grid_h, grid_w = target_y.shape[-2:]
        reg = model._regions(grid_h, grid_w, device, stride=output_stride)
        b_gt = regional_sum(target_y.unsqueeze(0).float(), reg.boxes, out_dtype=torch.float32)
        b_pred = out.b_region.float()

        # 32px regions
        s0_mask = (reg.scale_id == 0)
        gt_s0 = b_gt[0, 0, s0_mask].cpu().numpy()
        pred_s0 = b_pred[0, 0, s0_mask].cpu().numpy()

        empty_mask = (gt_s0 == 0.0)
        total_pred_s0 = np.sum(pred_s0)
        empty_pred_s0 = np.sum(pred_s0[empty_mask])
        empty_mass_fraction = (empty_pred_s0 / total_pred_s0) if total_pred_s0 > 0 else 0.0

        # Hard background loss on this image
        hard_bg_val = float(topk_hard_background_loss(out.y, target_y.unsqueeze(0), ratio=0.05, stride=4).item())
        all_hard_bg_losses[img_id] = hard_bg_val

        signed_err = pred_count - gt_count
        abs_err = abs(signed_err)

        # Collect raw logits directly from regional head for empty boxes
        # We can pass features through regional head trunk & mean_head
        p4, p8, p16 = model._extract_carrier_features(img)
        feats = model.region_head._collect_region_features((p4, p8, p16), reg)
        h = model.region_head.trunk(feats)
        mean_raw = model.region_head.mean_head(h).squeeze(-1)[0, s0_mask].cpu().numpy()
        empty_raws = mean_raw[empty_mask]
        all_raw_empty_logits.extend(empty_raws)

        record = {
            "id": img_id,
            "gt": gt_count,
            "pred": pred_count,
            "signed_err": signed_err,
            "abs_err": abs_err,
            "empty_mass_fraction": empty_mass_fraction,
            "empty_pred_mean": float(np.mean(pred_s0[empty_mask])) if empty_mask.any() else 0.0,
            "occupied_pred_mean": float(np.mean(pred_s0[~empty_mask])) if (~empty_mask).any() else 0.0,
            "hard_bg": hard_bg_val,
        }

        if 100.0 <= gt_count <= 500.0:
            medium_records.append(record)

    # 1. Medium-Only Regression
    print(f"\n{'='*80}\n[1] MEDIUM-ONLY ANALYSIS (N={len(medium_records)})\n{'='*80}")
    med_signed = np.array([r["signed_err"] for r in medium_records])
    med_empty_frac = np.array([r["empty_mass_fraction"] for r in medium_records])
    med_abs = np.array([r["abs_err"] for r in medium_records])

    slope_m, int_m = np.polyfit(med_empty_frac, med_signed, 1)
    corr_signed = np.corrcoef(med_empty_frac, med_signed)[0, 1]
    corr_abs = np.corrcoef(med_empty_frac, med_abs)[0, 1]

    print(f"Correlation (Signed Error vs Empty Mass Fraction) on Medium: {corr_signed:+.4f} (R^2 = {corr_signed**2:.4f})")
    print(f"Linear Fit: Signed_Err = {slope_m:.2f} * EmptyMassFrac + {int_m:.2f}")
    print(f"Correlation (Absolute Error vs Empty Mass Fraction) on Medium: {corr_abs:+.4f}")

    # Inspect the 4 over-counted images
    overcounted_ids = ["IMG_112", "IMG_157", "IMG_2", "IMG_47"]
    print(f"\n--- Checking the 4 Over-counted Images in Top 10 ---")
    print(f"{'ID':<10} {'GT':<8} {'Pred':<8} {'Bias':<10} | {'EmptyMassFrac':<15} {'MeanEmptyBox':<15} {'HardBG Loss':<12}")
    print("-" * 80)
    for oid in overcounted_ids:
        # find in test records
        # find from medium or re-fetch
        r = next((x for x in medium_records if x["id"] == oid), None)
        if r is None:
            # IMG_2 is dense
            # get hard_bg
            print(f"{oid:<10} (Dense image - see hard_bg: {all_hard_bg_losses.get(oid, 0.0):.6f})")
        else:
            print(f"{r['id']:<10} {r['gt']:<8.1f} {r['pred']:<8.1f} {r['signed_err']:<+10.1f} | {r['empty_mass_fraction']*100:<14.2f}% {r['empty_pred_mean']:<15.4f} {r['hard_bg']:<12.6f}")

    # Compare hard_bg across groups
    print(f"\n{'='*80}\n[2] HARD BACKGROUND LOSS ON TEST SET\n{'='*80}")
    print(f"Overall Test (N=182) Mean HardBG: {np.mean(list(all_hard_bg_losses.values())):.6f}")
    top4_over = [all_hard_bg_losses[oid] for oid in overcounted_ids if oid in all_hard_bg_losses]
    print(f"Top 4 Over-counted Images Mean HardBG: {np.mean(top4_over):.6f}")
    worst6_under = ["IMG_8", "IMG_165", "IMG_90", "IMG_36", "IMG_127", "IMG_131"]
    top6_under = [all_hard_bg_losses[uid] for uid in worst6_under if uid in all_hard_bg_losses]
    print(f"Top 6 Under-counted Images Mean HardBG: {np.mean(top6_under):.6f}")

    # 3. Direct empirical inspection of raw logits in empty boxes
    print(f"\n{'='*80}\n[3] EMPIRICAL RAW LOGIT DISTRIBUTION IN EMPTY 32PX BOXES (N={len(all_raw_empty_logits)})\n{'='*80}")
    raws = np.array(all_raw_empty_logits)
    print(f"Mean raw logit:   {np.mean(raws):.4f}")
    print(f"Median raw logit: {np.median(raws):.4f}")
    print(f"Std dev:          {np.std(raws):.4f}")
    print(f"Min raw logit:    {np.min(raws):.4f}")
    print(f"Max raw logit:    {np.max(raws):.4f}")
    print(f"Percentiles [10%, 25%, 50%, 75%, 90%]:")
    print(f"  {np.percentile(raws, [10, 25, 50, 75, 90])}")

    # Corresponding softplus * 64
    pred_boxes = 64.0 * np.log1p(np.exp(raws))
    print(f"Implied predicted people/box (64 * softplus(raw)):")
    print(f"  Mean: {np.mean(pred_boxes):.4f} | Median: {np.median(pred_boxes):.4f} | [10%, 90%]: {np.percentile(pred_boxes, [10, 90])}")


if __name__ == "__main__":
    main()
