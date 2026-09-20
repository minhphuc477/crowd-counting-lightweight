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


@torch.no_grad()
def collect_regional_pairs(model, manifest_path: str, device: torch.device, output_stride: int = 4):
    ds = CrowdManifestDataset(manifest_path, train=False, output_stride=output_stride)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    reg_gt = {0: [], 1: [], 2: []}
    reg_pred = {0: [], 1: [], 2: []}
    raw_logits_empty = {0: [], 1: [], 2: []}

    for idx, batch in enumerate(loader):
        item = batch[0]
        img = item["image"].unsqueeze(0).to(device)
        target_y = item["target_y"].to(device)

        out = model(img)
        grid_h, grid_w = target_y.shape[-2:]
        reg = model._regions(grid_h, grid_w, device, stride=output_stride)

        b_gt = regional_sum(target_y.unsqueeze(0).float(), reg.boxes, out_dtype=torch.float32)
        b_pred = out.b_region.float()

        # Also get trunk features / mean_head raw logit if accessible
        for sid in [0, 1, 2]:
            mask = (reg.scale_id == sid)
            gt_s = b_gt[0, 0, mask].cpu().numpy()
            pred_s = b_pred[0, 0, mask].cpu().numpy()
            reg_gt[sid].extend(gt_s)
            reg_pred[sid].extend(pred_s)

    return reg_gt, reg_pred


def analyze_slopes(reg_gt, reg_pred, dataset_name: str):
    scale_names = {0: "32px", 1: "64px", 2: "128px"}
    thresholds = {0: [10.0], 1: [40.0], 2: [100.0, 160.0]}

    print(f"\n{'='*80}\nREGIONAL SLOPES FOR {dataset_name}\n{'='*80}")
    print(f"{'Scale':<8} {'Thresh':<10} {'N_reg':<8} | {'Mean GT':<10} {'Mean Pred':<10} {'Bias':<10} | {'Slope':<10} {'Intercept':<10} {'R^2':<8}")
    print("-" * 80)

    for sid in [0, 1, 2]:
        gts = np.array(reg_gt[sid])
        preds = np.array(reg_pred[sid])

        # All regions
        slope_all, int_all = np.polyfit(gts, preds, 1)
        r2_all = np.corrcoef(gts, preds)[0, 1] ** 2 if np.std(preds) > 1e-8 else 0.0
        bias_all = np.mean(preds - gts)
        print(f"{scale_names[sid]:<8} {'ALL':<10} {len(gts):<8} | {np.mean(gts):<10.2f} {np.mean(preds):<10.2f} {bias_all:<10.2f} | {slope_all:<10.4f} {int_all:<10.2f} {r2_all:<8.4f}")

        # Dense regions
        for th in thresholds[sid]:
            dense_mask = gts >= th
            n_d = int(dense_mask.sum())
            if n_d > 2:
                gts_d = gts[dense_mask]
                preds_d = preds[dense_mask]
                slope_d, int_d = np.polyfit(gts_d, preds_d, 1)
                r2_d = np.corrcoef(gts_d, preds_d)[0, 1] ** 2 if np.std(preds_d) > 1e-8 else 0.0
                bias_d = np.mean(preds_d - gts_d)
                print(f"{scale_names[sid]:<8} {f'>={th:.0f}':<10} {n_d:<8} | {np.mean(gts_d):<10.2f} {np.mean(preds_d):<10.2f} {bias_d:<10.2f} | {slope_d:<10.4f} {int_d:<10.2f} {r2_d:<8.4f}")
            else:
                print(f"{scale_names[sid]:<8} {f'>={th:.0f}':<10} {n_d:<8} | N/A (too few regions)")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt")
    print(f"Loading checkpoint: {ckpt_path} on {device}")
    model, _, _, _ = load_model_from_ckpt(ckpt_path, device=device, use_ema=True)
    model.eval()

    # 1. Evaluate Test Set
    print("\n>>> Collecting regional pairs on TEST set (182 images)...")
    test_gt, test_pred = collect_regional_pairs(model, "data/sha_a_test.jsonl", device)
    analyze_slopes(test_gt, test_pred, "TEST SET (182 IMAGES)")

    # 2. Evaluate Train Set
    print("\n>>> Collecting regional pairs on TRAIN set (300 images)...")
    train_gt, train_pred = collect_regional_pairs(model, "data/sha_a_train_all.jsonl", device)
    analyze_slopes(train_gt, train_pred, "TRAIN SET (300 IMAGES)")


if __name__ == "__main__":
    main()
