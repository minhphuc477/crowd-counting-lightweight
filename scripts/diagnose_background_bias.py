import json
import torch
import numpy as np
from pathlib import Path
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.data import CrowdManifestDataset, collate_eval
from torch.utils.data import DataLoader

def main():
    ckpt_path = "runs/sha_a/rmr_v63/v63_champion_occlusion_robust_1500ep/best_val_mae.pt"
    manifest_path = "data/sha_a_test.jsonl"
    
    print("Loading model...")
    model, uniform_rel, cfg, ckpt = load_model_from_ckpt(ckpt_path, device="cpu", use_ema=True)
    model_cfg = getattr(model, "cfg", None)
    stride = 2 if getattr(model_cfg, "subpixel_stride2", False) else 4
    
    dataset = CrowdManifestDataset(manifest_path, train=False, output_stride=stride, max_size=2048)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)
    
    # Select 5 worst overcounting images and 5 normal images to test
    target_ids = ["IMG_157", "IMG_111", "IMG_173", "IMG_60", "IMG_175", "IMG_8", "IMG_63", "IMG_90"]
    
    print(f"Running diagnostic on sample images...")
    results = {}
    with torch.no_grad():
        for batch in loader:
            sample = batch[0]
            sid = sample.get("id", "")
            if sid not in target_ids:
                continue
            image = sample["image"].unsqueeze(0)
            target = sample["target_y"]
            gt = float(target.sum().item())
            
            # Forward with TTA
            out1 = model(image)
            image_flip = torch.flip(image, dims=[-1])
            out2 = model(image_flip)
            y_flip = torch.flip(out2["y"], dims=[-1])
            y_tta = 0.5 * (out1["y"] + y_flip)[0]
            
            raw_pred = float(y_tta.sum().item())
            
            # Test various floor thresholds tau
            # Quadratic smooth floor: if y < tau: y^2 / (2*tau), else: y - 0.5*tau
            floored = {}
            for tau in [0.0005, 0.001, 0.002, 0.003, 0.005, 0.008]:
                tau_t = float(tau)
                y_pos = y_tta.clamp_min(0.0)
                y_fl = torch.where(y_pos > tau_t, y_pos - 0.5 * tau_t, y_pos.square() / (2.0 * tau_t))
                floored[tau] = float(y_fl.sum().item())
            
            # Also test hard threshold: y[y < tau] = 0
            hard_thresh = {}
            for tau in [0.0005, 0.001, 0.002, 0.003, 0.005]:
                y_pos = y_tta.clamp_min(0.0)
                y_h = torch.where(y_pos > tau, y_pos, torch.zeros_like(y_pos))
                hard_thresh[tau] = float(y_h.sum().item())

            results[sid] = {
                "gt": gt,
                "raw_pred": raw_pred,
                "err_raw": raw_pred - gt,
                "floored": floored,
                "hard_thresh": hard_thresh,
                "num_cells": y_tta.numel(),
            }

    print("\n=== DIAGNOSTIC RESULTS (Post-Processing Floor on y) ===")
    print(f"{'ID':<8} | {'GT':<5} | {'Raw':<6} | {'ErrRaw':<7} | {'tau=0.001':<9} | {'tau=0.002':<9} | {'tau=0.003':<9} | {'tau=0.005':<9}")
    print("-" * 75)
    for sid, r in results.items():
        f1 = r["floored"][0.001]
        f2 = r["floored"][0.002]
        f3 = r["floored"][0.003]
        f5 = r["floored"][0.005]
        print(f"{sid:<8} | {r['gt']:<5.0f} | {r['raw_pred']:<6.1f} | {r['err_raw']:<+7.1f} | {f1:<6.1f}({f1-r['gt']:<+5.0f}) | {f2:<6.1f}({f2-r['gt']:<+5.0f}) | {f3:<6.1f}({f3-r['gt']:<+5.0f}) | {f5:<6.1f}({f5-r['gt']:<+5.0f})")

if __name__ == "__main__":
    main()
