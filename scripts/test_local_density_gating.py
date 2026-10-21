import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.data import CrowdManifestDataset, collate_eval
from torch.utils.data import DataLoader
from rmr_core.operators.regions import RegionSet
from rmr_core.operators.adjoint import regional_adjoint

def main():
    ckpt_path = "runs/sha_a/rmr_v63/v63_champion_occlusion_robust_1500ep/best_val_mae.pt"
    manifest_path = "data/sha_a_test.jsonl"
    
    print("Loading model...")
    model, uniform_rel, cfg, ckpt = load_model_from_ckpt(ckpt_path, device="cpu", use_ema=True)
    model_cfg = getattr(model, "cfg", None)
    stride = 2 if getattr(model_cfg, "subpixel_stride2", False) else 4
    
    dataset = CrowdManifestDataset(manifest_path, train=False, output_stride=stride, max_size=2048)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)
    
    target_ids = ["IMG_157", "IMG_111", "IMG_173", "IMG_60", "IMG_175", "IMG_8", "IMG_63", "IMG_90"]
    
    print("Testing Regional Evidence Support Gating on y...")
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
            
            # Forward pass
            out = model(image)
            y = out["y"][0, 0]  # [H, W]
            raw_pred = float(y.sum().item())
            
            # Inspect regional count predictions b_solver
            # The model output has 'b_pred' or 'b_solver' or 'density_field'
            # Let's see what is in 'out'
            # In out, we have 'iterates', 's_field', 'b_pred', etc.
            
            # Let's test spatial thresholding using local average density:
            # If a pixel has no heads within 32px (local neighborhood count < 0.2 people),
            # it is background!
            k_size = 33
            pad = k_size // 2
            y_2d = y.unsqueeze(0).unsqueeze(0)
            local_sum = F.avg_pool2d(F.pad(y_2d, (pad, pad, pad, pad), mode="replicate"), 
                                     kernel_size=k_size, stride=1) * (k_size * k_size)
            
            # If local 33x33 window (66x66 original pixels) has total sum < 0.5 people, it's empty background!
            gated_preds = {}
            for empty_thresh in [0.2, 0.5, 0.8, 1.0]:
                fg_mask = torch.sigmoid((local_sum - empty_thresh) / 0.2)
                y_gated = y_2d * fg_mask
                gated_preds[empty_thresh] = float(y_gated.sum().item())
            
            results[sid] = {
                "gt": gt,
                "raw": raw_pred,
                "err_raw": raw_pred - gt,
                "gated": gated_preds,
            }
            
    print("\n=== RESULTS: Local Density Gating ===")
    print(f"{'ID':<8} | {'GT':<5} | {'Raw':<6} | {'ErrRaw':<7} | {'Thresh=0.2':<10} | {'Thresh=0.5':<10} | {'Thresh=0.8':<10}")
    print("-" * 75)
    for sid, r in results.items():
        g2 = r["gated"][0.2]
        g5 = r["gated"][0.5]
        g8 = r["gated"][0.8]
        print(f"{sid:<8} | {r['gt']:<5.0f} | {r['raw']:<6.1f} | {r['err_raw']:<+7.1f} | {g2:<6.1f}({g2-r['gt']:<+5.0f}) | {g5:<6.1f}({g5-r['gt']:<+5.0f}) | {g8:<6.1f}({g8-r['gt']:<+5.0f})")

if __name__ == "__main__":
    main()
