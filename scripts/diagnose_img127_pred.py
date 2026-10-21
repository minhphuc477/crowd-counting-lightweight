import torch
import numpy as np
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.data import CrowdManifestDataset, collate_eval
from torch.utils.data import DataLoader

ckpt_path = "runs/sha_a/rmr_v63/v63_champion_occlusion_robust_1500ep/best_val_mae.pt"
model, uniform_rel, cfg, ckpt = load_model_from_ckpt(ckpt_path, device="cpu", use_ema=True)
dataset = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, output_stride=2, max_size=2048)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

for batch in loader:
    sample = batch[0]
    sid = sample.get("id")
    if sid in ["IMG_127", "IMG_63", "IMG_90"]:
        image = sample["image"].unsqueeze(0)
        target = sample["target_y"].squeeze()
        pts = sample["points"]
        h_orig, w_orig = sample["height"], sample["width"]
        
        with torch.no_grad():
            # TTA forward
            out1 = model(image)
            image_flip = torch.flip(image, dims=[-1])
            out2 = model(image_flip)
            y_flip = torch.flip(out2["y"], dims=[-1])
            y_tta = 0.5 * (out1["y"] + y_flip)[0, 0]
            
        gt_total = float(target.sum().item())
        pred_total = float(y_tta.sum().item())
        
        # Divide into vertical strips: Top (Y: 0 - 33%), Mid (Y: 33% - 66%), Bot (Y: 66% - 100%)
        gh, gw = y_tta.shape
        t_mid = gh // 3
        b_mid = 2 * gh // 3
        
        gt_top = float(target[:t_mid, :].sum().item())
        gt_mid = float(target[t_mid:b_mid, :].sum().item())
        gt_bot = float(target[b_mid:, :].sum().item())
        
        pred_top = float(y_tta[:t_mid, :].sum().item())
        pred_mid = float(y_tta[t_mid:b_mid, :].sum().item())
        pred_bot = float(y_tta[b_mid:, :].sum().item())
        
        print(f"\n=================== {sid} ===================")
        print(f"Total: GT={gt_total:.0f} | Pred={pred_total:.1f} | Err={pred_total - gt_total:+.1f}")
        print(f"Top 1/3 (Horizon/Sky): GT={gt_top:.0f} | Pred={pred_top:.1f} | Err={pred_top - gt_top:+.1f}")
        print(f"Mid 1/3 (Main Crowd):  GT={gt_mid:.0f} | Pred={pred_mid:.1f} | Err={pred_mid - gt_mid:+.1f}")
        print(f"Bot 1/3 (Foreground):  GT={gt_bot:.0f} | Pred={pred_bot:.1f} | Err={pred_bot - gt_bot:+.1f}")
