import torch
import numpy as np
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.data import CrowdManifestDataset, collate_eval
from torch.utils.data import DataLoader

ckpt_path = "runs/sha_a/rmr_v63/v63_champion_occlusion_robust_1500ep/best_val_mae.pt"
manifest_path = "data/sha_a_test.jsonl"
model, uniform_rel, cfg, ckpt = load_model_from_ckpt(ckpt_path, device="cpu", use_ema=True)
dataset = CrowdManifestDataset(manifest_path, train=False, output_stride=2, max_size=2048)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

target_ids = ["IMG_157", "IMG_111", "IMG_8", "IMG_90"]
for batch in loader:
    sample = batch[0]
    sid = sample.get("id", "")
    if sid not in target_ids:
        continue
    image = sample["image"].unsqueeze(0)
    target = sample["target_y"]
    gt = float(target.sum().item())
    
    with torch.no_grad():
        out = model(image)
        y = out["y"][0, 0]
        y0 = out["iterates"][0][0, 0] if "iterates" in out else y
        
    # Check GT vs Pred mass inside GT bounding box / crowd region vs outside
    # Target points:
    pts = sample.get("points")
    # Where target > 0 vs target == 0
    gt_mask = (target > 0)
    # Dilate GT mask by 8 pixels to encompass heads
    import torch.nn.functional as F
    t_2d = target.squeeze()
    gt_dilated = F.max_pool2d(t_2d.view(1, 1, t_2d.shape[0], t_2d.shape[1]).float(), kernel_size=17, stride=1, padding=8)[0, 0] > 0
    
    mass_in_crowd = float(y[gt_dilated].sum().item())
    mass_in_bg = float(y[~gt_dilated].sum().item())
    total_pred = float(y.sum().item())
    
    y0_in_crowd = float(y0[gt_dilated].sum().item())
    y0_in_bg = float(y0[~gt_dilated].sum().item())
    
    print(f"\n--- {sid} (GT: {gt:.0f}, Pred: {total_pred:.1f}, Err: {total_pred - gt:+.1f}) ---")
    print(f"Image grid: {y.shape[0]}x{y.shape[1]} = {y.numel()} cells. Crowd cells: {int(gt_dilated.sum().item())} ({gt_dilated.float().mean()*100:.1f}%), BG cells: {int((~gt_dilated).sum().item())}")
    print(f"Final y : In-Crowd = {mass_in_crowd:.1f} ({mass_in_crowd/total_pred*100:.1f}%) | BG = {mass_in_bg:.1f} ({mass_in_bg/total_pred*100:.1f}%)")
    print(f"y0 (raw): In-Crowd = {y0_in_crowd:.1f} | BG = {y0_in_bg:.1f} | Total y0 = {float(y0.sum().item()):.1f}")
