import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import torch
from PIL import Image
from pathlib import Path
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.transforms import normalize_image
import torchvision.transforms.functional as TF

ckpt_path = "runs/sha_a/rmr_v66/v66_exp3_abl_alpha50_fed_1500ep/best_val_mae.pt"
device = torch.device("cpu")

model, is_ema, ckpt, cfg = load_model_from_ckpt(ckpt_path, device=device, use_ema=True)
model.eval()

manifest_path = "data/sha_a_test.jsonl"
with open(manifest_path, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

target_ids = ["IMG_8", "IMG_90", "IMG_92", "IMG_127", "IMG_157", "IMG_111"]

print("=== EXACT AUTHORITATIVE DIAGNOSTIC INSPECTION (V66 EXP 3) ===\n")

for s in samples:
    s_id = s.get("id", "")
    if s_id not in target_ids:
        continue
        
    rel_path = s.get("image", s.get("image_path"))
    img_path = Path("data") / rel_path if not os.path.isabs(rel_path) and not rel_path.startswith("data") else Path(rel_path)
    img = Image.open(img_path).convert("RGB")
    tensor_img = TF.to_tensor(img)
    tensor_img = normalize_image(tensor_img).unsqueeze(0)
    
    with torch.no_grad():
        out = model(tensor_img)
        
    gt = float(s.get("count", len(s.get("points", []))))
    y0_sum = float(out["y0"].sum().item())
    y_sum = float(out["y"].sum().item())
    
    delta = y_sum - y0_sum
    err = y_sum - gt
    
    print(f"Image: {s_id:7s} | GT: {gt:6.1f} | y0 (Fine Head): {y0_sum:6.1f} | y (Final Solver): {y_sum:6.1f} | Delta (Solver Gain): {delta:+6.1f} | Final Error: {err:+6.1f}")
