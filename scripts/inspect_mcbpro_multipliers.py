import torch
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.data import CrowdManifestDataset, collate_eval
from torch.utils.data import DataLoader

ckpt_path = "runs/sha_a/rmr_v63/v63_champion_occlusion_robust_1500ep/best_val_mae.pt"
model, uniform_rel, cfg, ckpt = load_model_from_ckpt(ckpt_path, device="cpu", use_ema=True)
dataset = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, output_stride=2, max_size=2048)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

for batch in loader:
    sample = batch[0]
    if sample.get("id") == "IMG_157":
        image = sample["image"].unsqueeze(0)
        with torch.no_grad():
            out = model(image)
            print(f"out keys: {list(out.keys())}")
            y = out["y"][0, 0]
            y0 = out["iterates"][0][0, 0] if "iterates" in out else y
            b_pred = out.get("b_pred")
            
        print("IMG_157 Inspection:")
        print(f"y sum: {float(y.sum().item()):.2f}, y0 sum: {float(y0.sum().item()):.2f}")
        rn_mult = out.get("rn_multiplier")
        if rn_mult is not None:
            print(f"rn_multiplier stats: min={float(rn_mult.min().item()):.4f}, mean={float(rn_mult.mean().item()):.4f}, max={float(rn_mult.max().item()):.4f}")
        b_s = out.get("b_solver")
        if b_s is not None:
            print(f"b_solver stats: min={float(b_s.min().item()):.4f}, mean={float(b_s.mean().item()):.4f}, max={float(b_s.max().item()):.4f}, sum={float(b_s.sum().item()):.2f}")
        break
