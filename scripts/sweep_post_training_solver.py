import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.data import CrowdManifestDataset, collate_eval
from torch.utils.data import DataLoader
import torch, numpy as np

val_ds = CrowdManifestDataset('data/sha_a_test.jsonl', train=False, output_stride=4)
loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

runs = [
    ('v49_m1_m04_anchor', 'M1 (m04 Anchor)'),
    ('v49_m4_track_b_nextgen_w64', 'M4 (Track B W64)')
]

print("=" * 85)
print(f"{'Run':<20} | {'T=0 (y0)':<10} | {'T=2':<8} | {'T=4':<8} | {'T=6 (train)':<12} | {'T=8':<8} | {'T=10':<8}")
print("=" * 85)

for r_name, label in runs:
    ckpt_p = f'runs/sha_a/{r_name}/best_val_mae.pt'
    cfg_p = f'runs/sha_a/{r_name}/resolved_config.yaml'
    
    model, _, _, _ = load_model_from_ckpt(ckpt_p, dev, use_ema=True, config_path=cfg_p)
    model.eval()
    
    # We want to test different iterations T
    # model has solver inside or unroll loop
    # Let's inspect model.solver
    orig_iter = model.config.iterations if hasattr(model.config, 'iterations') else 6
    
    t_vals = [0, 2, 4, 6, 8, 10]
    t_maes = []
    
    for t in t_vals:
        if t == 0:
            # y0 carrier
            errs = []
            with torch.no_grad():
                for batch in loader:
                    item = batch[0]
                    img = item['image'].unsqueeze(0).to(dev)
                    gt = float(item['target_y'].sum().item())
                    out = model(img)
                    errs.append(abs(float(out.y0.sum().item()) - gt))
            t_maes.append(np.mean(errs))
        else:
            # Set iterations
            if hasattr(model, 'solver') and hasattr(model.solver, 'iterations'):
                model.solver.iterations = t
            elif hasattr(model, 'iterations'):
                model.iterations = t
                
            errs = []
            with torch.no_grad():
                for batch in loader:
                    item = batch[0]
                    img = item['image'].unsqueeze(0).to(dev)
                    gt = float(item['target_y'].sum().item())
                    out = model(img)
                    errs.append(abs(float(out.y.sum().item()) - gt))
            t_maes.append(np.mean(errs))
            
    print(f"{label:<20} | {t_maes[0]:<10.2f} | {t_maes[1]:<8.2f} | {t_maes[2]:<8.2f} | {t_maes[3]:<12.2f} | {t_maes[4]:<8.2f} | {t_maes[5]:<8.2f}")

print("=" * 85)
