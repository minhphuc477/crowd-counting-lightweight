import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.data import CrowdManifestDataset, collate_eval

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ds = CrowdManifestDataset('data/sha_a_test.jsonl', train=False, output_stride=4)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

models = {}
for name, p in [
    ('v19', 'runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt'),
    ('v30', 'runs/sha_a/rmr_v30_step0_v19_anchor/best_val_mae.pt'),
    ('v31', 'runs/sha_a/rmr_v31_step0_v19_anchor/best_val_mae.pt'),
]:
    m, uniform, _, _ = load_model_from_ckpt(Path(p), device, use_ema=True)
    models[name] = m

records = []
for batch in loader:
    sample = batch[0]
    img = sample['image'].unsqueeze(0).to(device)
    target_y = sample['target_y'].to(device)
    gt = float(target_y.sum().item())
    img_id = Path(sample.get('path', sample.get('image', 'img'))).stem

    preds = {}
    for name, m in models.items():
        with torch.no_grad():
            out = m(img)
            preds[name] = float(out.y.sum().item())

    records.append({
        'id': img_id,
        'gt': gt,
        'pred_v19': preds['v19'],
        'pred_v30': preds['v30'],
        'pred_v31': preds['v31'],
        'ae_v19': abs(preds['v19'] - gt),
        'ae_v30': abs(preds['v30'] - gt),
        'ae_v31': abs(preds['v31'] - gt),
        'bias_v19': preds['v19'] - gt,
        'bias_v30': preds['v30'] - gt,
        'bias_v31': preds['v31'] - gt,
    })

df = pd.DataFrame(records)
dense_df = df[df['gt'] > 500].copy()
dense_df['diff_ae_v30_v19'] = dense_df['ae_v30'] - dense_df['ae_v19']

print("="*80)
print(f"DENSE SUBSET (GT > 500, N={len(dense_df)}):")
print(f"Mean GT: {dense_df['gt'].mean():.1f}")
print(f"Dense MAE: v19={dense_df['ae_v19'].mean():.2f}, v30={dense_df['ae_v30'].mean():.2f}, v31={dense_df['ae_v31'].mean():.2f}")
print(f"Dense Bias: v19={dense_df['bias_v19'].mean():.2f}, v30={dense_df['bias_v30'].mean():.2f}, v31={dense_df['bias_v31'].mean():.2f}")
print(f"Mean Pred: v19={dense_df['pred_v19'].mean():.1f}, v30={dense_df['pred_v30'].mean():.1f}, v31={dense_df['pred_v31'].mean():.1f}")

# Slope of prediction on dense
slope_v19, _ = np.polyfit(dense_df['gt'], dense_df['pred_v19'], 1)
slope_v30, _ = np.polyfit(dense_df['gt'], dense_df['pred_v30'], 1)
slope_v31, _ = np.polyfit(dense_df['gt'], dense_df['pred_v31'], 1)
print(f"Dense Slopes: v19={slope_v19:.4f}, v30={slope_v30:.4f}, v31={slope_v31:.4f}")

print("\n" + "="*80)
print("TOP 10 DENSE IMAGES WHERE V30 DEGRADED THE MOST COMPARED TO V19:")
print("="*80)
print(dense_df.sort_values('diff_ae_v30_v19', ascending=False)[['id', 'gt', 'pred_v19', 'pred_v30', 'ae_v19', 'ae_v30', 'diff_ae_v30_v19']].head(10).to_string(index=False))
