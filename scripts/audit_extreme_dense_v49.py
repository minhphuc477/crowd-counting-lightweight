import json
import numpy as np
import pandas as pd

with open("runs/v49_per_image_predictions.json") as f:
    preds = json.load(f)

# models
models = list(preds.keys())
gt = preds[models[0]]['gt']
img_ids = preds[models[0]]['img_id']

# create a dataframe
df = pd.DataFrame({'img_id': img_ids, 'gt': gt})

for m in models:
    df[f'{m}_y0'] = preds[m]['y0']
    df[f'{m}_y*'] = preds[m]['ystar']
    df[f'{m}_err*'] = preds[m]['err_star']

# filter to extreme dense images (gt > 1000)
extreme_df = df[df['gt'] > 1000].sort_values(by='gt', ascending=False)

print("=" * 130)
print(f"{'Image':<12} | {'GT':<6} | {'M1 (m04) y*':<12} | {'M2 (RepDW) y*':<14} | {'M3 (+Def) y*':<12} | {'M4 (TrkB) y*':<12} | {'Best Model'}")
print("=" * 130)

for _, row in extreme_df.iterrows():
    img = row['img_id']
    g = int(row['gt'])
    m1_val = row['M1 (m04 Anchor)_y*']
    m2_val = row['M2 (RepDW Neck)_y*']
    m3_val = row['M3 (+Deficit Seed)_y*']
    m4_val = row['M4 (Track B W64)_y*']
    
    errs = {
        'M1': abs(m1_val - g),
        'M2': abs(m2_val - g),
        'M3': abs(m3_val - g),
        'M4': abs(m4_val - g)
    }
    best = min(errs, key=errs.get)
    
    print(f"{img:<12} | {g:<6} | {m1_val:<12.1f} | {m2_val:<14.1f} | {m3_val:<12.1f} | {m4_val:<12.1f} | {best} (err={errs[best]:.1f})")

print("=" * 130)
print(f"Mean Extreme Dense MAE (15 images):")
for m in models:
    sub_err = np.mean([abs(preds[m]['ystar'][i] - gt[i]) for i in range(len(gt)) if gt[i] > 1000])
    print(f"  {m:<25}: {sub_err:.2f}")
