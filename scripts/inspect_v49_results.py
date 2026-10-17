import json
import os
import pandas as pd
import numpy as np

runs = [
    'v49_m1_m04_anchor',
    'v49_m2_repdw_neck_on_m04',
    'v49_m3_repdw_deficit_seeding',
    'v49_m4_track_b_nextgen_w64'
]

print("=" * 80)
print(f"{'Run Name':<30} | {'Best Ep':<7} | {'Val MAE':<8} | {'Val RMSE':<8} | {'Eval MAE':<8} | {'Eval RMSE':<8}")
print("-" * 80)

for r in runs:
    csv_path = os.path.join('runs', 'sha_a', r, 'train_log.csv')
    sum_path = os.path.join('runs', 'sha_a', r, 'eval_val', 'summary.json')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_idx = df['val_mae'].idxmin()
        best_row = df.loc[best_idx]
        b_ep = int(best_row['epoch'])
        b_mae = best_row['val_mae']
        b_rmse = best_row['val_rmse']
    else:
        b_ep, b_mae, b_rmse = -1, float('nan'), float('nan')
        
    if os.path.exists(sum_path):
        with open(sum_path) as f:
            s = json.load(f)
        e_mae = s.get('mae', float('nan'))
        e_rmse = s.get('rmse', float('nan'))
    else:
        e_mae, e_rmse = float('nan'), float('nan')
        
    print(f"{r:<30} | {b_ep:<7} | {b_mae:<8.2f} | {b_rmse:<8.2f} | {e_mae:<8.2f} | {e_rmse:<8.2f}")

print("=" * 80)
