import json
import os
import pandas as pd

runs = [
    ('v49_m1_m04_anchor', 'M1 (m04 Anchor)'),
    ('v49_m2_repdw_neck_on_m04', 'M2 (RepDW Neck)'),
    ('v49_m3_repdw_deficit_seeding', 'M3 (+Deficit Seed)'),
    ('v49_m4_track_b_nextgen_w64', 'M4 (Track B W64)')
]

rows = []

for r, name in runs:
    p = os.path.join('runs', 'sha_a', r, 'eval_val', 'summary.json')
    cfg_p = os.path.join('runs', 'sha_a', r, 'resolved_config.yaml')
    log_p = os.path.join('runs', 'sha_a', r, 'train_log.csv')
    
    with open(p) as f:
        s = json.load(f)
    
    # get best epoch from train_log.csv
    best_ep = -1
    best_val_mae = float('nan')
    if os.path.exists(log_p):
        df = pd.read_csv(log_p)
        best_idx = df['val_mae'].idxmin()
        best_ep = int(df.loc[best_idx, 'epoch'])
        best_val_mae = df.loc[best_idx, 'val_mae']
        
    rows.append({
        'Model': name,
        'Params': '104,441' if 'M1' in name else ('103,359' if 'M2' in name or 'M3' in name else '122,403'),
        'Best Ep': best_ep,
        'Val MAE': s.get('mae', 0.0),
        'RMSE': s.get('rmse', 0.0),
        'NAE': s.get('nae', 0.0),
        'Bias': s.get('bias', 0.0),
        'Sparse (<=100)': s.get('mae_sparse', 0.0),
        'Moderate (100-500)': s.get('mae_moderate', 0.0),
        'Dense (>500)': s.get('mae_dense', 0.0),
        'Help Rate (%)': s.get('solver_help_fraction', 0.0) * 100.0,
        'Energy Monotonic (%)': s.get('energy_monotonic_fraction', 0.0) * 100.0,
        'GAME0': s.get('GAME0', 0.0),
        'GAME1': s.get('GAME1', 0.0),
        'GAME2': s.get('GAME2', 0.0),
        'GAME3': s.get('GAME3', 0.0),
        'P90AE': s.get('P90AE', 0.0),
        'MaxAE': s.get('MaxAE', 0.0),
    })

res_df = pd.DataFrame(rows)
print(res_df.to_string(index=False))
