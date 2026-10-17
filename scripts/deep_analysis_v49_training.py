import pandas as pd
import numpy as np

runs = [
    ('v49_m1_m04_anchor', 'M1 (m04 Anchor)'),
    ('v49_m2_repdw_neck_on_m04', 'M2 (RepDW Neck)'),
    ('v49_m3_repdw_deficit_seeding', 'M3 (+Deficit Seed)'),
    ('v49_m4_track_b_nextgen_w64', 'M4 (Track B W64)')
]

dfs = {}
for r_name, label in runs:
    p = f"runs/sha_a/{r_name}/train_log.csv"
    dfs[label] = pd.read_csv(p)

print("=" * 110)
print("I. CONVERGENCE TIMELINE & BEST EPOCHS")
print("=" * 110)
print(f"{'Model':<22} | {'Best Ep':<7} | {'Min Val MAE':<11} | {'At Ep RMSE':<10} | {'Ep 1000 MAE':<11} | {'Ep 1000 RMSE':<12} | {'Overfit Delta'}")
print("-" * 110)

for label, df in dfs.items():
    best_idx = df['val_mae'].idxmin()
    best_row = df.loc[best_idx]
    best_ep = int(best_row['epoch'])
    min_mae = best_row['val_mae']
    min_rmse = best_row['val_rmse']
    
    last_row = df.iloc[-1]
    last_mae = last_row['val_mae']
    last_rmse = last_row['val_rmse']
    overfit = last_mae - min_mae
    
    print(f"{label:<22} | {best_ep:<7} | {min_mae:<11.2f} | {min_rmse:<10.2f} | {last_mae:<11.2f} | {last_rmse:<12.2f} | {overfit:+.2f}")

print("\n" + "=" * 110)
print("II. SLICE METRICS AT BEST EPOCH (TRAIN_LOG VAL)")
print("=" * 110)
print(f"{'Model':<22} | {'Best Ep':<7} | {'Overall':<8} | {'Sparse':<8} | {'Moderate':<10} | {'Dense':<8} | {'Bias':<7} | {'Solver Help'}")
print("-" * 110)

for label, df in dfs.items():
    best_idx = df['val_mae'].idxmin()
    row = df.loc[best_idx]
    ep = int(row['epoch'])
    mae = row['val_mae']
    sp = row['val_mae_sparse']
    mod = row['val_mae_moderate']
    dense = row['val_mae_dense']
    bias = row['val_bias']
    help_f = row['solver_help_fraction'] * 100
    print(f"{label:<22} | {ep:<7} | {mae:<8.2f} | {sp:<8.2f} | {mod:<10.2f} | {dense:<8.2f} | {bias:<+7.1f} | {help_f:.1f}%")

print("\n" + "=" * 110)
print("III. TRAJECTORY CHECKPOINTS (EPOCHS 200, 400, 600, 800, 1000)")
print("=" * 110)

for label, df in dfs.items():
    print(f"\n--- {label} ---")
    print(f"{'Epoch':<7} | {'Val MAE':<9} | {'Val RMSE':<9} | {'Sparse':<8} | {'Moderate':<10} | {'Dense':<8} | {'Train Total':<12} | {'Train Count':<12} | {'Help Rate'}")
    print("-" * 95)
    for target_ep in [100, 200, 400, 600, 800, 1000]:
        sub = df[df['epoch'] == target_ep]
        if not sub.empty:
            r = sub.iloc[0]
            mae = r['val_mae']
            rmse = r['val_rmse']
            sp = r['val_mae_sparse']
            mod = r['val_mae_moderate']
            dense = r['val_mae_dense']
            t_tot = r['train_total']
            t_cnt = r['train_count']
            hf = r['solver_help_fraction'] * 100
            print(f"{target_ep:<7} | {mae:<9.2f} | {rmse:<9.2f} | {sp:<8.2f} | {mod:<10.2f} | {dense:<8.2f} | {t_tot:<12.4f} | {t_cnt:<12.4f} | {hf:.1f}%")

print("\n" + "=" * 110)
print("IV. TOP-10 BEST EPOCHS FOR EACH MODEL")
print("=" * 110)
for label, df in dfs.items():
    top5 = df.sort_values(by='val_mae').head(5)
    eps = [f"Ep {int(r['epoch'])}: {r['val_mae']:.2f}" for _, r in top5.iterrows()]
    print(f"{label:<22}: " + " | ".join(eps))
