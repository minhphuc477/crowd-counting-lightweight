import json
import numpy as np
import pandas as pd

with open("runs/v49_per_image_predictions.json") as f:
    preds = json.load(f)

m1_pred = np.array(preds['M1 (m04 Anchor)']['ystar'])
m2_pred = np.array(preds['M2 (RepDW Neck)']['ystar'])
m3_pred = np.array(preds['M3 (+Deficit Seed)']['ystar'])
m4_pred = np.array(preds['M4 (Track B W64)']['ystar'])
gt = np.array(preds['M1 (m04 Anchor)']['gt'])

# 1. Simple Average Ensembles
ens_m1_m4 = (m1_pred + m4_pred) / 2.0
ens_m1_m2 = (m1_pred + m2_pred) / 2.0
ens_all = (m1_pred + m2_pred + m3_pred + m4_pred) / 4.0

def metrics(p, g):
    err = np.abs(p - g)
    rmse = np.sqrt(np.mean((p - g)**2))
    sp = np.mean(err[g <= 100])
    mod = np.mean(err[(g > 100) & (g <= 500)])
    dense = np.mean(err[g > 500])
    extr = np.mean(err[g > 1000])
    return np.mean(err), rmse, sp, mod, dense, extr

print("=" * 95)
print(f"{'Method':<30} | {'MAE':<7} | {'RMSE':<7} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7} | {'Extr>1k':<7}")
print("=" * 95)

for name, p in [
    ("M1 (m04 Anchor)", m1_pred),
    ("M2 (RepDW Neck)", m2_pred),
    ("M4 (Track B NextGen)", m4_pred),
    ("Ensemble [M1 + M4] (50/50)", ens_m1_m4),
    ("Ensemble [M1 + M2] (50/50)", ens_m1_m2),
    ("Ensemble [All 4 Models]", ens_all),
]:
    m, r, sp, md, dn, ex = metrics(p, gt)
    print(f"{name:<30} | {m:<7.2f} | {r:<7.2f} | {sp:<7.2f} | {md:<7.2f} | {dn:<7.2f} | {ex:<7.2f}")

# Density-Gated Ensemble: If M4 predicts <= 500 use M2/M4, if > 500 use M1
density_gated = np.where(m4_pred <= 100, m2_pred, np.where(m4_pred <= 500, (m1_pred + m4_pred) / 2.0, m1_pred))
m, r, sp, md, dn, ex = metrics(density_gated, gt)
print(f"{'Density-Gated [M2/M4/M1]':<30} | {m:<7.2f} | {r:<7.2f} | {sp:<7.2f} | {md:<7.2f} | {dn:<7.2f} | {ex:<7.2f}")

# Oracle (best of M1 vs M4 for each image)
oracle_1_4 = np.where(np.abs(m1_pred - gt) < np.abs(m4_pred - gt), m1_pred, m4_pred)
m, r, sp, md, dn, ex = metrics(oracle_1_4, gt)
print(f"{'Oracle [M1 vs M4]':<30} | {m:<7.2f} | {r:<7.2f} | {sp:<7.2f} | {md:<7.2f} | {dn:<7.2f} | {ex:<7.2f}")

# Oracle across all 4
all_preds = np.stack([m1_pred, m2_pred, m3_pred, m4_pred])
all_errs = np.abs(all_preds - gt[None, :])
best_idx = np.argmin(all_errs, axis=0)
oracle_all = np.take_along_axis(all_preds, best_idx[None, :], axis=0).squeeze(0)
m, r, sp, md, dn, ex = metrics(oracle_all, gt)
print(f"{'Oracle [All 4 Models]':<30} | {m:<7.2f} | {r:<7.2f} | {sp:<7.2f} | {md:<7.2f} | {dn:<7.2f} | {ex:<7.2f}")
print("=" * 95)
