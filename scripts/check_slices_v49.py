import json
import os

runs = [
    'v49_m1_m04_anchor',
    'v49_m2_repdw_neck_on_m04',
    'v49_m3_repdw_deficit_seeding',
    'v49_m4_track_b_nextgen_w64'
]

print("=" * 110)
print(f"{'Run':<30} | {'MAE':<8} | {'RMSE':<8} | {'Sparse':<10} | {'Moderate':<10} | {'Dense':<10} | {'Help Rate':<10}")
print("-" * 110)

for r in runs:
    p = os.path.join('runs', 'sha_a', r, 'eval_val', 'summary.json')
    if not os.path.exists(p):
        continue
    with open(p) as f:
        s = json.load(f)
    mae = s.get('mae', 0.0)
    rmse = s.get('rmse', 0.0)
    sp = s.get('mae_sparse', 0.0)
    mod = s.get('mae_moderate', 0.0)
    dense = s.get('mae_dense', 0.0)
    help_rate = s.get('solver_help_fraction', 0.0) * 100.0
    print(f"{r:<30} | {mae:<8.2f} | {rmse:<8.2f} | {sp:<10.2f} | {mod:<10.2f} | {dense:<10.2f} | {help_rate:<9.1f}%")

print("=" * 110)
