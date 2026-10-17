import json
import os
import torch
import numpy as np
import pandas as pd

runs = [
    'v49_m1_m04_anchor',
    'v49_m2_repdw_neck_on_m04',
    'v49_m3_repdw_deficit_seeding',
    'v49_m4_track_b_nextgen_w64'
]

print("=" * 100)
print(f"{'Metric':<35} | {'M1 (Anchor)':<15} | {'M2 (RepDW)':<15} | {'M3 (+Deficit)':<15} | {'M4 (Track B W64)':<15}")
print("=" * 100)

summaries = {}
for r in runs:
    p = os.path.join('runs', 'sha_a', r, 'eval_val', 'summary.json')
    if os.path.exists(p):
        with open(p) as f:
            summaries[r] = json.load(f)
    else:
        summaries[r] = {}

metrics_to_show = [
    ('best_epoch', 'Best Val Epoch'),
    ('mae', 'Overall Val MAE (y*)'),
    ('rmse', 'Overall Val RMSE (y*)'),
    ('nae', 'Overall Val NAE (y*)'),
    ('net_bias', 'Overall Net Bias (y*)'),
    ('carrier_mae', 'Carrier MAE (y0)'),
    ('carrier_rmse', 'Carrier RMSE (y0)'),
    ('carrier_nae', 'Carrier NAE (y0)'),
    ('carrier_net_bias', 'Carrier Net Bias (y0)'),
    ('solver_help_fraction', 'Solver Help Fraction'),
    ('solver_harm_fraction', 'Solver Harm Fraction'),
    ('energy_monotonic_fraction', 'Monotonic Energy Frac'),
]

for key, label in metrics_to_show:
    vals = []
    for r in runs:
        s = summaries[r]
        v = s.get(key, None)
        if v is None:
            vals.append("N/A")
        elif isinstance(v, float):
            vals.append(f"{v:10.2f}")
        else:
            vals.append(f"{v!s:>10}")
    print(f"{label:<35} | {vals[0]:<15} | {vals[1]:<15} | {vals[2]:<15} | {vals[3]:<15}")

print("-" * 100)
# Also check if there are slice metrics in summary
s0 = summaries[runs[0]]
slice_keys = [k for k in s0.keys() if 'sparse' in k or 'dense' in k or 'mod' in k or 'slice' in k]
print(f"Slice keys found: {slice_keys}")
