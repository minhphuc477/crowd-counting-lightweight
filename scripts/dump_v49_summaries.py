import json
import os
import glob

runs = [
    'v49_m1_m04_anchor',
    'v49_m2_repdw_neck_on_m04',
    'v49_m3_repdw_deficit_seeding',
    'v49_m4_track_b_nextgen_w64'
]

for r in runs:
    sum_path = os.path.join('runs', 'sha_a', r, 'eval_val', 'summary.json')
    if not os.path.exists(sum_path):
        continue
    with open(sum_path) as f:
        s = json.load(f)
    print(f"==================== {r} ====================")
    for k, v in s.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k}: {sub_v}")
        elif isinstance(v, list):
            print(f"  {k}: list of len {len(v)}")
        else:
            print(f"  {k}: {v}")
    print()
