import json
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# Check if per-image predictions or logs exist in eval_val
for name, p_str in [
    ("Step0", "runs/sha_a/rmr_v31_step0_v19_anchor"),
    ("H1", "runs/sha_a/rmr_v31_h1_stride2_area_norm"),
    ("H2", "runs/sha_a/rmr_v31_h2_density_gated_anscombe"),
    ("H3", "runs/sha_a/rmr_v31_h3_anisotropic_sirt"),
    ("H4", "runs/sha_a/rmr_v31_h4_composite_sub50"),
]:
    p = Path(p_str) / "eval_val" / "summary.json"
    with open(p, "r") as f:
        data = json.load(f)
    print(f"\n{'='*60}\n{name} Detailed Summary:")
    print(f"  MAE: {data.get('MAE'):.2f} | RMSE: {data.get('RMSE'):.2f} | Bias: {data.get('Bias'):.2f}")
    print(f"  Sparse: {data.get('mae_sparse'):.2f} | Moderate: {data.get('mae_moderate'):.2f} | Dense: {data.get('mae_dense'):.2f}")
    print(f"  MedianAE: {data.get('MedianAE'):.2f} | P90AE: {data.get('P90AE'):.2f} | P95AE: {data.get('P95AE'):.2f} | MaxAE: {data.get('MaxAE'):.2f}")
