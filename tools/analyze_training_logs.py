import pandas as pd
import glob
from pathlib import Path

runs = sorted(glob.glob("runs/sha_a/matrix/*"))
header = f"{'Run':<30} | {'Best Ep':>7} | {'Val MAE':>7} | {'RMSE':>7} | {'Sparse':>6} | {'Mod':>6} | {'Dense':>7} | {'Bias':>7}"
print(header)
print("-" * len(header))
for r in runs:
    p = Path(r)
    csv_p = p / "train_log.csv"
    if not csv_p.is_file():
        continue
    df = pd.read_csv(csv_p)
    best_idx = df["val_mae"].idxmin()
    row = df.loc[best_idx]
    ep = int(row["epoch"])
    mae = row["val_mae"]
    rmse = row["val_rmse"]
    sparse = row["val_mae_sparse"]
    mod = row["val_mae_moderate"]
    dense = row["val_mae_dense"]
    bias = row["val_bias"]
    print(f"{p.name:<30} | {ep:>7} | {mae:>7.2f} | {rmse:>7.2f} | {sparse:>6.2f} | {mod:>6.2f} | {dense:>7.2f} | {bias:>+7.2f}")
