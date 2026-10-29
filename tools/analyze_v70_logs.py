import glob
import os
import pandas as pd

logs = sorted(glob.glob('runs/sha_a/rmr_v70/v70_track_b_*/train_log.csv'))
for l in logs:
    run_name = os.path.basename(os.path.dirname(l))
    df = pd.read_csv(l)
    eval_df = df[df['val_mae'].notna()]
    best_row = eval_df.loc[eval_df['val_mae'].idxmin()]
    last_row = eval_df.iloc[-1]
    print("=" * 80)
    print(f"{run_name}: Total Epochs logged: {len(df)}")
    print(f"  BEST: Epoch {int(best_row['epoch'])}: MAE={best_row['val_mae']:.2f}, RMSE={best_row['val_rmse']:.2f}, NAE={best_row['val_nae']:.4f}, Bias={best_row['val_bias']:.2f}")
    print(f"  LAST: Epoch {int(last_row['epoch'])}: MAE={last_row['val_mae']:.2f}, RMSE={last_row['val_rmse']:.2f}, NAE={last_row['val_nae']:.4f}, Bias={last_row['val_bias']:.2f}")
    
    # Check density bin breakdown at best epoch
    sparse = best_row.get('val_mae_sparse', 'N/A')
    mod = best_row.get('val_mae_moderate', 'N/A')
    dense = best_row.get('val_mae_dense', 'N/A')
    print(f"  Density Breakdown @ Best: Sparse={sparse:.2f}, Moderate={mod:.2f}, Dense={dense:.2f}")
    
    # Check carrier vs solver difference if logged
    y0_mae = best_row.get('mae_reg_y0', 'N/A')
    y1_mae = best_row.get('mae_reg_y1', 'N/A')
    if isinstance(y0_mae, (int, float)):
        print(f"  Regional Carrier y0 MAE: {y0_mae:.2f} vs Solver y MAE: {y1_mae:.2f}")
