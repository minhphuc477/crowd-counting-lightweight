import glob
import json
import os
import torch

run_dirs = sorted(glob.glob('runs/sha_a/rmr_v70/v70_track_b_*'))
print(f"Found {len(run_dirs)} V70 Track B experiment runs:\n")

summary_table = []

for r in run_dirs:
    run_name = os.path.basename(r)
    print("=" * 80)
    print(f"RUN: {run_name}")
    print("=" * 80)
    
    # 1. Inspect summary.json if available
    sum_p = os.path.join(r, 'eval_val', 'summary.json')
    eval_metrics = {}
    if os.path.exists(sum_p):
        with open(sum_p) as f:
            eval_metrics = json.load(f)
        mae = eval_metrics.get('MAE') or eval_metrics.get('val_mae')
        rmse = eval_metrics.get('RMSE') or eval_metrics.get('val_rmse')
        game0 = eval_metrics.get('GAME0')
        game1 = eval_metrics.get('GAME1')
        game2 = eval_metrics.get('GAME2')
        game3 = eval_metrics.get('GAME3')
        nae = eval_metrics.get('NAE')
        bias = eval_metrics.get('Bias')
        print(f"  [eval_val/summary.json]: MAE={mae:.2f}, RMSE={rmse:.2f}, GAME0={game0:.2f}, GAME1={game1:.2f}, GAME2={game2:.2f}, GAME3={game3:.2f}, NAE={nae:.4f}, Bias={bias:.2f}")
    
    # 2. Inspect best_val_mae.pt / best_model.pt directly
    for cand in ['best_val_mae.pt', 'best_model.pt', 'last.pt']:
        pt_path = os.path.join(r, cand)
        if os.path.exists(pt_path):
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
            ep = ckpt.get('epoch', 'N/A')
            best_m = ckpt.get('best_mae', 'N/A')
            git_c = str(ckpt.get('git_commit', ckpt.get('provenance', {}).get('git_commit', 'N/A')))
            cfg = ckpt.get('config', {})
            m_cfg = cfg.get('model', {})
            l_cfg = cfg.get('loss', {})
            t_cfg = cfg.get('train', {})
            
            # Count trainable parameters from model state dict
            model_sd = ckpt.get('model', {})
            n_params = sum(v.numel() for v in model_sd.values() if isinstance(v, torch.Tensor) and v.is_floating_point())
            has_ema = 'ema_model' in ckpt
            
            print(f"  [{cand}]: Saved Epoch={ep}, best_mae={best_m}, git_commit={git_c[:8]}")
            print(f"    Trainable Params: {n_params:,} | EMA Shadow Present: {has_ema}")
            print(f"    Model Config: stats={m_cfg.get('regional_feature_stats')}, neck={m_cfg.get('neck_type')}, solver={m_cfg.get('solver_mode')}, enable_solver={m_cfg.get('enable_solver')}")
            print(f"    Loss Config: type={l_cfg.get('allocation_loss_type')}, lam_rot={l_cfg.get('lambda_radon_ot')}, lam_dm16={l_cfg.get('lambda_flat_dm16')}, lam_bay={l_cfg.get('lambda_bayesian')}, dm_target={l_cfg.get('dm_target')}")
            print(f"    Train Config: lr={t_cfg.get('lr')}, epochs={t_cfg.get('epochs')}, solver_warmup={t_cfg.get('solver_warmup_epochs')}, solver_ramp={t_cfg.get('solver_ramp_epochs')}")
            
            summary_table.append({
                'run': run_name,
                'mae': mae if eval_metrics else best_m,
                'rmse': eval_metrics.get('RMSE', 'N/A'),
                'game0': eval_metrics.get('GAME0', 'N/A'),
                'game1': eval_metrics.get('GAME1', 'N/A'),
                'game2': eval_metrics.get('GAME2', 'N/A'),
                'game3': eval_metrics.get('GAME3', 'N/A'),
                'nae': eval_metrics.get('NAE', 'N/A'),
                'epoch': ep,
                'params': n_params,
                'stats': m_cfg.get('regional_feature_stats'),
                'neck': m_cfg.get('neck_type'),
                'solver': m_cfg.get('solver_mode') if m_cfg.get('enable_solver', True) else 'direct_no_solver',
                'loss_type': l_cfg.get('allocation_loss_type'),
            })
            break

print("\n" + "=" * 90)
print(f"{'Run':35s} | {'MAE':6s} | {'RMSE':6s} | {'GAME0':6s} | {'NAE':6s} | {'Epoch':5s} | {'Params':8s} | {'Loss Type':22s}")
print("-" * 90)
for row in summary_table:
    mae_str = f"{row['mae']:.2f}" if isinstance(row['mae'], (int, float)) else str(row['mae'])
    rmse_str = f"{row['rmse']:.2f}" if isinstance(row['rmse'], (int, float)) else str(row['rmse'])
    g0_str = f"{row['game0']:.2f}" if isinstance(row['game0'], (int, float)) else str(row['game0'])
    nae_str = f"{row['nae']:.4f}" if isinstance(row['nae'], (int, float)) else str(row['nae'])
    print(f"{row['run']:35s} | {mae_str:6s} | {rmse_str:6s} | {g0_str:6s} | {nae_str:6s} | {str(row['epoch']):5s} | {row['params']:8,d} | {str(row['loss_type']):22s}")
print("=" * 90)
