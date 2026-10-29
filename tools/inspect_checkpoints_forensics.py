import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rmr_core.training import safe_torch_load

ckpts = {
    "V70_Exp4b_Champion": Path("runs/sha_a/rmr_v70/v70_track_b_exp4b_no_solver/best_val_mae.pt"),
    "V71_Exp1_Direct_4Proj_L1": Path("runs/sha_a/rmr_v71/v71_track_b_exp1_pure_radon_ot_l1_sub50/best_val_mae.pt"),
    "V71_Exp2_Direct_8Proj_L1": Path("runs/sha_a/rmr_v71/v71_track_b_exp2_radon_ot_8projections_l1/best_val_mae.pt"),
    "V71_Exp3_PCBPRO_4Proj_L1": Path("runs/sha_a/rmr_v71/v71_track_b_exp3_peak_concentrated_solver/best_val_mae.pt"),
    "V71_Exp4_PCBPRO_8Proj_L1": Path("runs/sha_a/rmr_v71/v71_track_b_exp4_full_synergy_sub50/best_val_mae.pt"),
}

print("=" * 100)
print("FORENSIC INSPECTION OF .pt CHECKPOINTS")
print("=" * 100)

for name, p in ckpts.items():
    print(f"\n--- Checkpoint: {name} ({p}) ---")
    if not p.exists():
        print("  File not found!")
        continue
    
    ckpt = safe_torch_load(p, map_location="cpu", weights_only=True)
    print(f"  Keys in checkpoint: {list(ckpt.keys())}")
    print(f"  Saved Epoch: {ckpt.get('epoch')}")
    print(f"  Saved Val MAE: {ckpt.get('val_mae')}")
    print(f"  Saved Val RMSE: {ckpt.get('val_rmse')}")
    
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("train", {})
    
    print(f"  Config Model:")
    print(f"    output_stride: {model_cfg.get('output_stride')}, subpixel_stride2: {model_cfg.get('subpixel_stride2')}")
    print(f"    feature_width: {model_cfg.get('feature_width')}, backbone: {model_cfg.get('backbone_name')}")
    print(f"    neck_type: {model_cfg.get('neck_type')}, aspp_dilations: {model_cfg.get('aspp_dilations')}")
    print(f"    enable_solver: {model_cfg.get('enable_solver')}, solver_mode: {model_cfg.get('solver_mode')}")
    print(f"    regional_feature_stats: {model_cfg.get('regional_feature_stats')}")
    
    print(f"  Config Loss:")
    print(f"    dm_target: {loss_cfg.get('dm_target')}, alloc_loss: {loss_cfg.get('allocation_loss_type')}")
    print(f"    lambda_radon_ot: {loss_cfg.get('lambda_radon_ot')}, projections: {loss_cfg.get('radon_ot_projections')}")
    print(f"    lambda_bayesian: {loss_cfg.get('lambda_bayesian')}")
    print(f"    lambda_count: {loss_cfg.get('lambda_count')}, lambda_count_l1: {loss_cfg.get('lambda_count_l1')}")
    print(f"    lambda_region_nb: {loss_cfg.get('lambda_region_nb')}, faithful_regional_nb: {loss_cfg.get('faithful_regional_nb')}")
    
    print(f"  Best MAE in ckpt: {ckpt.get('best_mae')}")
    
    state_dict = ckpt.get("model", {})
    total_params = sum(t.numel() for t in state_dict.values())
    trainable_params = sum(t.numel() for k, t in state_dict.items() if not k.endswith("running_mean") and not k.endswith("running_var") and not k.endswith("num_batches_tracked"))
    print(f"  State Dict entries: {len(state_dict)}")
    print(f"  Total tensor elements in state_dict: {total_params:,}")
    print(f"  Estimated trainable params: {trainable_params:,}")
    
    # Check for any solver parameters
    solver_keys = [k for k in state_dict.keys() if "solver" in k]
    print(f"  Solver state_dict keys: {solver_keys if solver_keys else 'None (Pure parameter-free adjoint operator)'}")
