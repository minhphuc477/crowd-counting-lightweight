import json
from pathlib import Path
import yaml

runs_dir = Path("runs/sha_a")
all_runs = []

for p in sorted(runs_dir.rglob("summary.json")):
    eval_dir = p.parent
    run_dir = eval_dir.parent
    
    # Check if this is an experiment root
    cfg_file = run_dir / "resolved_config.yaml"
    ckpt_file = run_dir / "best_val_mae.pt"
    
    with open(p) as f:
        try:
            s_data = json.load(f)
        except Exception:
            continue
            
    mae = s_data.get("MAE", s_data.get("mae", float("nan")))
    rmse = s_data.get("RMSE", s_data.get("rmse", float("nan")))
    game3 = s_data.get("GAME3", s_data.get("game3", float("nan")))
    nae = s_data.get("NAE", s_data.get("nae", float("nan")))
    bias = s_data.get("Bias", s_data.get("bias", float("nan")))
    
    # Read config
    cfg = {}
    if cfg_file.exists():
        with open(cfg_file) as f:
            try:
                cfg = yaml.safe_load(f)
            except Exception:
                pass
                
    m_cfg = cfg.get("model", {})
    l_cfg = cfg.get("loss", {})
    
    all_runs.append({
        "path": str(run_dir.relative_to(runs_dir)),
        "eval_type": eval_dir.name,
        "mae": mae,
        "rmse": rmse,
        "game3": game3,
        "nae": nae,
        "bias": bias,
        "stride": m_cfg.get("output_stride"),
        "subpixel": m_cfg.get("subpixel_stride2"),
        "feature_width": m_cfg.get("feature_width"),
        "neck": m_cfg.get("neck_type"),
        "solver": m_cfg.get("solver_mode") if m_cfg.get("enable_solver") else "direct",
        "alloc_loss": l_cfg.get("allocation_loss_type"),
        "radon_proj": l_cfg.get("radon_ot_projections"),
        "l_radon": l_cfg.get("lambda_radon_ot"),
        "l_bayes": l_cfg.get("lambda_bayesian"),
        "l_count": l_cfg.get("lambda_count"),
        "l_l1": l_cfg.get("lambda_count_l1"),
        "l_reg_nb": l_cfg.get("lambda_region_nb"),
    })

print(f"Total evaluated runs found: {len(all_runs)}")
print("=" * 130)
print(f"{'Run Path':<45} | {'Eval':<10} | {'MAE':<7} | {'RMSE':<7} | {'GAME3':<7} | {'Bias':<7} | {'Stride':<6} | {'Solver':<12} | {'Alloc Loss':<15}")
print("-" * 130)

# Sort by MAE
valid_runs = [r for r in all_runs if not (r['mae'] != r['mae'])] # filter out nan
for r in sorted(valid_runs, key=lambda x: x["mae"]):
    solver_str = str(r['solver'])[:12]
    alloc_str = str(r['alloc_loss'])[:15]
    print(f"{r['path'][:45]:<45} | {r['eval_type'][:10]:<10} | {r['mae']:<7.2f} | {r['rmse']:<7.2f} | {r['game3']:<7.2f} | {r['bias']:<+7.2f} | {str(r['stride']):<6} | {solver_str:<12} | {alloc_str:<15}")
