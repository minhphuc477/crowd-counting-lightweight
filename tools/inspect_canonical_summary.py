import json
from pathlib import Path
import yaml

runs_dir = Path("runs/canonical")
abl_dir = runs_dir / "ablations"

all_runs = sorted(list(runs_dir.glob("*")) + list(abl_dir.glob("*")))

print("=" * 160)
print(f"{'Run / Minimal Architecture Variant':<42} | {'Params':>7} | {'MAE':>6} | {'RMSE':>6} | {'Bias':>7} | {'Sparse':>6} | {'Mod':>6} | {'Dense':>6} | {'G0':>6} | {'G1':>6} | {'G2':>6} | {'G3':>6}")
print("-" * 160)

for r in all_runs:
    if not r.is_dir() or r.name == "ablations":
        continue
    
    raw_summary = {}
    for p in [r / "eval_val" / "summary.json", r / "summary.json"]:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw_summary = json.load(f)
                break
            except Exception:
                pass

    cfg = {}
    cfg_p = r / "resolved_config.yaml"
    if cfg_p.is_file():
        try:
            with open(cfg_p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass

    mae = raw_summary.get("mae") or raw_summary.get("MAE")
    rmse = raw_summary.get("rmse") or raw_summary.get("RMSE")
    bias = raw_summary.get("bias") or raw_summary.get("Bias")
    sparse = raw_summary.get("mae_sparse")
    mod = raw_summary.get("mae_moderate")
    dense = raw_summary.get("mae_dense")
    g0 = raw_summary.get("GAME0")
    g1 = raw_summary.get("GAME1")
    g2 = raw_summary.get("GAME2")
    g3 = raw_summary.get("GAME3")

    m_cfg = cfg.get("model", {})
    # estimate params or get from log
    params = m_cfg.get("max_trainable_params", "-")

    m_str = f"{mae:.2f}" if mae is not None else "-"
    rm_str = f"{rmse:.2f}" if rmse is not None else "-"
    b_str = f"{bias:+.2f}" if bias is not None else "-"
    sp_str = f"{sparse:.2f}" if sparse is not None else "-"
    mo_str = f"{mod:.2f}" if mod is not None else "-"
    de_str = f"{dense:.2f}" if dense is not None else "-"
    g0_str = f"{g0:.1f}" if g0 is not None else "-"
    g1_str = f"{g1:.1f}" if g1 is not None else "-"
    g2_str = f"{g2:.1f}" if g2 is not None else "-"
    g3_str = f"{g3:.1f}" if g3 is not None else "-"

    name = r.relative_to(runs_dir).as_posix()
    print(f"{name:<42} | {str(params):>7} | {m_str:>6} | {rm_str:>6} | {b_str:>7} | {sp_str:>6} | {mo_str:>6} | {de_str:>6} | {g0_str:>6} | {g1_str:>6} | {g2_str:>6} | {g3_str:>6}")

print("=" * 160)
