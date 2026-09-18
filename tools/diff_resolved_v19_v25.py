import yaml
from pathlib import Path

p19 = Path("runs/sha_a/rmr_v19_canonical_isotropic/resolved_config.yaml")
p25_np = Path("runs/sha_a/rmr_v25_ablation_no_perspective/resolved_config.yaml")

with open(p19, "r") as f:
    c19 = yaml.safe_load(f)

with open(p25_np, "r") as f:
    c25 = yaml.safe_load(f)

print("="*70)
print("RESOLVED CONFIG DIFF: v19 Canonical vs v25 Ablation No Perspective")
print("="*70)

for section in ["seed", "data", "model", "loss", "train", "eval"]:
    v1 = c19.get(section)
    v2 = c25.get(section)
    if isinstance(v1, dict) and isinstance(v2, dict):
        all_k = sorted(set(v1.keys()) | set(v2.keys()))
        for k in all_k:
            val1 = v1.get(k, "<MISSING>")
            val2 = v2.get(k, "<MISSING>")
            if val1 != val2:
                print(f"[{section}] {k:<28} | v19: {str(val1):<18} | v25_np: {str(val2):<18}")
    else:
        if v1 != v2:
            print(f"{section:<32} | v19: {str(v1):<18} | v25_np: {str(v2):<18}")
