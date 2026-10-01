import yaml
import glob
from pathlib import Path

runs = sorted(glob.glob("runs/sha_a/matrix/*"))
configs = {}
for r in runs:
    p = Path(r)
    cfg_p = p / "resolved_config.yaml"
    if cfg_p.is_file():
        with open(cfg_p, "r", encoding="utf-8") as f:
            configs[p.name] = yaml.safe_load(f)

# Base config is m02_legacy_anchor
base_cfg = configs.get("m02_legacy_anchor", {})
m_base = base_cfg.get("model", {})
l_base = base_cfg.get("loss", {})

print(f"Total runs loaded: {len(configs)}")
print("Key differences compared to m02_legacy_anchor:")
print("=" * 80)

for name, cfg in configs.items():
    m = cfg.get("model", {})
    l = cfg.get("loss", {})
    diffs = []
    
    # Check model keys
    all_m_keys = set(m.keys()).union(m_base.keys())
    for k in sorted(all_m_keys):
        v1 = m_base.get(k)
        v2 = m.get(k)
        if v1 != v2:
            diffs.append(f"model.{k}: {v1} -> {v2}")
            
    # Check loss keys
    all_l_keys = set(l.keys()).union(l_base.keys())
    for k in sorted(all_l_keys):
        v1 = l_base.get(k)
        v2 = l.get(k)
        if v1 != v2:
            diffs.append(f"loss.{k}: {v1} -> {v2}")
            
    print(f"[{name}]")
    if not diffs:
        print("  (Identical to m02)")
    else:
        for d in diffs:
            print(f"  {d}")
    print()
