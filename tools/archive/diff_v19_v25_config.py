import yaml

with open("configs/rmr_v19/rmr_v19_canonical_isotropic.yaml", "r") as f:
    c19 = yaml.safe_load(f)

with open("configs/rmr_v25/rmr_v25_canonical.yaml", "r") as f:
    c25 = yaml.safe_load(f)

print("="*60)
print("CONFIG DIFFERENCES: v19 Canonical vs v25 Canonical")
print("="*60)

for section in ["data", "model", "loss", "train"]:
    s19 = c19.get(section, {})
    s25 = c25.get(section, {})
    all_keys = sorted(set(s19.keys()) | set(s25.keys()))
    diffs = []
    for k in all_keys:
        v19_val = s19.get(k, "<MISSING>")
        v25_val = s25.get(k, "<MISSING>")
        if v19_val != v25_val:
            diffs.append((k, v19_val, v25_val))
    if diffs:
        print(f"\n--- Section: [{section}] ---")
        for k, v1, v2 in diffs:
            print(f"  {k:<30} | v19: {str(v1):<20} | v25: {str(v2):<20}")
