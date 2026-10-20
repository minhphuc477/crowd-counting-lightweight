import os
import yaml
import csv

target_dir = os.path.join("runs", "sha_a")
incomplete = []
finished = []

for root, dirs, files in os.walk(target_dir):
    if "train_log.csv" in files:
        csv_path = os.path.join(root, "train_log.csv")
        cfg_path = os.path.join(root, "resolved_config.yaml")
        last_pt = os.path.join(root, "last.pt")

        target_epochs = 1000
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    c = yaml.safe_load(f)
                    target_epochs = c.get("training", {}).get("epochs", c.get("train", {}).get("epochs", 1000))
            except Exception:
                pass

        lines = 0
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                lines = max(0, sum(1 for _ in r) - 1)
        except Exception:
            pass

        rel = os.path.relpath(root, target_dir)
        has_last = os.path.exists(last_pt)
        if lines < target_epochs:
            incomplete.append((rel, lines, target_epochs, has_last))
        else:
            finished.append((rel, lines, target_epochs))

print(f"Total runs found: {len(incomplete) + len(finished)}")
print(f"Finished runs: {len(finished)}")
print(f"Incomplete runs: {len(incomplete)}")
print("-" * 80)
for r, l, t, h in sorted(incomplete):
    print(f"  [INCOMPLETE] {r}: done={l}/{t} epochs | last.pt={'YES' if h else 'NO'}")
