import re
from pathlib import Path

doc_path = Path("AGENT_02_RMR_IMPLEMENTATION_EXPERIMENT_MASTER.md")
text = doc_path.read_text(encoding="utf-8")

pattern = re.compile(r"## File:\s*`([^`]+)`\s*\n\s*```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)
matches = pattern.findall(text)
print(f"Found {len(matches)} files in AGENT_02")

base_dir = Path("rmr_count_reference_code/rmr_count_reference")

for rel_path, content in matches:
    target = base_dir / rel_path.strip()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"Wrote {target} ({len(content)} chars)")

print("Unpack complete!")
