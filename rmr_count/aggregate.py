from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    args = ap.parse_args()
    rows = [json.loads(Path(p).read_text()) for p in args.summaries]
    keys = sorted(set.intersection(*(set(r) for r in rows)))
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        if all(isinstance(v, (int, float)) for v in vals):
            a = np.asarray(vals, dtype=np.float64)
            out[k] = {
                "mean": float(a.mean()),
                "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "n": len(a),
            }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
