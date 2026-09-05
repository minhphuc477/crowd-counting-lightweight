from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--train-out", required=True, type=Path)
    ap.add_argument("--val-out", required=True, type=Path)
    ap.add_argument("--val-count", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    lines = [x for x in args.manifest.read_text().splitlines() if x.strip()]
    idx = list(range(len(lines)))
    random.Random(args.seed).shuffle(idx)
    n_val = args.val_count if args.val_count is not None else max(1, round(len(lines) * args.val_fraction))
    val_idx = set(idx[:n_val])
    train = [line for i, line in enumerate(lines) if i not in val_idx]
    val = [line for i, line in enumerate(lines) if i in val_idx]
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_out.write_text("\n".join(train) + "\n")
    args.val_out.write_text("\n".join(val) + "\n")
    print(f"total={len(lines)} train={len(train)} val={len(val)} seed={args.seed}")


if __name__ == "__main__":
    main()
