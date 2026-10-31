#!/usr/bin/env python3
"""Root evaluation entry point alias for RMR Crowd Counting Framework.

Delegates directly to eval.main() so that evaluation can be invoked via:
    python evaluate.py --checkpoint <checkpoint_path>
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rmr_v3.eval import main

if __name__ == "__main__":
    main()
