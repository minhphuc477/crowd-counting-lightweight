#!/usr/bin/env python3
"""Root training entry point for RMR Crowd Counting Framework.

Delegates directly to rmr_v3.train.main() so that training can be invoked from
the repository root via:
    python train.py --config <config_path>
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rmr_v3.train import main

if __name__ == "__main__":
    main()
