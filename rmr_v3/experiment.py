from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rmr_v3.train import main

if __name__ == "__main__":
    main()
