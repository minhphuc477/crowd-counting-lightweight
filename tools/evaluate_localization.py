"""Backward-compatible entry point for :mod:`tools.eval_localization`."""

import os
import sys

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from tools.eval_localization import main


if __name__ == "__main__":
    main()
