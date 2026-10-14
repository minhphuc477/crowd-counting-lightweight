from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def get_git_info() -> tuple[str, bool]:
    """Retrieve the current git commit hash and working tree dirty status."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        commit = "unknown"
    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty = bool(status)
    except Exception:
        dirty = False
    return commit, dirty


def compute_file_sha256(path: Path | str) -> str:
    """Compute deterministic SHA256 hex digest of a file in 64KB blocks.

    For text-based manifest/config files, line endings are normalized (\\r\\n -> \\n)
    to guarantee cross-platform deterministic hashing between Linux (LF) and Windows (CRLF).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for SHA256 computation: {p}")
    h = hashlib.sha256()
    is_text = p.suffix.lower() in (".jsonl", ".json", ".yaml", ".yml", ".txt", ".csv")
    if is_text:
        with open(p, "rb") as f:
            content = f.read().replace(b"\r\n", b"\n")
            h.update(content)
    else:
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
    return h.hexdigest()


__all__ = [
    "compute_file_sha256",
    "get_git_info",
]
