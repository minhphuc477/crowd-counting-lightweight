import csv
import os
import time
from typing import Any, Dict, List, Optional


def _retry_open(filepath: str, mode: str, max_retries: int = 5, retry_delay: float = 0.2, **kwargs):
    """Open file with retry on transient Windows file lock errors."""
    for attempt in range(max_retries):
        try:
            return open(filepath, mode, **kwargs)
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (attempt + 1))


class CSVLogger:
    """CSV logger that expands the schema instead of silently dropping new metrics."""

    def __init__(self, filepath: str, fieldnames: Optional[List[str]] = None):
        self.filepath = filepath
        self.fieldnames = list(fieldnames) if fieldnames else None
        self.file_initialized = os.path.exists(filepath) and os.path.getsize(filepath) > 0
        if self.file_initialized and self.fieldnames is None:
            with _retry_open(filepath, "r", newline="", encoding="utf-8") as f:
                self.fieldnames = next(csv.reader(f), None)

    def _expand_schema(self, row: Dict[str, Any]) -> None:
        missing = [k for k in row if k not in (self.fieldnames or [])]
        if not missing:
            return
        old_rows = []
        if self.file_initialized:
            with _retry_open(self.filepath, "r", newline="", encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f))
        self.fieldnames = list(self.fieldnames or []) + missing
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        with _retry_open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(old_rows)
        self.file_initialized = True

    def log(self, row: Dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        if not self.file_initialized:
            os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
            with _retry_open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
            self.file_initialized = True
        self._expand_schema(row)
        with _retry_open(self.filepath, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
            f.flush()

