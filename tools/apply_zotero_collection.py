"""Gracefully restart Zotero and inject RMR-Net collection."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path

from create_zotero_collection import create_rmr_collection

def main():
    print("[1/4] Closing Zotero to release SQLite database lock...")
    subprocess.run(["taskkill", "/IM", "zotero.exe", "/F"], capture_output=True)
    time.sleep(2)

    print("[2/4] Creating dedicated collection and linking 19 RMR-Net items...")
    success = create_rmr_collection("Lightweight Crowd Counting (RMR-Net)")
    if not success:
        print("[ERROR] Failed to modify Zotero database.")
        return

    print("[3/4] Reopening Zotero Desktop application...")
    zotero_exe = Path("C:/Program Files/Zotero/zotero.exe")
    if zotero_exe.exists():
        subprocess.Popen([str(zotero_exe)])
        print("[4/4] Zotero restarted successfully!")
    else:
        print("[WARN] Zotero binary not found at default path.")

if __name__ == "__main__":
    main()
