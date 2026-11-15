"""Inspect and un-trash RMR-Net items in Zotero."""

import sqlite3
import subprocess
import time
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")

def inspect():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT itemID, dateDeleted FROM deletedItems")
    deleted = cur.fetchall()
    print(f"Total items in deletedItems: {len(deleted)}")
    rmr_deleted = [d for d in deleted if d[0] in range(390, 412)]
    print(f"RMR items in deletedItems: {len(rmr_deleted)}")
    for d in rmr_deleted:
        print(f" - Item {d[0]} deleted at {d[1]}")
    conn.close()

if __name__ == "__main__":
    inspect()
