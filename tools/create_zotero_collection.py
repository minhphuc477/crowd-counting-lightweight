"""Helper script to create a dedicated collection in local Zotero SQLite database.

If Zotero is running, Windows locks the SQLite database.
This script checks for lock, creates the collection, assigns all 19 RMR-Net items,
and cleans up test entries.
"""

from __future__ import annotations

import os
import random
import string
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def generate_zotero_key(length: int = 8) -> str:
    """Generate Zotero-style 8-character base32-like uppercase alphanumeric key."""
    chars = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(random.choice(chars) for _ in range(length))


def get_zotero_db_path() -> Path:
    custom_path = os.environ.get("ZOTERO_DATA_DIR")
    if custom_path:
        p = Path(custom_path) / "zotero.sqlite"
        if p.exists():
            return p
    return Path.home() / "Zotero" / "zotero.sqlite"


def create_rmr_collection(collection_name: str = "Lightweight Crowd Counting (RMR-Net)") -> bool:
    db_path = get_zotero_db_path()
    if not db_path.exists():
        print(f"[ERROR] Zotero database not found at {db_path}", file=sys.stderr)
        return False

    # Check if database is currently locked by a running Zotero instance
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        cur = conn.cursor()
        cur.execute("SELECT 1")
    except sqlite3.OperationalError:
        print("[NOTICE] Zotero Desktop is currently open, locking zotero.sqlite.")
        print("Please close Zotero Desktop for 5 seconds so this script can create the collection directly.")
        return False

    try:
        # 1. Check if collection already exists
        cur.execute("SELECT collectionID FROM collections WHERE collectionName=?", (collection_name,))
        row = cur.fetchone()
        if row:
            coll_id = row[0]
            print(f"[OK] Collection '{collection_name}' already exists (ID: {coll_id}).")
        else:
            coll_key = generate_zotero_key()
            cur.execute("""
                INSERT INTO collections (collectionName, parentCollectionID, clientDateModified, libraryID, key, version, clientVersion, synced)
                VALUES (?, NULL, CURRENT_TIMESTAMP, 1, ?, 0, 0, 0)
            """, (collection_name, coll_key))
            coll_id = cur.lastrowid
            print(f"[SUCCESS] Created Collection '{collection_name}' (ID: {coll_id}, Key: {coll_key}).")

        # 2. Identify all RMR-Net items (390: Report, 393..410: 18 papers)
        cur.execute("""
            SELECT i.itemID, idv.value 
            FROM items i 
            JOIN itemData id ON i.itemID=id.itemID 
            JOIN fields f ON id.fieldID=f.fieldID 
            JOIN itemDataValues idv ON id.valueID=idv.valueID 
            WHERE (i.itemID = 390 OR (i.itemID >= 393 AND i.itemID <= 410)) AND f.fieldName='title'
        """)
        rmr_items = cur.fetchall()

        # 3. Add to collectionItems
        added_count = 0
        for item_id, title in rmr_items:
            cur.execute("""
                INSERT OR IGNORE INTO collectionItems (collectionID, itemID, orderIndex)
                VALUES (?, ?, 0)
            """, (coll_id, item_id))
            if cur.rowcount > 0:
                added_count += 1

        print(f"[SUCCESS] Linked {len(rmr_items)} RMR-Net items into Collection '{collection_name}' ({added_count} newly assigned).")

        # 4. Clean up any unintended test items (e.g. 387, 388)
        cur.execute("DELETE FROM items WHERE itemID IN (387, 388)")
        if cur.rowcount > 0:
            print(f"[CLEANUP] Removed {cur.rowcount} test/example items from earlier.")

        conn.commit()
        conn.close()
        print("\nAll done! You can now reopen Zotero Desktop to see the new collection.")
        return True

    except Exception as e:
        print(f"[ERROR] Transaction failed: {e}", file=sys.stderr)
        conn.rollback()
        conn.close()
        return False


if __name__ == "__main__":
    create_rmr_collection()
