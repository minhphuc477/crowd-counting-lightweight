"""Inspect attachments in collection 4."""

import sqlite3
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")

def main():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    cur.execute("""
        SELECT ia.itemID, ia.parentItemID, ia.contentType, ia.path
        FROM itemAttachments ia
        JOIN collectionItems ci ON ia.parentItemID = ci.itemID
        WHERE ci.collectionID = 4
        LIMIT 10
    """)
    atts = cur.fetchall()
    print(f"Sample attachments in collection 4: {len(atts)}")
    for a in atts:
        print(a)
    conn.close()

if __name__ == "__main__":
    main()
