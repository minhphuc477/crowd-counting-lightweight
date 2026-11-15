"""Inspect items 393-410 and map to PDFs in data/papers/."""

import sqlite3
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")
PAPERS_DIR = Path("F:/lightweightcrcn/data/papers")

def main():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    cur.execute("""
        SELECT i.itemID, idv.value
        FROM items i
        JOIN itemData id ON i.itemID = id.itemID
        JOIN itemDataValues idv ON id.valueID = idv.valueID
        WHERE i.itemID >= 390 AND id.fieldID = 1
        ORDER BY i.itemID
    """)
    items = cur.fetchall()
    print("Found items:")
    for it in items:
        print(f"[{it[0]}] {it[1]}")
    
    pdfs = list(PAPERS_DIR.glob("*.pdf"))
    print("\nPDFs in data/papers/:")
    for p in sorted(pdfs):
        print(f" - {p.name}")
    conn.close()

if __name__ == "__main__":
    main()
