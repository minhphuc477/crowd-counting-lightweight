"""Inspect full schema of an attachment item."""

import sqlite3
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")

def main():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    # Check item 108
    cur.execute("SELECT * FROM items WHERE itemID = 108")
    print("items row:", cur.fetchone())
    cur.execute("SELECT * FROM itemAttachments WHERE itemID = 108")
    print("itemAttachments row:", cur.fetchone())
    cur.execute("SELECT id.fieldID, f.fieldName, idv.value FROM itemData id JOIN fields f ON id.fieldID = f.fieldID JOIN itemDataValues idv ON id.valueID = idv.valueID WHERE id.itemID = 108")
    print("itemData rows:", cur.fetchall())
    conn.close()

if __name__ == "__main__":
    main()
