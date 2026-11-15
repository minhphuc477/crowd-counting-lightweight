"""Inspect attachment item in items table."""

import sqlite3
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")

def main():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT * FROM items WHERE itemID = 108")
    print("items row for attachment 108:", cur.fetchall())
    cur.execute("SELECT * FROM itemData WHERE itemID = 108")
    print("itemData for attachment 108:", cur.fetchall())
    conn.close()

if __name__ == "__main__":
    main()
