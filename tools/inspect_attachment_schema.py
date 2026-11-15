"""Inspect itemAttachments schema and existing rows."""

import sqlite3
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")

def main():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(itemAttachments)")
    print("itemAttachments columns:", cur.fetchall())
    
    cur.execute("SELECT * FROM itemAttachments LIMIT 5")
    print("Sample itemAttachments:", cur.fetchall())
    conn.close()

if __name__ == "__main__":
    main()
