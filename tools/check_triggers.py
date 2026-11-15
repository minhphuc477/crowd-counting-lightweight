"""Check triggers and foreign keys on items and deletedItems."""

import sqlite3
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")

def main():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'")
    triggers = cur.fetchall()
    print(f"Total triggers: {len(triggers)}")
    for t in triggers:
        if any(x in t[0].lower() for x in ['item', 'delete', 'trash']):
            print(f"Trigger: {t[0]}\nSQL: {t[1]}\n")
    conn.close()

if __name__ == "__main__":
    main()
