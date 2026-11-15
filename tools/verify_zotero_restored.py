"""Verify restored state in Zotero DB."""

import sqlite3
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")

def main():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM deletedItems WHERE itemID >= 390")
    print("Items >= 390 in deletedItems (Trash):", cur.fetchone()[0])
    
    cur.execute("SELECT COUNT(*) FROM collectionItems WHERE collectionID = 5")
    print("Items in Collection 5 (RMR-Net):", cur.fetchone()[0])
    
    cur.execute("SELECT COUNT(*) FROM itemAttachments WHERE parentItemID >= 390")
    print("Attachments for items >= 390:", cur.fetchone()[0])
    
    cur.execute("""
        SELECT ci.itemID, idv.value, 
               (SELECT COUNT(*) FROM itemAttachments ia WHERE ia.parentItemID = ci.itemID) as att_count,
               (SELECT COUNT(*) FROM itemNotes inote WHERE inote.parentItemID = ci.itemID) as note_count
        FROM collectionItems ci
        JOIN itemData id ON ci.itemID = id.itemID
        JOIN itemDataValues idv ON id.valueID = idv.valueID
        WHERE ci.collectionID = 5 AND id.fieldID = 1
        ORDER BY ci.orderIndex
    """)
    rows = cur.fetchall()
    print("\nVerified Collection 5 items:")
    for r in rows:
        print(f"[{r[0]}] {r[1]} -> {r[2]} attachments, {r[3]} notes")
    conn.close()

if __name__ == "__main__":
    main()
