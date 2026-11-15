"""Check item attachments in Zotero DB."""

import sqlite3
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")

def main():
    conn = sqlite3.connect(f"file:{ZOTERO_DB}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT ia.itemID, ia.parentItemID, ia.contentType, ia.path FROM itemAttachments ia WHERE ia.parentItemID >= 390")
    atts = cur.fetchall()
    print(f"Attachments for parentItemID >= 390: {len(atts)}")
    for a in atts:
        print(a)
    
    # Also check itemNotes
    cur.execute("SELECT inote.itemID, inote.parentItemID FROM itemNotes inote WHERE inote.parentItemID >= 390")
    notes = cur.fetchall()
    print(f"Notes for parentItemID >= 390: {len(notes)}")
    for n in notes:
        print(n)
    conn.close()

if __name__ == "__main__":
    main()
