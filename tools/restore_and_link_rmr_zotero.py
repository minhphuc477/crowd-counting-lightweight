"""Restore RMR-Net items from Trash and attach local PDFs/reports in Zotero."""

from __future__ import annotations

import os
import random
import sqlite3
import subprocess
import time
from pathlib import Path

ZOTERO_DB = Path("C:/Users/MPhuc/Zotero/zotero.sqlite")
ZOTERO_EXE = Path("C:/Program Files/Zotero/zotero.exe")
PAPERS_DIR = Path("F:/lightweightcrcn/data/papers")
REPORTS_DIR = Path("F:/lightweightcrcn/docs/reports")

# Mapping of itemID to PDF filename
PAPER_PDF_MAP = {
    393: "burt1983laplacian.pdf",
    394: "wang2018hdc.pdf",
    395: "lin2017fpn.pdf",
    396: "sandler2018mobilenetv2.pdf",
    397: "yang2018denseaspp.pdf",
    398: "chen2019octconv.pdf",
    399: "liu2019cannet.pdf",
    400: "liu2019asff.pdf",
    401: "tan2020efficientdet.pdf",
    402: "touvron2021cait.pdf",
    403: "thanasutives2020msfanet.pdf",
    404: "zhang2016mcnn.pdf",
    405: "li2018csrnet.pdf",
    406: "ma2019bayesian.pdf",
    407: "wang2020dmcount.pdf",
    408: "wang2020nwpu.pdf",
    409: "liang2021fidtm.pdf",
    410: "shu2022chfl.pdf",
}

KEY_CHARS = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

def generate_key(existing_keys: set[str]) -> str:
    while True:
        k = "".join(random.choices(KEY_CHARS, k=8))
        if k not in existing_keys:
            existing_keys.add(k)
            return k

def get_or_create_value_id(cur: sqlite3.Cursor, value: str) -> int:
    cur.execute("SELECT valueID FROM itemDataValues WHERE value = ?", (value,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO itemDataValues (value) VALUES (?)", (value,))
    return cur.lastrowid

def main():
    print("[1/5] Terminating Zotero process to unlock database...")
    subprocess.run(["taskkill", "/IM", "zotero.exe", "/F"], capture_output=True)
    time.sleep(2)

    print("[2/5] Connecting to Zotero SQLite database...")
    conn = sqlite3.connect(ZOTERO_DB)
    cur = conn.cursor()

    # 1. Check and un-trash items >= 390
    cur.execute("SELECT itemID FROM deletedItems WHERE itemID >= 390")
    trashed = [r[0] for r in cur.fetchall()]
    print(f"Found {len(trashed)} items in deletedItems: {trashed}")
    if trashed:
        cur.execute("DELETE FROM deletedItems WHERE itemID >= 390")
        print(f"[SUCCESS] Removed {len(trashed)} items from deletedItems (restored from Trash)!")

    # 2. Clean up test item 411 if present
    cur.execute("DELETE FROM collectionItems WHERE itemID = 411")
    cur.execute("DELETE FROM itemData WHERE itemID = 411")
    cur.execute("DELETE FROM items WHERE itemID = 411")

    # 3. Ensure Collection 5 exists and link all 19 items
    cur.execute("SELECT collectionID FROM collections WHERE collectionID = 5")
    if not cur.fetchone():
        print("[ERROR] Collection 5 not found!")
        conn.close()
        return

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Fetch existing keys
    cur.execute("SELECT key FROM items")
    existing_keys = {r[0] for r in cur.fetchall()}

    # 4. Link 19 items to Collection 5
    target_items = [390] + list(range(393, 411))
    for order_idx, item_id in enumerate(target_items):
        cur.execute(
            "INSERT OR REPLACE INTO collectionItems (collectionID, itemID, orderIndex) VALUES (?, ?, ?)",
            (5, item_id, order_idx),
        )
        cur.execute("UPDATE items SET clientDateModified = ? WHERE itemID = ?", (now_str, item_id))
    print(f"[SUCCESS] Linked {len(target_items)} parent items into Collection 5.")

    # 5. Add linked PDF attachments for items 393-410
    print("[3/5] Attaching PDF full-text files to paper items...")
    cur.execute("SELECT MAX(itemID) FROM items")
    next_item_id = (cur.fetchone()[0] or 500) + 1

    attached_count = 0
    for parent_id, pdf_name in PAPER_PDF_MAP.items():
        pdf_path = PAPERS_DIR / pdf_name
        if not pdf_path.exists():
            print(f"[WARN] PDF not found: {pdf_path}")
            continue

        # Check if already attached
        cur.execute("SELECT itemID FROM itemAttachments WHERE parentItemID = ? AND contentType = 'application/pdf'", (parent_id,))
        if cur.fetchone():
            continue

        att_item_id = next_item_id
        next_item_id += 1
        att_key = generate_key(existing_keys)

        # 1. Insert into items
        cur.execute(
            """INSERT INTO items (itemID, itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
               VALUES (?, 3, ?, ?, ?, 1, ?, 1, 0)""",
            (att_item_id, now_str, now_str, now_str, att_key),
        )

        # 2. Insert into itemAttachments (linkMode 2 = linked file)
        abs_path = str(pdf_path.resolve())
        cur.execute(
            """INSERT INTO itemAttachments (itemID, parentItemID, linkMode, contentType, path, syncState)
               VALUES (?, ?, 2, 'application/pdf', ?, 0)""",
            (att_item_id, parent_id, abs_path),
        )

        # 3. Insert title in itemData
        val_id = get_or_create_value_id(cur, f"Full Text PDF: {pdf_name}")
        cur.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, 1, ?)", (att_item_id, val_id))

        attached_count += 1

    print(f"[SUCCESS] Created {attached_count} PDF attachments linked to papers.")

    # 6. Add Markdown and HTML attachments to Item 390 (Conversation Log / Technical Report)
    report_md = REPORTS_DIR / "RMR_NET_TECHNICAL_AUDIT_AND_CONVERSATION_LOG.md"
    report_html = REPORTS_DIR / "RMR_NET_TECHNICAL_AUDIT_AND_CONVERSATION_LOG.html"

    for r_path, mime_type, label in [
        (report_md, "text/markdown", "Audit Report (Markdown): RMR_NET_TECHNICAL_AUDIT_AND_CONVERSATION_LOG.md"),
        (report_html, "text/html", "Audit Report (HTML): RMR_NET_TECHNICAL_AUDIT_AND_CONVERSATION_LOG.html"),
    ]:
        if r_path.exists():
            cur.execute("SELECT itemID FROM itemAttachments WHERE parentItemID = 390 AND contentType = ?", (mime_type,))
            if not cur.fetchone():
                att_item_id = next_item_id
                next_item_id += 1
                att_key = generate_key(existing_keys)

                cur.execute(
                    """INSERT INTO items (itemID, itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
                       VALUES (?, 3, ?, ?, ?, 1, ?, 1, 0)""",
                    (att_item_id, now_str, now_str, now_str, att_key),
                )
                cur.execute(
                    """INSERT INTO itemAttachments (itemID, parentItemID, linkMode, contentType, path, syncState)
                       VALUES (?, 390, 2, ?, ?, 0)""",
                    (att_item_id, mime_type, str(r_path.resolve())),
                )
                val_id = get_or_create_value_id(cur, label)
                cur.execute("INSERT INTO itemData (itemID, fieldID, valueID) VALUES (?, 1, ?)", (att_item_id, val_id))
                print(f"[SUCCESS] Attached report: {label}")

    conn.commit()
    conn.close()
    print("[4/5] Database updated and committed successfully.")

    print("[5/5] Re-launching Zotero Desktop...")
    if ZOTERO_EXE.exists():
        subprocess.Popen([str(ZOTERO_EXE)])
        print("[SUCCESS] Zotero Desktop restarted!")
    else:
        print("[WARN] Zotero executable not found at standard path.")

if __name__ == "__main__":
    main()
