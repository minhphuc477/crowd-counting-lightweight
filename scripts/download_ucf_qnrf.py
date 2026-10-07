"""Download UCF-QNRF dataset archive (4.24 GB) with persistent resume and auto-retry."""
from __future__ import annotations

import os
import ssl
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

EXPECTED_SIZE = 4552702871  # Exact bytes from CRCV Content-Range header


def download_qnrf(chunk_size: int = 4 * 1024 * 1024) -> Path:
    data_dir = Path("data")
    dest_path = data_dir / "archives" / "UCF-QNRF_ECCV18.zip"
    part_path = data_dir / "archives" / "UCF-QNRF_ECCV18.zip.part"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If dest_path exists but is incomplete, rename back to part_path
    if dest_path.exists():
        if dest_path.stat().st_size == EXPECTED_SIZE and zipfile.is_zipfile(dest_path):
            print(f"UCF-QNRF fully downloaded and verified: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
            return dest_path
        else:
            print(f"Incomplete zip file found ({dest_path.stat().st_size} bytes). Renaming to .zip.part to resume...")
            if part_path.exists():
                part_path.unlink()
            dest_path.rename(part_path)

    url = "https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    while True:
        downloaded = part_path.stat().st_size if part_path.exists() else 0
        if downloaded >= EXPECTED_SIZE:
            print("Download reached expected size!")
            break

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Range": f"bytes={downloaded}-",
        }
        print(f"Resuming download from byte {downloaded} ({downloaded / (1024*1024):.1f}/{EXPECTED_SIZE / (1024*1024):.1f} MB)...")

        req = urllib.request.Request(url, headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                with open(part_path, "ab") as f:
                    last_print = 0.0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_print > 1.0:
                            last_print = now
                            mb = downloaded / (1024 * 1024)
                            tot_mb = EXPECTED_SIZE / (1024 * 1024)
                            pct = 100.0 * downloaded / EXPECTED_SIZE
                            elapsed = max(now - t0, 0.001)
                            speed = (downloaded - (part_path.stat().st_size - len(chunk))) / (1024 * 1024 * elapsed)
                            sys.stdout.write(f"\r  QNRF: {pct:5.1f}% [{mb:6.1f}/{tot_mb:6.1f} MB] speed: {speed:5.1f} MB/s")
                            sys.stdout.flush()
            print()
        except Exception as e:
            print(f"\nConnection closed or error: {e}. Retrying in 3 seconds...")
            time.sleep(3)

    if part_path.exists():
        time.sleep(1.5)
        for attempt in range(10):
            try:
                part_path.rename(dest_path)
                break
            except PermissionError:
                time.sleep(1.0)

    if not dest_path.exists():
        raise FileNotFoundError(f"Failed to rename {part_path} to {dest_path}")

    mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"Successfully downloaded and verified UCF-QNRF: {mb:.2f} MB")
    return dest_path


if __name__ == "__main__":
    download_qnrf()
