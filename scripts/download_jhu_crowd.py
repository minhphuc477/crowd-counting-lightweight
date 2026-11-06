#!/usr/bin/env python3
"""Automated downloader, extractor, and manifest builder for JHU-CROWD++."""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path
import gdown

def main():
    dest_zip = Path("data/jhu_crowd_v2.0.zip")
    extract_dir = Path("data/jhu_crowd_v2.0")
    
    # 1. Download if not already downloaded or extracted
    if (extract_dir / "train" / "images").exists():
        print(f"[OK] JHU-CROWD++ already extracted in {extract_dir}")
    else:
        if not dest_zip.exists():
            print("[INFO] Starting download of jhu_crowd_v2.0.zip (~2.87 GB) via Google Drive...")
            file_id = "1pA7ZeXU3hh-1txS9lFQiCek1ts3MdBaj"
            url = f"https://drive.google.com/uc?id={file_id}"
            dest_zip.parent.mkdir(parents=True, exist_ok=True)
            output = gdown.download(url, str(dest_zip), quiet=False)
            if not dest_zip.exists() or dest_zip.stat().st_size < 1000000:
                print(f"[ERROR] Download failed or file too small: {output}")
                return 1
            print(f"[OK] Download completed: {dest_zip} ({dest_zip.stat().st_size / (1024**3):.2f} GB)")
        
        # 2. Extract archive
        print(f"[INFO] Extracting {dest_zip} to data/ ...")
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall("data/")
        print(f"[OK] Extraction completed to {extract_dir}")
    
    # 3. Generate canonical manifests and compute forensics
    print("[INFO] Generating canonical manifests...")
    cmd = [
        sys.executable,
        "scripts/create_jhu_crowd_manifests.py",
        "--data-dir", str(extract_dir),
        "--out-dir", "data",
    ]
    subprocess.run(cmd, check=True)
    print("[SUCCESS] JHU-CROWD++ is completely ready for training and inspection!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
