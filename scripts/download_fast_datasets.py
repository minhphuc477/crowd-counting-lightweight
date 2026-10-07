"""Download fast crowd counting benchmarks: UCF_CC_50, Mall, UCSD."""
from __future__ import annotations

import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download file with progress reporting and resume check."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Already exists: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return dest_path

    print(f"Downloading {url} -> {dest_path}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")

        with open(part_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = 100.0 * downloaded / total_size
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    elapsed = max(time.time() - t0, 0.001)
                    speed = mb / elapsed
                    sys.stdout.write(f"\r  Progress: {pct:5.1f}% [{mb:6.1f}/{tot_mb:6.1f} MB] at {speed:5.1f} MB/s")
                    sys.stdout.flush()

        print()
        if part_path.exists():
            part_path.rename(dest_path)

    elapsed = max(time.time() - t0, 0.001)
    mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded {dest_path.name}: {mb:.2f} MB in {elapsed:.1f}s ({mb/elapsed:.1f} MB/s)")
    return dest_path


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        (
            "https://www.crcv.ucf.edu/data/ucf-cc-50/UCFCrowdCountingDataset_CVPR13.rar",
            data_dir / "archives" / "UCFCrowdCountingDataset_CVPR13.rar",
        ),
        (
            "https://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/mall_dataset.zip",
            data_dir / "archives" / "mall_dataset.zip",
        ),
        (
            "http://visal.cs.cityu.edu.hk/static/downloads/ucsdpeds_gt.zip",
            data_dir / "archives" / "ucsdpeds_gt.zip",
        ),
        (
            "http://visal.cs.cityu.edu.hk/static/downloads/ucsdpeds_vidf.zip",
            data_dir / "archives" / "ucsdpeds_vidf.zip",
        ),
    ]

    for url, dest in targets:
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Error downloading {url}: {e}")


if __name__ == "__main__":
    main()
