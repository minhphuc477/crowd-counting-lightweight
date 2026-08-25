import os
import sys
import urllib.request
import zipfile
import shutil


def download_file(url: str, dest_path: str):
    """Download a file with a progress bar."""
    print(f"Downloading from: {url}")
    print(f"Destination: {dest_path}")
    
    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloaded: {mb:.1f} MB")
        sys.stdout.flush()
        
    urllib.request.urlretrieve(url, dest_path, reporthook=report)
    print("\nDownload complete!")


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip archive."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


if __name__ == "__main__":
    # Standard public mirrors for ShanghaiTech
    # 1. Dropbox direct link (dl=1) or Kaggle/Academic mirror
    print("ShanghaiTech Dataset Downloader")
