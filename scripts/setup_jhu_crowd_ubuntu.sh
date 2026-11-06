#!/usr/bin/env bash
# ==============================================================================
# Canonical Setup Script for JHU-CROWD++ Dataset on Ubuntu Server
# ==============================================================================
set -euo pipefail

echo "===================================================================="
echo " JHU-CROWD++ Automated Setup & Ingestion Pipeline"
echo "===================================================================="

DATA_DIR="data/jhu_crowd_v2.0"
ZIP_PATH="data/jhu_crowd_v2.0.zip"

mkdir -p data

GOOGLE_DRIVE_ID="1pA7ZeXU3hh-1txS9lFQiCek1ts3MdBaj"

if [ -d "$DATA_DIR/train/images" ]; then
    echo "[OK] JHU-CROWD++ already extracted in $DATA_DIR"
elif [ -f "$ZIP_PATH" ]; then
    echo "[INFO] Extracting $ZIP_PATH..."
    unzip -q "$ZIP_PATH" -d data/
else
    echo "[INFO] Downloading JHU-CROWD++ (~2.87 GB) via gdown..."
    pip install -q gdown
    gdown "https://drive.google.com/uc?id=${GOOGLE_DRIVE_ID}" -O "$ZIP_PATH"
    echo "[INFO] Extracting $ZIP_PATH..."
    unzip -q "$ZIP_PATH" -d data/
fi

echo "[INFO] Generating canonical manifests (train: 2272, val: 500, test: 1600)..."
python scripts/create_jhu_crowd_manifests.py --data-dir "$DATA_DIR" --out-dir data

echo ""
echo "===================================================================="
echo " [SUCCESS] JHU-CROWD++ Manifests Generated Successfully!"
echo "===================================================================="
echo "To launch 104k benchmark training on JHU-CROWD++:"
echo "  # 1. From-Scratch 104k (0% ImageNet, lr=5e-4):"
echo "  python -m rmr_v3.train --config configs/benchmarks/exp_jhu_direct_scratch_104k.yaml"
echo ""
echo "  # 2. Pre-trained 104k (ImageNet init, lr=1e-4):"
echo "  python -m rmr_v3.train --config configs/benchmarks/exp_jhu_direct_pretrained_104k.yaml"
echo "===================================================================="
