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
    if command -v unzip >/dev/null 2>&1; then
        unzip -q "$ZIP_PATH" -d data/
    else
        python3 -m zipfile -e "$ZIP_PATH" data/
    fi
else
    echo "[INFO] Downloading JHU-CROWD++ (~3.08 GB) via gdown..."
    pip install -q gdown
    gdown "https://drive.google.com/uc?id=${GOOGLE_DRIVE_ID}" -O "$ZIP_PATH"
    echo "[INFO] Extracting $ZIP_PATH..."
    if command -v unzip >/dev/null 2>&1; then
        unzip -q "$ZIP_PATH" -d data/
    else
        python3 -m zipfile -e "$ZIP_PATH" data/
    fi
fi

echo "[INFO] Generating canonical manifests (train: 2272, val: 500, test: 1600)..."
python scripts/create_jhu_crowd_manifests.py --data-dir "$DATA_DIR" --out-dir data

CLEAN_ZIP="${CLEAN_ZIP:-0}"
if [[ "${1:-}" == "--clean-zip" || "$CLEAN_ZIP" == "1" ]]; then
    if [ -f "$ZIP_PATH" ]; then
        echo "[INFO] Cleaning up $ZIP_PATH to free ~3.08 GB disk space..."
        rm -f "$ZIP_PATH"
        echo "[OK] Deleted $ZIP_PATH successfully."
    fi
elif [ -f "$ZIP_PATH" ]; then
    echo ""
    echo "[TIP] To delete the zip archive and free ~3.08 GB disk space, run:"
    echo "  rm -f $ZIP_PATH"
fi

echo ""
echo "===================================================================="
echo " [SUCCESS] JHU-CROWD++ Manifests Generated Successfully!"
echo "===================================================================="
echo "To launch 104k benchmark training on JHU-CROWD++:"
echo "  # 1. From-Scratch Flagship 104k (0% ImageNet, lr=5e-4):"
echo "  python -m rmr_v3.train --config configs/benchmarks/exp_jhu_direct_scratch_104k.yaml"
echo ""
echo "  # 2. Pre-trained Flagship 104k (ImageNet init, lr=1e-4):"
echo "  python -m rmr_v3.train --config configs/benchmarks/exp_jhu_direct_pretrained_104k.yaml"
echo ""
echo "  # 3. Systematic Ablation Suite (15 configs in configs/benchmarks/ablations_jhu/):"
echo "  # e.g., Pure Bayesian:"
echo "  python -m rmr_v3.train --config configs/benchmarks/ablations_jhu/exp_jhu_abl_a2_pure_bayesian_104k.yaml"
echo "  # e.g., Pure Radon-OT:"
echo "  python -m rmr_v3.train --config configs/benchmarks/ablations_jhu/exp_jhu_abl_a3_pure_radon_ot_104k.yaml"
echo ""
echo "  # 4. Fine-Grained Attribute & Environmental Robustness Evaluator (Weather, Distractor, Density):"
echo "  python scripts/eval_jhu_attributes.py --checkpoint runs/benchmarks/exp_jhu_direct_scratch_104k/best_val_mae.pt"
echo "===================================================================="
