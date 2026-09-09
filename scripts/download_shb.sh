#!/usr/bin/env bash
# ==============================================================================
# download_shb.sh: ShanghaiTech Part B Dataset Verification & Download
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DATA_DIR="${ROOT_DIR}/data"
SHB_DIR="${DATA_DIR}/part_B_final"
TRAIN_IMG="${SHB_DIR}/train_data/images"
TRAIN_GT="${SHB_DIR}/train_data/ground_truth"
TEST_IMG="${SHB_DIR}/test_data/images"
TEST_GT="${SHB_DIR}/test_data/ground_truth"

echo "======================================================================"
echo " ShanghaiTech Part B (SHB) Verification & Setup"
echo " Working directory: ${ROOT_DIR}"
echo "======================================================================"

check_shb_complete() {
    if [ -d "${TRAIN_IMG}" ] && [ -d "${TRAIN_GT}" ] && [ -d "${TEST_IMG}" ] && [ -d "${TEST_GT}" ]; then
        local n_train_img=$(find "${TRAIN_IMG}" -maxdepth 1 -name "*.jpg" | wc -l)
        local n_train_gt=$(find "${TRAIN_GT}" -maxdepth 1 -name "*.mat" | wc -l)
        local n_test_img=$(find "${TEST_IMG}" -maxdepth 1 -name "*.jpg" | wc -l)
        local n_test_gt=$(find "${TEST_GT}" -maxdepth 1 -name "*.mat" | wc -l)
        if [ "${n_train_img}" -eq 400 ] && [ "${n_train_gt}" -eq 400 ] && \
           [ "${n_test_img}" -eq 316 ] && [ "${n_test_gt}" -eq 316 ]; then
            return 0
        fi
    fi
    return 1
}

if check_shb_complete; then
    echo "[OK] ShanghaiTech Part B is already present and complete in data/part_B_final/:"
    echo "     - Train images: 400 (.jpg), Ground Truth: 400 (.mat)"
    echo "     - Test images:  316 (.jpg), Ground Truth: 316 (.mat)"
else
    echo "[INFO] data/part_B_final is missing or incomplete. Checking Git repository..."
    
    # 1. Attempt checkout from tracked git files
    if git ls-files --error-unmatch data/part_B_final/train_data/images/IMG_1.jpg >/dev/null 2>&1; then
        echo "[INFO] Restoring data/part_B_final from Git tracking..."
        git checkout HEAD -- data/part_B_final
    elif git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "[INFO] Pulling data/part_B_final from origin/RMR..."
        git fetch origin RMR
        git checkout origin/RMR -- data/part_B_final || true
    fi

    # 2. Check if git checkout succeeded
    if ! check_shb_complete; then
        echo "[INFO] Attempting download from Kaggle dataset mirror..."
        mkdir -p "${DATA_DIR}"
        TMP_ZIP="${DATA_DIR}/shanghaitech.zip"

        if command -v kaggle >/dev/null 2>&1; then
            echo "[INFO] Using Kaggle CLI..."
            kaggle datasets download -d timmaggioni/shanghaitech-crowd-counting -p "${DATA_DIR}"
            if [ -f "${DATA_DIR}/shanghaitech-crowd-counting.zip" ]; then
                mv "${DATA_DIR}/shanghaitech-crowd-counting.zip" "${TMP_ZIP}"
            fi
        elif command -v gdown >/dev/null 2>&1; then
            echo "[INFO] Using gdown mirror..."
            gdown "1I_rA58G5l11iYIbbM1v_Y1d2q4Z13WfD" -O "${TMP_ZIP}" || true
        fi

        if [ -f "${TMP_ZIP}" ]; then
            echo "[INFO] Extracting ShanghaiTech archive..."
            unzip -q -o "${TMP_ZIP}" -d "${DATA_DIR}/extracted_sh"
            
            # Reorganize if extracted with nested paths
            if [ -d "${DATA_DIR}/extracted_sh/part_B_final" ]; then
                mv "${DATA_DIR}/extracted_sh/part_B_final" "${DATA_DIR}/"
            elif [ -d "${DATA_DIR}/extracted_sh/ShanghaiTech/part_B" ]; then
                mv "${DATA_DIR}/extracted_sh/ShanghaiTech/part_B" "${SHB_DIR}"
            fi
            rm -rf "${DATA_DIR}/extracted_sh" "${TMP_ZIP}"
        fi
    fi
fi

# Final verification
if check_shb_complete; then
    echo "[SUCCESS] ShanghaiTech Part B is fully ready!"
else
    echo "[WARNING] Could not automatically obtain complete part_B_final."
    echo "          Run: git pull origin RMR"
    exit 1
fi

# Ensure manifests exist
echo "----------------------------------------------------------------------"
echo " Verifying Manifests (data/shb_train_all.jsonl & data/shb_test.jsonl)"
echo "----------------------------------------------------------------------"

PYTHON_BIN="python"
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

if [ ! -f "${DATA_DIR}/shb_train_all.jsonl" ]; then
    echo "[INFO] Generating data/shb_train_all.jsonl..."
    "${PYTHON_BIN}" -m rmr_count.prepare_manifest \
        --images "${TRAIN_IMG}" \
        --annotations "${TRAIN_GT}" \
        --dataset sha_b \
        --out "${DATA_DIR}/shb_train_all.jsonl"
fi

if [ ! -f "${DATA_DIR}/shb_test.jsonl" ]; then
    echo "[INFO] Generating data/shb_test.jsonl..."
    "${PYTHON_BIN}" -m rmr_count.prepare_manifest \
        --images "${TEST_IMG}" \
        --annotations "${TEST_GT}" \
        --dataset sha_b \
        --out "${DATA_DIR}/shb_test.jsonl"
fi

echo "[DONE] SHB manifests verified: $(wc -l < "${DATA_DIR}/shb_train_all.jsonl") train samples, $(wc -l < "${DATA_DIR}/shb_test.jsonl") test samples."
