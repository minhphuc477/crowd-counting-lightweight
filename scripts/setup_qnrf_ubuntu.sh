#!/usr/bin/env bash
# ==============================================================================
# Script tự động tải và cấu hình trọn gói UCF-QNRF Dataset trên Ubuntu
# Dung lượng tải: ~4.24 GB (4,552,702,871 bytes)
# Kết quả: data/ucf_qnrf/ + data/qnrf_train.jsonl (1,201 ảnh) + data/qnrf_test.jsonl (334 ảnh)
# ==============================================================================

set -euo pipefail

DATA_ROOT="data"
ARCHIVE_DIR="${DATA_ROOT}/archives"
DEST_DIR="${DATA_ROOT}/ucf_qnrf"
ZIP_FILE="${ARCHIVE_DIR}/UCF-QNRF_ECCV18.zip"
EXPECTED_BYTES=4552702871
URL="https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip"

echo "======================================================================"
echo ">>> KHỞI ĐỘNG CÀI ĐẶT UCF-QNRF TRÊN UBUNTU"
echo "======================================================================"

# 1. Tạo các thư mục cần thiết
mkdir -p "${ARCHIVE_DIR}"
mkdir -p "${DEST_DIR}"

# 2. Kiểm tra xem dataset đã được giải nén sẵn chưa
if [ -d "${DEST_DIR}/UCF-QNRF_ECCV18/Train" ] || [ -d "${DEST_DIR}/Train" ]; then
    echo "[INFO] UCF-QNRF đã tồn tại trong ${DEST_DIR}. Bỏ qua bước tải."
else
    # Kiểm tra xem file zip đã tải đủ chưa
    NEEDS_DOWNLOAD=1
    if [ -f "${ZIP_FILE}" ]; then
        ACTUAL_BYTES=$(stat -c%s "${ZIP_FILE}" 2>/dev/null || stat -f%z "${ZIP_FILE}" 2>/dev/null || echo 0)
        if [ "${ACTUAL_BYTES}" -ge "${EXPECTED_BYTES}" ]; then
            echo "[INFO] File zip đã tồn tại đủ dung lượng: ${ACTUAL_BYTES} bytes."
            NEEDS_DOWNLOAD=0
        else
            echo "[WARN] File zip chưa hoàn chỉnh (${ACTUAL_BYTES} / ${EXPECTED_BYTES} bytes). Tiếp tục tải tiếp (resume)..."
        fi
    fi

    if [ "${NEEDS_DOWNLOAD}" -eq 1 ]; then
        echo "[INFO] Đang tải UCF-QNRF (~4.24 GB) từ server Đại học Central Florida (CRCV)..."
        
        # Ưu tiên aria2c (tối đa 16 luồng song song, cực nhanh trên server)
        if command -v aria2c &> /dev/null; then
            echo "[INFO] Phát hiện aria2c. Tải 16 luồng song song (tối ưu hóa băng thông)..."
            aria2c -c -s 16 -x 16 -k 1M --check-certificate=false \
                -d "${ARCHIVE_DIR}" -o "UCF-QNRF_ECCV18.zip" "${URL}"
        elif command -v wget &> /dev/null; then
            echo "[INFO] Dùng wget với chế độ resume (-c)..."
            wget -c --no-check-certificate --tries=0 --read-timeout=60 -O "${ZIP_FILE}" "${URL}"
        elif command -v curl &> /dev/null; then
            echo "[INFO] Dùng curl với chế độ resume (-C -)..."
            curl -k -C - --retry 999 --retry-max-time 0 -L "${URL}" -o "${ZIP_FILE}"
        else
            echo "[INFO] Dùng script Python tích hợp..."
            python3 scripts/download_ucf_qnrf.py
        fi
    fi

    # 3. Giải nén dữ liệu
    echo "----------------------------------------------------------------------"
    echo "[INFO] Đang giải nén ${ZIP_FILE} vào ${DEST_DIR}..."
    if command -v unzip &> /dev/null; then
        unzip -q -n "${ZIP_FILE}" -d "${DEST_DIR}"
    else
        python3 -c "import zipfile; zipfile.ZipFile('${ZIP_FILE}').extractall('${DEST_DIR}')"
    fi
    echo "[INFO] Giải nén hoàn tất!"
fi

# 4. Kiểm tra thư viện scipy để đọc file annotation .mat
python3 -c "import scipy" 2>/dev/null || {
    echo "[WARN] Thiếu thư viện scipy để đọc file .mat. Đang cài đặt..."
    pip install scipy
}

# 5. Sinh canonical manifest files
echo "----------------------------------------------------------------------"
echo "[INFO] Đang sinh file manifest chuẩn: qnrf_train.jsonl, qnrf_test.jsonl..."
python3 scripts/create_ucf_qnrf_manifests.py

# 6. Kiểm tra nghiệm thu
echo "======================================================================"
echo ">>> KIỂM TRA NGHIỆM THU DỮ LIỆU UCF-QNRF:"
echo "======================================================================"
TRAIN_COUNT=$(wc -l < "${DATA_ROOT}/qnrf_train.jsonl" 2>/dev/null || echo 0)
TEST_COUNT=$(wc -l < "${DATA_ROOT}/qnrf_test.jsonl" 2>/dev/null || echo 0)

echo "Train samples: ${TRAIN_COUNT} (Kỳ vọng: 1201)"
echo "Test samples:  ${TEST_COUNT} (Kỳ vọng: 334)"

if [ "${TRAIN_COUNT}" -eq 1201 ] && [ "${TEST_COUNT}" -eq 334 ]; then
    echo ">>> [THÀNH CÔNG] Dữ liệu UCF-QNRF đã sẵn sàng 100% để huấn luyện!"
else
    echo ">>> [CẢNH BÁO] Số lượng mẫu chưa khớp kỳ vọng. Vui lòng kiểm tra lại!"
fi
