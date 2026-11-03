# THIẾT KẾ THÍ NGHIỆM TOÀN BỘ DATASETS & KIỂM TOÁN CHUẨN MỰC (104K DIRECT FEEDFORWARD)

> **Trạng thái**: **16/16 CONFIGS ĐÃ KIỂM TOÁN VÀ PASS 100%**  
> **Cam kết cốt lõi**: Trainable Parameters = **104,006 params** (Tuyệt đối $\le 104,441$ budget), Stride-2 Subpixel, Direct Feedforward ($y = y_0$), Dual Bayesian + Radon-OT Loss.  
> **Git Commit**: `b50a04f` trên nhánh `RMR`.

---

## 1. TỔNG QUAN HỆ THỐNG DATASET & MA TRẬN THÍ NGHIỆM

Để bài báo khoa học đạt chuẩn mực hội nghị top-tier (CVPR/ECCV/ICCV/IEEE TPAMI), một kiến trúc không thể chỉ báo cáo kết quả trên một tập dữ liệu ShanghaiTech Part A mà bắt buộc phải chứng minh khả năng tổng quát hóa (generalization) và tính ưu việt trên toàn bộ các benchmark kinh điển của lĩnh vực Crowd Counting.

Dưới đây là ma trận 4 benchmark chính đã được tích hợp đầy đủ manifest, dữ liệu ảnh trên ổ đĩa và cấu hình chuẩn hóa:

| Benchmark | Số mẫu Train | Số mẫu Test | Đặc tính mật độ & độ phân giải | Chế độ Scratch (0% Pretrain) | Chế độ Pretrained (ImageNet) |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **ShanghaiTech Part A (SHA)** | 300 | 182 | Cực kỳ đông đúc (Dense), kích thước trung bình $589 \times 868$ | ✅ Đã chạy (MAE 70.50 TTA) | ✅ Đã chạy (MAE 64.91 TTA) |
| **ShanghaiTech Part B (SHB)** | 400 | 316 | Mật độ thưa (Sparse), góc rộng đường phố, $768 \times 1024$ | ✅ Config chuẩn hóa (1500ep) | ✅ Config chuẩn hóa (1500ep) |
| **UCF-QNRF** | 1,201 | 334 | Đa dạng cực lớn, ảnh độ phân giải siêu cao (lên tới $6000 \times 4000$) | ✅ Config chuẩn hóa (1000ep) | ✅ Config chuẩn hóa (1000ep) |
| **UCF-CC-50** | 40 / fold | 10 / fold | 50 ảnh cực khó, đánh giá 5-fold Cross Validation chuẩn mực | ✅ 5-Fold F1-F5 (1000ep) | ✅ 5-Fold F1-F5 (1000ep) |
| **NWPU-Crowd** | 3,109 | 500 val | Mật độ và bối cảnh đa dạng nhất, ~40GB ảnh (Manifest sẵn sàng) | ⏳ Cần tải raw images | ⏳ Cần tải raw images |

---

## 2. TIÊU CHUẨN KIẾN TRÚC & LOSS BẤT BIẾN (INVARIANTS)

Tất cả 16 thí nghiệm đều tuân thủ nghiêm ngặt chuẩn mực Champion đã được kiểm chứng bằng thực nghiệm:

1. **Kiến trúc Champion 104k**:
   - Backbone: `mobilenetv4_conv_small_050.e3000_r224_in1k` (81,728 params).
   - Neck: `ASPP-Lite` với dilations $[2, 4, 8]$ + Gated Global Average Pooling (`feature_width: 32`, 19,872 params).
   - Head: Subpixel Stride-2 Polyphase Head (2,406 params).
   - **Tổng tham số huấn luyện (Trainable Parameters)**: **Đúng 104,006 parameters** (Headroom an toàn: 435 params dưới trần $104,441$).
   - **Direct Feedforward**: `enable_solver: false`, $y = y_0$ (loại bỏ hoàn toàn SIRT iterative loops và blur do ma trận vận chuyển).

2. **Hàm mất mát trực giao (Orthogonal Supervision)**:
   - $\mathcal{L}_{\text{Radon-OT}}$ ($\lambda = 0.85$): Vận chuyển khối lượng vi mô đa hướng không cần tính ma trận khoảng cách 2D đầy đủ.
   - $\mathcal{L}_{\text{Bayesian}}$ ($\lambda = 0.15$): Neo giữ điểm cực đại mật độ tại đúng tọa độ head ground-truth.
   - $\mathcal{L}_{\text{Count-NB}}$ ($\lambda = 0.50$): Bảo toàn tổng số đếm vĩ mô với phân phối Negative Binomial (chống over-fitting ngoại lai).
   - Triệt tiêu hoàn toàn các loss gây nhiễu: $\lambda_{\text{flat}}=0.0, \lambda_{\text{reg}}=0.0, \lambda_{\text{cell}}=0.0$.

3. **Data Augmentation Đầy Đủ**:
   - Random Crop $512 \times 512$, Random Rescale $[0.60, 1.40]$, Horizontal Flip ($p=0.5$).
   - Color Jitter (Brightness 0.2, Contrast 0.2, Saturation 0.3, Gamma $[0.6, 1.5]$, Grayscale $p=0.3$).
   - Random Erasing ($p=0.35$, scale $[0.005, 0.03]$, max 3 patches).

---

## 3. BẢNG KẾT QUẢ KIỂM TOÁN TỰ ĐỘNG (16/16 PASSED)

Kiểm toán đã chạy qua từng file YAML trong `configs/benchmarks/`, khởi tạo trực tiếp mô hình bằng `rmr_v3.engine.make_model`, kiểm tra kích thước tensor đầu ra, kiểm tra số tham số, và load thử mẫu ảnh thực tế từ ổ đĩa:

```text
===============================================================================================
BENCHMARK SUITE AUDIT SUMMARY: 16/16 PASSED
===============================================================================================
PASS | exp_qnrf_direct_pretrained_104k.yaml | Params: 104006  | Data: 1201 / 334  | Pretrained: True  | LR: 0.0001
PASS | exp_qnrf_direct_scratch_104k.yaml   | Params: 104006  | Data: 1201 / 334  | Pretrained: False | LR: 0.0005
PASS | exp_sha_direct_pretrained_104k.yaml | Params: 104006  | Data:  300 / 182  | Pretrained: True  | LR: 0.0001
PASS | exp_sha_direct_scratch_104k.yaml    | Params: 104006  | Data:  300 / 182  | Pretrained: False | LR: 0.0005
PASS | exp_shb_direct_pretrained_104k.yaml | Params: 104006  | Data:  400 / 316  | Pretrained: True  | LR: 0.0001
PASS | exp_shb_direct_scratch_104k.yaml    | Params: 104006  | Data:  400 / 316  | Pretrained: False | LR: 0.0005
PASS | exp_ucf_cc_50_fold1.yaml            | Params: 104006  | Data:   40 / 10   | Pretrained: True  | LR: 0.0001
PASS | exp_ucf_cc_50_fold2.yaml            | Params: 104006  | Data:   40 / 10   | Pretrained: True  | LR: 0.0001
PASS | exp_ucf_cc_50_fold3.yaml            | Params: 104006  | Data:   40 / 10   | Pretrained: True  | LR: 0.0001
PASS | exp_ucf_cc_50_fold4.yaml            | Params: 104006  | Data:   40 / 10   | Pretrained: True  | LR: 0.0001
PASS | exp_ucf_cc_50_fold5.yaml            | Params: 104006  | Data:   40 / 10   | Pretrained: True  | LR: 0.0001
PASS | exp_ucf_cc_50_scratch_fold1.yaml    | Params: 104006  | Data:   40 / 10   | Pretrained: False | LR: 0.0005
PASS | exp_ucf_cc_50_scratch_fold2.yaml    | Params: 104006  | Data:   40 / 10   | Pretrained: False | LR: 0.0005
PASS | exp_ucf_cc_50_scratch_fold3.yaml    | Params: 104006  | Data:   40 / 10   | Pretrained: False | LR: 0.0005
PASS | exp_ucf_cc_50_scratch_fold4.yaml    | Params: 104006  | Data:   40 / 10   | Pretrained: False | LR: 0.0005
PASS | exp_ucf_cc_50_scratch_fold5.yaml    | Params: 104006  | Data:   40 / 10   | Pretrained: False | LR: 0.0005
===============================================================================================
```

---

## 4. HƯỚNG DẪN LỆNH CHẠY HUẤN LUYỆN (EXECUTION COMMANDS)

Để chạy các thí nghiệm benchmark trên server hoặc GPU local, chỉ cần kích hoạt môi trường và thực thi các câu lệnh chuẩn dưới đây:

### A. ShanghaiTech Part B (SHB)
```bash
# SHB Scratch (0% ImageNet, 1500 epochs, lr=5e-4)
python -m rmr_v3.train --config configs/benchmarks/exp_shb_direct_scratch_104k.yaml

# SHB Pretrained (ImageNet weights, 1500 epochs, lr=1e-4)
python -m rmr_v3.train --config configs/benchmarks/exp_shb_direct_pretrained_104k.yaml
```

### B. UCF-QNRF
```bash
# UCF-QNRF Scratch (0% ImageNet, 1000 epochs, lr=5e-4, max_size=2048)
python -m rmr_v3.train --config configs/benchmarks/exp_qnrf_direct_scratch_104k.yaml

# UCF-QNRF Pretrained (ImageNet weights, 1000 epochs, lr=1e-4, max_size=2048)
python -m rmr_v3.train --config configs/benchmarks/exp_qnrf_direct_pretrained_104k.yaml
```

### C. UCF-CC-50 (5-Fold Cross Validation)
```bash
# Pretrained 5-Folds
python -m rmr_v3.train --config configs/benchmarks/exp_ucf_cc_50_fold1.yaml
python -m rmr_v3.train --config configs/benchmarks/exp_ucf_cc_50_fold2.yaml
python -m rmr_v3.train --config configs/benchmarks/exp_ucf_cc_50_fold3.yaml
python -m rmr_v3.train --config configs/benchmarks/exp_ucf_cc_50_fold4.yaml
python -m rmr_v3.train --config configs/benchmarks/exp_ucf_cc_50_fold5.yaml
```

---

## 5. ĐÁNH GIÁ TTA (TEST-TIME AUGMENTATION) SAU KHI TRAIN

Sau khi checkpoint tốt nhất được lưu tại thư mục `runs/benchmarks/<exp_name>/best_checkpoint.pth`, có thể chạy ngay script đánh giá TTA để tối ưu MAE thêm $\sim 2-3$ điểm:

```bash
python scripts/eval_tta.py \
  --checkpoint runs/benchmarks/<exp_name>/best_checkpoint.pth \
  --manifest data/<manifest_test>.jsonl \
  --data-root data \
  --output-stride 2
```
