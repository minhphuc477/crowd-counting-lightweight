# BÁO CÁO KIỂM TOÁN TOÀN BỘ 37 FILE YAML & HIỆU CHUẨN SIÊU THAM SỐ HỌC

> **Phạm vi kiểm toán**: Toàn bộ 37 cấu hình YAML trong repository (`configs/benchmarks/`, `configs/benchmarks/ablations_shb/`, `configs/fairness_study/`, `configs/rmr_v73/`).  
> **Trạng thái kiểm toán**: **100% (37/37) PASS TẤT CẢ CÁC RÀNG BUỘC KỸ THUẬT & TOÁN HỌC**.  
> **Git Commit cập nhật**: `0d3503c` trên nhánh `RMR`.

---

## 1. MA TRẬN HIỆU CHUẨN SIÊU THAM SỐ THEO PHÂN PHỐI THỰC NGHIỆM

Thay vì dùng chung một bộ siêu tham số cho mọi tập dữ liệu (sai lầm phổ biến khiến mô hình bị suy giảm hiệu năng), kiểm toán đã đo đạc phân phối hình học chính xác của từng dataset và hiệu chuẩn tương ứng:

| Dataset | Số mẫu (Tr/Te) | Count TB | $k$-NN Median Dist | Prior Density $m_0$ (Stride 2) | Bayesian $\sigma_{\max}$ | Background Ratio | Đánh giá Phân tầng Bins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ShanghaiTech Part A (SHA)** | 300 / 182 | 501.4 | **$10.71$ px** | **$0.003941$** (Đo đạc: 0.00395) | **$4.0$** (Cụm đông) | $0.05$ | `[100.0, 500.0]` |
| **ShanghaiTech Part B (SHB)** | 400 / 316 | 123.2 | **$15.89$ px** | **$0.000625$** (Thưa gấp 6.3x) | **$8.0$** (Đầu lớn/rộng) | **$0.10$** (Ngoại cảnh rộng) | `[50.0, 150.0]` |
| **UCF-QNRF** | 1,201 / 334 | 842.2 | **$22.80$ px** | **$0.001498$** (Ảnh siêu lớn) | **$8.0$** (Scale đa dạng) | $0.05$ | `[200.0, 1000.0]` |
| **UCF-CC-50** | 40 / 10 | 1279.5 | **$8.54$ px** | **$0.008416$** (Siêu nghẽn nhỏ) | **$4.0$** (Cực nghẽn) | $0.05$ | `[500.0, 1500.0]` |

---

## 2. KẾT QUẢ KIỂM TRA TỪNG THÀNH PHẦN HYPERPARAMETER

### 2.1. Cặp Siêu tham số Huấn luyện (Learning Rate & Backbone Scale)
Kiểm tra tính nhất quán giữa chế độ Pretrained vs. Scratch:
- **Chế độ Scratch (0% ImageNet)**:
  * Tất cả các file scratch đều có `pretrained: false`.
  * `train.lr: 0.0005` (Tốc độ học tối ưu cho khởi tạo ngẫu nhiên).
  * `model.backbone_lr_scale: 1.0` và `train.backbone_lr_scale: 1.0` (Backbone học cùng tốc độ với Neck/Head, không bị kìm hãm ở mức 0.1).
- **Chế độ Pretrained (ImageNet Fine-Tuning)**:
  * Tất cả các file pretrained đều có `pretrained: true`.
  * `train.lr: 0.0001`.
  * `model.backbone_lr_scale: 0.1` và `train.backbone_lr_scale: 0.1` (Kìm hãm tốc độ học của backbone đã học tốt đặc trưng ImageNet để chống catastrophic forgetting).

### 2.2. Kiểm Tra Đường dẫn Dữ liệu & Xung đột Thư mục Đầu ra (Output Collision)
- **Đường dẫn**: 100% `train_manifest` và `val_manifest` đều trỏ chính xác đến các file tồn tại trên ổ đĩa.
- **Xung đột Thư mục Đầu ra (`output_dir`)**: 37/37 cấu hình đều có đường dẫn `output_dir` hoàn toàn độc lập, không có bất kỳ cấu hình nào bị ghi đè chéo khi chạy đồng thời hoặc theo mẻ (batch).

### 2.3. Ràng buộc Ngân sách Tham số (Parameter Ceiling Invariant)
- **Track A (Benchmark & Ablation chuẩn)**: Toàn bộ 30 cấu hình Track A có số tham số huấn luyện nằm trong khoảng **$103,744 - 104,006$ parameters**, tuyệt đối tuân thủ trần $\le 104,441$.
- **Track B (Dung lượng mở rộng)**: 7 cấu hình `rmr_v73` có số tham số trong khoảng **$113,463 - 127,895$ parameters**, tuyệt đối tuân thủ trần $\le 200,000$.

---

## 3. BẢNG LOG KIỂM TOÁN TỔNG THỂ (37/37 PASSED)

```text
==============================================================================================================
DEEP CONFIG AUDIT SUMMARY: 37 files checked | 0 CRITICAL ISSUES DETECTED
==============================================================================================================
[OK   ] configs/benchmarks/exp_qnrf_direct_pretrained_104k.yaml  | Params: 104006 | Pretrained: True  | LR: 0.0001 | m0: 0.001498
[OK   ] configs/benchmarks/exp_qnrf_direct_scratch_104k.yaml     | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.001498
[OK   ] configs/benchmarks/exp_sha_direct_pretrained_104k.yaml   | Params: 104006 | Pretrained: True  | LR: 0.0001 | m0: 0.00394075
[OK   ] configs/benchmarks/exp_sha_direct_scratch_104k.yaml      | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.00394075
[OK   ] configs/benchmarks/exp_shb_direct_pretrained_104k.yaml   | Params: 104006 | Pretrained: True  | LR: 0.0001 | m0: 0.000625
[OK   ] configs/benchmarks/exp_shb_direct_scratch_104k.yaml      | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.000625
[OK   ] configs/benchmarks/exp_ucf_cc_50_fold1.yaml              | Params: 104006 | Pretrained: True  | LR: 0.0001 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_fold2.yaml              | Params: 104006 | Pretrained: True  | LR: 0.0001 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_fold3.yaml              | Params: 104006 | Pretrained: True  | LR: 0.0001 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_fold4.yaml              | Params: 104006 | Pretrained: True  | LR: 0.0001 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_fold5.yaml              | Params: 104006 | Pretrained: True  | LR: 0.0001 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_scratch_fold1.yaml      | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_scratch_fold2.yaml      | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_scratch_fold3.yaml      | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_scratch_fold4.yaml      | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.008416
[OK   ] configs/benchmarks/exp_ucf_cc_50_scratch_fold5.yaml      | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.008416
[OK   ] configs/benchmarks/ablations_shb/exp_shb_abl_a1_flagship_104k.yaml              | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.000625
[OK   ] configs/benchmarks/ablations_shb/exp_shb_abl_a2_pure_bayesian_104k.yaml         | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.000625
[OK   ] configs/benchmarks/ablations_shb/exp_shb_abl_a3_pure_radon_ot_104k.yaml         | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.000625
[OK   ] configs/benchmarks/ablations_shb/exp_shb_abl_a4_gaussian_mse_104k.yaml          | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.000625
[OK   ] configs/benchmarks/ablations_shb/exp_shb_abl_b1_solver_sirt_104k.yaml           | Params: 103744 | Pretrained: False | LR: 0.0005 | m0: 0.000625
[OK   ] configs/benchmarks/ablations_shb/exp_shb_abl_b2_sigma4_uncalibrated_104k.yaml   | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.000625
[OK   ] configs/benchmarks/ablations_shb/exp_shb_abl_b3_bg_ratio_005_104k.yaml          | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.000625
[OK   ] configs/fairness_study/exp_a1_scratch_104k_proposed.yaml                       | Params: 103744 | Pretrained: False | LR: 0.0005 | m0: 0.00394075
[OK   ] configs/fairness_study/exp_a2_scratch_104k_mse.yaml                            | Params: 103744 | Pretrained: False | LR: 0.0005 | m0: 0.00394075
[OK   ] configs/fairness_study/exp_a3_pretrained_104k_mse.yaml                         | Params: 103744 | Pretrained: True  | LR: 0.0001 | m0: 0.00394075
[OK   ] configs/fairness_study/exp_b1_abl_pure_radon_ot.yaml                           | Params: 103744 | Pretrained: True  | LR: 0.0001 | m0: 0.00394075
[OK   ] configs/fairness_study/exp_b2_abl_radon_bayesian.yaml                          | Params: 103744 | Pretrained: True  | LR: 0.0001 | m0: 0.00394075
[OK   ] configs/fairness_study/exp_b3_abl_2d_sinkhorn.yaml                             | Params: 103744 | Pretrained: True  | LR: 0.0001 | m0: 0.00394075
[OK   ] configs/fairness_study/exp_scratch_direct_dual_bayesian_radon_ot.yaml          | Params: 104006 | Pretrained: False | LR: 0.0005 | m0: 0.00394075
[OK   ] configs/rmr_v73/v73_track_b_exp1_flagship_direct_seed42.yaml                   | Params: 113463 | Pretrained: True  | LR: 0.0001 | m0: 0.015763
[OK   ] configs/rmr_v73/v73_track_b_exp2_pure_bayesian_1loss_seed42.yaml               | Params: 113463 | Pretrained: True  | LR: 0.0001 | m0: 0.015763
[OK   ] configs/rmr_v73/v73_track_b_exp3_capacity_w64_hb2_seed42.yaml                   | Params: 127895 | Pretrained: True  | LR: 0.0001 | m0: 0.015763
[OK   ] configs/rmr_v73/v73_track_b_exp4_abl_standard_gap_seed42.yaml                   | Params: 113463 | Pretrained: True  | LR: 0.0001 | m0: 0.015763
[OK   ] configs/rmr_v73/v73_track_b_exp5_bayesian_count_l1_seed42.yaml                  | Params: 113463 | Pretrained: True  | LR: 0.0001 | m0: 0.015763
[OK   ] configs/rmr_v73/v73_track_b_exp6_flagship_seed7.yaml                            | Params: 113463 | Pretrained: True  | LR: 0.0001 | m0: 0.015763
[OK   ] configs/rmr_v73/v73_track_b_exp7_flagship_seed101.yaml                          | Params: 113463 | Pretrained: True  | LR: 0.0001 | m0: 0.015763
==============================================================================================================
```
