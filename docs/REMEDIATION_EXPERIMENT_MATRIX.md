# REMEDIATION & COMPLETION EXPERIMENT MATRIX (RMR-v3, $\le 104\text{k}$ Params)

Tài liệu đặc tả bộ **11 cấu hình thí nghiệm khắc phục (`configs/benchmarks/remediation/*.yaml`)** được thiết kế sau kiểm toán để giải quyết trực tiếp các biến nhiễu (confounders), các đối chứng Pretrained còn thiếu, và lỗi tỷ lệ thang đo của `Radon-OT`.

---

## 1. Bảng Thiết Kế 11 Thí Nghiệm Khắc Phục (`configs/benchmarks/remediation/`)

| STT | Tệp YAML (`configs/benchmarks/remediation/...`) | Dataset / Chế độ | Tham số | Thay đổi Cốt lõi so với Baseline Tương ứng | Giả thuyết / Lỗ hổng Kiểm chứng |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **R1** | `exp_sha_pretrained_no_neck_102k.yaml` | SHA / Pretrained (`1500 ep`) | 102,246 | `aspp_dilations: [1]`, `use_aspp_gap: false` | Kiểm chứng xem No-Neck (`66.24` Scratch) có giữ lợi thế trước ASPP (`67.57`) ở Pretrained SHA hay không. |
| **R2** | `exp_qnrf_pretrained_no_neck_102k.yaml` | QNRF / Pretrained (`1000 ep`) | 102,246 | `aspp_dilations: [1]`, `use_aspp_gap: false` | Đối chứng Pretrained No-Neck với ASPP (`105.36` MAE) trên ảnh độ phân giải cực đại $\le 2048$. |
| **R3** | `exp_jhu_pretrained_no_neck_102k.yaml` | JHU / Pretrained (`800 ep`) | 102,246 | `aspp_dilations: [1]`, `use_aspp_gap: false` | Giải quyết nghịch lý Scratch No-Neck (`59.65`) tốt hơn Pretrained ASPP (`61.72` Val / `76.58` Test). |
| **R4** | `exp_jhu_pretrained_disp10_104k.yaml` | JHU / Pretrained (`800 ep`) | 104,006 | `count_nb_dispersion: 10.0` (thay `50.0`) | Kiểm chứng mức overdispersion mạnh hơn (`r=10`) ở Pretrained JHU (`59.83` vs `63.63` ở Scratch). |
| **R5** | `exp_shb_pretrained_bg005_104k.yaml` | SHB / Pretrained (`1500 ep`) | 104,006 | `bayesian_background_ratio: 0.05` (thay `0.10`) | Bổ sung đối chứng Pretrained cho `b3_bg_ratio_005` (`9.41` Scratch vs Pretrained `10.19` dùng `0.10`). |
| **R6** | `exp_sha_abl_pure_bayesian_matched_015_104k.yaml` | SHA / Scratch (`1500 ep`) | 104,006 | `allocation_loss_type: bayesian`, `lambda_bayesian: 0.15`, `lambda_radon_ot: 0.0` | Loại bỏ biến nhiễu của `a2_pure_bayesian` cũ (từng đổi `lambda_bayesian` từ `0.15` sang `1.0`). |
| **R7** | `exp_sha_pretrained_radon_scaled_104k.yaml` | SHA / Pretrained (`1500 ep`) | 104,006 | `radon_ot_scale_by_length: true`, giữ nguyên `lambda_radon_ot: 0.85` | Phép thử 1 biến cân bằng thang đo CDF thực sự cho `Radon-OT` (khắc phục tỷ trọng ~`0.06%` loss cũ mà không đổi $\lambda$). |
| **R8** | `exp_qnrf_pretrained_radon_scaled_104k.yaml` | QNRF / Pretrained (`1000 ep`) | 104,006 | `radon_ot_scale_by_length: true`, giữ nguyên `lambda_radon_ot: 0.85` | Kiểm chứng 1 biến `Radon-OT` chuẩn hóa chiều dài chiếu trên QNRF ở đúng trọng số baseline `0.85`. |
| **R9** | `exp_sha_gaussian_mse_clean_pretrained_104k.yaml` | SHA / Pretrained (`1500 ep`) | 104,006 | `cell_loss_mode: mse`, `sigma: 4.0`, `lambda_cell: 1.0`, tắt NB/Bayes/ROT | Đối chứng Gaussian MSE sạch ở cấu hình Pretrained (`lr=1e-4, 1500 ep`) trên `sha_a_train_all.jsonl`. |
| **R10** | `exp_sha_gaussian_mse_jitter_sigma8_104k.yaml` | SHA / Pretrained (`1500 ep`) | 104,006 | Giống R9 + `train_manifest: data/sha_a_train_jitter_sigma8.jsonl` | So sánh trực tiếp độ bền nhiễu vị trí (`sigma=8px`) của Gaussian MSE vs Dual Bayesian + Radon-OT (`70.05`). |
| **R11** | `exp_sha_direct_pure_99k.yaml` | SHA / Pretrained (`1500 ep`) | **99,924** | `enable_regional_head: false` (`region_head` bị loại bỏ hoàn toàn) | Chứng minh kiến trúc thuần Direct (`0.6449` GFLOPs @ 512) không mang `4,082` tham số idle tương đương bản `104,006`. |

---

## 2. Dữ Liệu Tổng Hợp Trước Khi Chạy Tiếp (`configs/training_sweeps/`)

* **8 runs `INCOMPLETE` cần resume kèm `--allow-cross-commit-resume`** (checkpoint hợp lệ `0` corrupt/NaN; mốc tại HEAD `1873cca`):
  - SHA: `exp_sha_crop_256_104k` (`824/1500`, best ep `555` MAE `80.48`), `exp_sha_drop_rho10_104k` (`1068/1500`, best ep `775` MAE `68.09`), `exp_sha_drop_rho20_104k` (`798/1500`, best ep `520` MAE `77.51`).
  - QNRF: `exp_qnrf_crop_768_104k` (`941/1000`, best ep `570` MAE `102.10`), `exp_qnrf_jitter_sigma8_104k` (`575/1000`, best ep `260` MAE `110.09`), `exp_qnrf_data_frac010_104k` (`13/1000`, MAE `531.64` — mới đầu train, có thể resume hoặc chạy lại `-o`), `exp_qnrf_seed_1337_104k` (`4/1000`, chưa qua lần eval đầu — chỉ có `last.pt`, có thể resume hoặc chạy lại `-o`).
  - JHU: `exp_jhu_data_frac010_104k` (`649/800`, best ep `540` MAE `86.91`).
* **4 runs `NOT_STARTED` trong `training_sweeps/`** (đã xóa 2 thư mục stub 0-epoch rỗng `exp_sha_crop_384_104k` và `exp_qnrf_gaussian_mse_clean_104k` chứa `0` file `.pt` để tránh `RuntimeError` chặn khởi chạy mới):
  - `exp_sha_crop_384_104k`, `exp_qnrf_gaussian_mse_clean_104k`, `exp_qnrf_gaussian_mse_jitter_sigma8_104k`, `exp_qnrf_gaussian_mse_jitter_sigma16_104k`.
* **Lưu ý đánh giá JHU Test cho R3/R4:** Sau khi huấn luyện xong `exp_jhu_pretrained_no_neck_102k` hoặc `disp10`, chạy `python -m rmr_v3.eval --checkpoint ... --manifest data/jhu_test.jsonl ...` để đo cả Official Test (`59.65` Scratch No-Neck mới đo trên Val `500` ảnh).

---

## 3. Lưu Ý Kỹ Thuật & Lộ Trình Phát Triển Tiếp Mô Hình (R&D Notes cho RMR-v4+)

Những ghi chú dưới đây rút ra trực tiếp từ kiểm toán kiến trúc, hàm loss và các mode thất bại thực nghiệm để định hướng nâng cấp mô hình tiếp theo (sau khi hoàn tất R1–R11 và 12 sweeps còn dở):

1. **Giải quyết dứt điểm `region_head` & Hiệu chuẩn Bất định (Uncertainty Collapse vs Saturation):**
   * *Hiện trạng:* Bản Direct (`104,006` params) giữ `4,082` tham số của `region_head` hoàn toàn không được huấn luyện (`cov50 = 0.5456` trên SHA y hệt nhau giữa mọi biến thể Direct); còn khi bật solver (`b1_solver_sirt`), phân tán vùng bị bão hòa hai đầu (`disp_sat_low = 0.1742`, `disp_sat_high = 0.4793`), kéo `cov50` lên `0.8525` (`+35.3%` over-coverage) và làm MAE xấu hơn (`77.48 vs 74.29`).
   * *Hướng phát triển tiếp:* Tách rạch ròi hai nhánh: (a) với đường Pure Direct tối giản, dùng `enable_regional_head: false` (**99,924** params, `0.6449` GFLOPs @ 512); (b) nếu muốn khôi phục nhánh bất định vùng, thử nghiệm các hook đã tích hợp sẵn trong `rmr_v3/config/schema.py` như `faithful_regional_nb` + `faithful_dispersion_detach` hoặc `lambda_trunc_nb` với trọng số nhỏ để đo hiệu chuẩn độc lập trước khi gắn solver phản hồi.

2. **Khử Sương Mù Nền (`65.57%` agg / `71.27%` per-image mean `bg_mass_5`) & Dương Tính Giả trên Ảnh Trống (`185.10` phantom trên 4 ảnh JHU):**
   * *Nguyên nhân toán học:* Hàm kích hoạt `Softplus` có đạo hàm tiến về $0$ theo hàm mũ ở miền âm nên gradient dập nền bị triệt tiêu dưới weight decay, để lại nền sương mù dương trên diện tích lớn.
   * *Hướng phát triển tiếp:* Thử nghiệm cổng support cho phép nền trống tiến về $0$ thực sự thông qua các hook trong `schema.py`: `hurdle_head` / `hurdle_gating_mode` / `lambda_hurdle`, hoặc `lambda_fg_gate` / `lambda_hard_bg` / `hard_bg_ratio` / `head_activation`, kết hợp bổ sung mẫu ảnh thuần nền (negative distractor crops) trong train để giảm sai số trên 31 ảnh distractor JHU Test (`70.74 MAE`).

3. **Thay thế Dilated ASPP `[2,4,8]` (Khắc phục Comb-Aliasing mà không mất Ngữ cảnh Thưa):**
   * *Hiện trạng:* `No-Neck` (`DW 3x3 d=1`, `102,246` params) thắng ASPP `[2,4,8]+GAP` trên SHA/QNRF/JHU Scratch nhưng thua nhẹ trên SHB thưa (`10.94 vs 10.49`), đồng thời cho thấy bước giãn nở thưa `[2,4,8]` làm mờ xung điểm ở $d < 4\text{px}$.
   * *Hướng phát triển tiếp:* Nếu kết quả R1–R3 xác nhận `No-Neck` cũng thắng ở Pretrained, phát triển cổ mạng pha trộn đa tỉ lệ bước nhỏ dày đặc (vd `d=[1, 2]` + high-res skip `C4`) hoặc kiểm chứng `fine_head_dilations` / `scale_conditioned_fine_head` giữ nguyên trần $\le 104\text{k}$ params.

4. **Khắc phục Sụp Đổ khi Giảm Scale (`0.5x -> 116.75 MAE`, consistency error `10.68%`) & Nén Ảnh JPEG (`Q=10 -> 132.63`):**
   * *Hướng phát triển tiếp:* Khai thác cặp loss nhất quán đa tỉ lệ `lambda_scale_align` (`scale_align_kernel`, `scale_align_mask_bg`) hiện có trong `schema.py`, mở rộng biên dưới `scale_range` từ `0.6` xuống `0.45–0.50`, và thêm augment JPEG quality nhẹ khi huấn luyện để tăng độ bền thực tế.

5. **Khắc phục Suy Giảm Pretrained trên Tập Siêu Nhỏ (`UCF_CC_50`: `329.99` Pretrained vs `298.65` Scratch):**
   * *Hướng phát triển tiếp:* Với tập chỉ ~40 ảnh train/fold, `backbone_lr_scale: 0.1` kìm hãm thích ứng đặc trưng miền dày từ ImageNet; cần thử `backbone_lr_scale` cao hơn (`0.5` hoặc `1.0`) kèm regularizer chặt hơn.

6. **Kỷ Luật Thiết Kế Thí Nghiệm & Profiling Bắt Buộc cho Công Bố:**
   * Luôn giữ quy tắc **1 biến thay đổi / 1 ablation** (tránh lỗi `a2` đổi đồng thời cả loại loss và `lambda_bayesian`, lỗi R7/R8 cũ từng đổi cả `lambda_radon_ot`, hay đặt tên `b1_solver_sirt` khi dùng `mcbpro`); bật `radon_ot_scale_by_length: true` ngay từ đầu khi đánh giá Radon-OT.
   * Tách validation nội bộ cho SHA/SHB/QNRF (hoặc báo cáo thêm `LastVal` và EMA cuối kỳ) vì SHB Pretrained trôi `+33.1%` sau epoch 380 (`10.19 -> 13.56`), đồng thời đo độ trễ GPU/CPU/VRAM bằng `scripts/profile_efficiency.py`.
7. **Ghi Nhận Confounders Lịch Sử & Ràng Buộc Khởi Tạo Cần Tránh Ở RMR-v4:**
   * `ablations_qnrf/c1..c6`: chạy với `ema_decay=0.999`, `warmup_epochs=10` (so với `a1` `ema_decay=None`, `warmup_epochs=15`), nên `c5_no_neck` (`108.42 MAE`) không phải so sánh sạch với `a1` (`109.80`); `R2` trên Pretrained QNRF dùng cùng giao thức sạch.
   * `ablations_shb/c1..c4`: đổi `bayesian_background_ratio: 0.05` nên baseline hợp lệ là `b3_bg_ratio_005` (`9.41 MAE`), không phải `a1` (`10.49`).
   * `b1_solver_sirt` sụp đổ (`77.48` SHA, `65.3%` bão hòa dispersion): nguyên nhân gốc do `regional_nb_beta: 1.0` + `normalize_beta_weights: false` làm `train_region_nb` bùng nổ lên `1633.90` ở epoch 100 (`95.8%` tổng loss), áp đảo hoàn toàn `count`/`bayesian`.
   * Quy ước `init_m0` & Default Schema: `FineMeasureHead`/`ResidualDilatedFineHead` luôn chia `init_m0 / 4.0` khi `subpixel_stride2=true` (vì `init_m0` được quy ước theo mật độ stride-4), trong khi các YAML benchmark/ablation hiện hành nhập mật độ stride-2 nên xảy ra chia 4 lần hai (và `exp_jhu_abl_c1_stride4` nhân 4 lần `init_m0` còn `sha/shb/qnrf` stride-4 thì không). Để bảo toàn tính đồng nhất với 52 runs đã hoàn tất, giữ nguyên `init_m0` trên `R1–R11` và chuẩn hóa bằng `init_m0: auto` từ RMR-v4 trở đi; đồng thời `CRITICAL_TRAIN_DEFAULTS["min_lr_ratio"]` (`0.05`) và fallback `early_stopping` (`False`) trong `trainer.py` đã được đồng bộ với runtime thực tế.
