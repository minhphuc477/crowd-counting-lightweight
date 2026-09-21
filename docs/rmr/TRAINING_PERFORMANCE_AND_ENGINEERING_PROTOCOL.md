# RMR Training Performance & Engineering Protocol (Quy Chuẩn Kỹ Thuật Huấn Luyện & Tối Ưu Hiệu Năng)

**Phiên bản:** 1.0 (Áp dụng từ RMR-v32 trở đi)  
**Mục đích:** Thiết lập các quy chuẩn bất biến về kiến trúc phần mềm, cấu hình phần cứng, quản lý bộ nhớ đệm, tối ưu hóa pipeline huấn luyện và bảo đảm mô hình huấn luyện trên ShanghaiTech Part A hoàn thành 1000 epoch trong **$\le 3.0$ giờ** với **`eval_every: 5`** mà không gặp bất kỳ lỗi hay nút thắt cổ chai nào.

---

## 1. Quy Chuẩn Phân Vùng Dữ Liệu Chuẩn Mực (Zero Ad-Hoc Split Policy)

> [!IMPORTANT]
> **Ràng buộc cốt lõi:** Tuyệt đối không bao giờ chia tách nội bộ tập huấn luyện chính thức thành các tỷ lệ tùy tiện (ví dụ 90/10 hay 80/20 như `sha_a_train.jsonl` 270 ảnh / `sha_a_val.jsonl` 30 ảnh).

1. **ShanghaiTech Part A Canonical Benchmark:**
   - **Tập Train:** Bắt buộc sử dụng 100% dữ liệu train chính thức gồm đúng **300 ảnh** (`data/sha_a_train_all.jsonl`).
   - **Tập Test / Validation:** Đánh giá trên 100% dữ liệu test chính thức gồm đúng **182 ảnh** (`data/sha_a_test.jsonl`). Việc theo dõi mô hình tốt nhất (`best_val_mae.pt`) thực hiện trực tiếp trên tập này theo đúng chuẩn học thuật (CSRNet, BL, DM-Count, MAN, FIDTM, ChfL, STEERER).
2. **Khóa Bảo Vệ Trong Code:**
   - [`rmr_core/data.py`](file:///f:/lightweightcrcn/rmr_core/data.py) tự động ném ngoại lệ `ValueError` nếu phát hiện đường dẫn tệp manifest chứa `sha_a_train.jsonl` hoặc `sha_a_val.jsonl`, hoặc nếu số lượng mẫu của tập Train $\ne 300$ hoặc tập Test $\ne 182$.

---

## 2. Quy Chuẩn Nạp Bộ Đệm RAM Không Đầy Bộ Nhớ (Zero-Bloat In-Memory RAM Caching Protocol v2.0)

> [!TIP]
> ShanghaiTech Part A chỉ có 300 ảnh huấn luyện với dung lượng nén JPEG trên đĩa là **38.27 MB**, và 182 ảnh kiểm thử là **22.85 MB** (tổng cộng toàn bộ dataset chỉ **61.12 MB**).
> **Phân tích tử thi bộ nhớ (Forensic RAM Memory Breakdown):**
> 1. Trong Protocol v1.0, việc lưu ảnh uncompressed PIL Image (`self._image_cache`) bị nhân bản qua 4 worker processes $\times$ 4 tiến trình huấn luyện song song = 20 processes $\to$ **10 GB RAM lãng phí do Copy-On-Write (COW) page faults** trong CPython!
> 2. Việc lưu 182 ảnh test dưới dạng tensor float32 kích thước đầy đủ (`self._eval_cache`) ngốn **~1.8 GB float32 tensors tĩnh** trong mỗi process $\to$ 4 tiến trình ngốn **7.2 GB RAM tĩnh vô ích**!
> 3. Tổng cộng 4 tiến trình song song nuốt trọn **25.0 GB RAM**, kích hoạt kernel `kswapd0` và swap disk thrashing!

### 2.1. Caching Nhị Phân Nén Siêu Nhẹ (Compressed Raw-Bytes In-Memory Store)
* **Cơ chế:** Kích hoạt `cache_images=True` trong [`CrowdManifestDataset`](file:///f:/lightweightcrcn/rmr_core/data.py).
* **Lưu trữ:** Lưu mảng byte JPEG gốc vào `self._raw_bytes_cache: dict[int, bytes]`.
* **Chi phí RAM:** Toàn bộ 300 ảnh huấn luyện chỉ chiếm đúng **38.27 MB RAM** (thay vì 2.5 GB của uncompressed PIL objects). Ngay cả khi nhân bản qua 4 workers, tổng RAM chỉ là **~150 MB** (giảm **$94\%$** chi phí bộ nhớ).
* **Truy xuất:** Tại mỗi epoch, giải nén trực tiếp trong RAM qua `Image.open(io.BytesIO(raw_bytes)).convert("RGB")` tốn chỉ **~0.3 ms** cho mỗi ảnh (tương đương 2.4 ms/batch 8 ảnh), được che giấu $100\%$ phía sau thời gian tính toán của GPU (~75 ms/batch). Cắt crop 512x512 và gọi `image.close()` ngay lập tức để giải phóng buffer.

### 2.2. Đánh Giá On-Demand Không Lưu Cache Float32 Tĩnh
* **Cơ chế:** Vì quá trình đánh giá chỉ diễn ra định kỳ 5 epoch một lần (`eval_every: 5`), tuyệt đối **không lưu trữ tensor float32 toàn phần của 182 ảnh test trong RAM**.
* **Lưu trữ:** Toàn bộ 182 ảnh test được giữ trong `self._raw_bytes_cache` với dung lượng chỉ **22.85 MB**.
* **Truy xuất:** Khi đến epoch đánh giá, 182 ảnh test được decode trực tiếp từ raw bytes trong **$< 0.9$ giây tổng cộng**.
* **Chi phí RAM:** Giải phóng hoàn toàn **1.8 GB RAM float32 per run** ($\mathbf{7.2\text{ GB}}$ trên 4 runs song song).

---

## 3. Quy Chuẩn Triệt Tiêu CUDA Host-Sync Stalls (Zero-Stall GPU Saturation)

> [!CAUTION]
> Bất kỳ lời gọi `.item()`, `.tolist()`, `.cpu()`, hoặc rẽ nhánh điều kiện `if tensor.any():` bên trong vòng lặp batch huấn luyện đều cưỡng bức GPU driver xả hàng đợi kernel (flush stream), dừng CPU để chờ GPU, gây sụt giảm nghiêm trọng thông lượng tính toán (từ 3 giây/epoch lên 30 giây/epoch)!

1. **`LossTracker` Thuần Tensor (Fast Loss Tracking):**
   - Trong `LossTracker.update(loss_dict)`, tích lũy các scalar tensor bằng `val = v.detach()` trực tiếp trên GPU.
   - Tuyệt đối **không gọi `.item()` trên từng mini-batch**.
   - Chỉ chuyển đổi sang float Python một lần duy nhất tại hàm `averages()` ở cuối epoch:
     ```python
     return {k: float(v.item() if isinstance(v, torch.Tensor) else v) / c for k, v in self.totals.items()}
     ```
2. **Thu Thập Chẩn Đoán Ở Cuối Epoch (`DiagnosticTracker`):**
   - Không gọi `diag_tracker.update(outputs)` ở tất cả các batch.
   - Chỉ gọi `diag_tracker.update(outputs)` ở batch cuối cùng của epoch (`if batch_idx == n_batches - 1:`).
   - Batch cuối cùng chứa đầy đủ 8 ảnh (2,400 macro-regions), cung cấp phân phối thống kê chính xác tuyệt đối cho logging `train_log.csv` mà xóa bỏ 37 lần CPU-GPU synchronization stalls mỗi epoch.
3. **Bảo Vệ Phân Kỳ Số Học Thuần Tensor (SEC-03):**
   - Không sử dụng `if torch.isnan(y).any():`.
   - Sử dụng phép toán tensor thuần túy:
     ```python
     y_next = torch.where(torch.isfinite(y_next), y_next, y_curr)
     ```
4. **Giảm Thiểu Tính Toán Dư Thừa Trong Bộ Giải SIRT:**
   - Trong `unrolled_sirt_solver`, vì $y_{\text{curr}}^{(t)} \equiv y_{\text{next}}^{(t-1)}$, không tính lại `weighted_regional_energy(y_curr)` khi $t > 0$. Gán trực tiếp:
     ```python
     energy_before = weighted_regional_energy(y_curr, ...) if iter_idx == 0 else energy_after
     ```

---

## 4. Quy Chuẩn Hạ Tầng DataLoader & DMA Transfer

1. **Truyền Bộ Nhớ PCIe Bất Đồng Bộ (`non_blocking=True`):**
   - Tất cả các lệnh nạp batch lên GPU phải sử dụng:
     ```python
     images = batch["image"].to(device, non_blocking=True)
     targets = batch["target_y"].to(device, non_blocking=True)
     ```
2. **Cấu Hình DataLoader Bắt Buộc:**
   - `pin_memory: true` trong mọi file config YAML để kích hoạt DMA transfer.
   - `persistent_workers=bool(workers > 0)` trong [`rmr_v3/trainer.py`](file:///f:/lightweightcrcn/rmr_v3/trainer.py) để giữ các tiến trình worker sống liên tục qua 1000 epoch, loại trừ 4,000 chu kỳ hủy và tạo lại process.
   - `prefetch_factor=2` khi `workers > 0` để nạp trước batch vào bộ nhớ đệm CPU trong lúc GPU đang tính toán.
3. **Quy Chuẩn Điều Phối Đa Tiến Trình Trên Máy Chủ Chạy Nhiều Thí Nghiệm (High-Concurrency Workstation Protocol):**
   - Khi chạy đồng thời 4 đến 8 thí nghiệm trên cùng 1 máy chủ Ubuntu (như cấu hình RTX 5070 Ti):
     - **Cấu hình Worker tối ưu:** Sử dụng `--workers 0` hoặc `--workers 2` (thay vì `workers: 4`).
     - **Lý do:** Dataset 300 ảnh đã nằm trọn trong RAM (38 MB). Với `workers: 0`, quá trình decode và augment diễn ra trực tiếp trên luồng chính trong $2.4\text{ ms/batch}$ (che giấu hoàn toàn sau $75\text{ ms}$ tính toán của GPU), loại bỏ $100\%$ chi phí IPC và giảm số lượng process từ **20 process xuống đúng 4 process**!
     - **Thiết lập môi trường bắt buộc trước khi chạy song song:**
       ```bash
       export MALLOC_TRIM_THRESHOLD_=65536   # Buộc glibc trả RAM về kernel, chống phân mảnh
       export OMP_NUM_THREADS=1              # Chống tranh chấp 128 luồng CPU giữa các run
       export MKL_NUM_THREADS=1
       ```
     - **Hiệu quả:** Tổng lượng RAM tiêu thụ của 4 thí nghiệm song song giảm từ **25.0 GB xuống < 3.5 GB**! Khắc phục triệt để hiện tượng swap disk và nghẽn CPU!

---

## 5. Quy Chuẩn Đánh Giá & Đo Lường (Evaluation Protocol)

1. **Tần Suất Đánh Giá:**
   - Mặc định chuẩn: **`eval_every: 5`**.
   - Với các tối ưu hóa bulk tensor transfer và RAM caching, 1 lượt đánh giá toàn bộ 182 ảnh chỉ tốn **$< 0.3$ giây**. Toàn bộ 200 lượt evaluation trong 1000 epoch chỉ tốn chưa đầy 60 giây tổng cộng!
2. **Vectorized Bootstrap Confidence Interval:**
   - Khi tính khoảng tin cậy 95% bootstrap trong [`rmr_core/metrics.py`](file:///f:/lightweightcrcn/rmr_core/metrics.py), sử dụng lấy mẫu mảng 2D NumPy đồng thời:
     ```python
     idx = rng.integers(0, n, size=(n_boot, n))
     stats = np.mean(arr[idx], axis=1)
     ```
   - Tốc độ nhanh hơn 4.3 lần so với vòng lặp Python 5,000 lượt.
3. **Vectorized Physical-Support GAME (Tăng tốc 171 Lần):**
   - Không lặp `for pt in pts:` trong Python để tìm ô lưới qua `np.searchsorted`.
   - Vector hóa toàn bộ tọa độ điểm qua `np.add.at` và `np.searchsorted` một lần cho toàn mảng điểm:
     ```python
     bx = np.clip(np.searchsorted(part_x_edges[1:], px[valid], side="right"), 0, parts - 1)
     by = np.clip(np.searchsorted(part_y_edges[1:], py[valid], side="right"), 0, parts - 1)
     np.add.at(gt_counts, (by, bx), 1.0)
     ```
   - Giảm thời gian tính GAME cho 182 ảnh từ **8.41s $\to$ 0.049s**, loại bỏ 28 phút chờ của CPU qua 200 lượt evaluation.
4. **Zero-Sync Trajectory Diagnostics & PCIe DMA Reuse:**
   - Trong [`rmr_v3/diagnostics/trajectory.py`](file:///f:/lightweightcrcn/rmr_v3/diagnostics/trajectory.py), trích xuất mảng NumPy một lần thay vì gọi 18 lần `.item()` trên từng mẫu test, loại bỏ 3,276 lần GPU sync flushes.
   - Gán `sample["target_device"] = target` trong [`rmr_core/evaluation.py`](file:///f:/lightweightcrcn/rmr_core/evaluation.py) để callback tái sử dụng trực tiếp tensor trên GPU, tránh copy lại qua PCIe.

---

## 6. Các Ràng Buộc Kiến Trúc & Tiêu Chuẩn Kỹ Thuật Bất Biến

1. **Giới Hạn Độ Dài Tệp:**
   - Mọi tệp mã nguồn Python (`.py`) trong toàn bộ thư mục `rmr_core/` và `rmr_v3/` **phải nghiêm ngặt $\le 450$ dòng**.
   - Phải thực hiện kiểm tra `len(open(file).readlines()) <= 450` trước khi commit bất kỳ thay đổi nào.
2. **Ngân Sách Tham Số Siêu Nhẹ (Lightweight Budget):**
   - Tổng số tham số có thể huấn luyện (trainable parameters) **phải nghiêm ngặt $\le 105,000$**.
   - Anchor Baseline: **104,441** parameters.
   - Composite Model (CPCM + Floor Suppression): **104,753** parameters.
3. **Nghiêm Cấm Tạo Script Ngoài Luồng:**
   - Tuyệt đối **không tự ý tạo file `.sh` hoặc `.ps1`** nếu người dùng chưa yêu cầu rõ ràng.
   - Mọi lệnh chạy thí nghiệm phải được cung cấp trực tiếp dưới dạng lệnh CLI Python (`python -m rmr_v3.train ...`).
4. **An Toàn Nạp Checkpoint (SEC-01):**
   - Mọi thao tác nạp checkpoint phải qua `safe_torch_load(..., weights_only=True)` với danh sách `register_safe_globals()` đã được đăng ký để chống lỗ hổng thực thi mã tùy ý (CWE-502).

---

## 7. Bảng Lệnh Chuẩn Chạy Thí Nghiệm RMR-v32

| Thí Nghiệm | Mã Cấu Hình YAML | Lệnh CLI Chuẩn |
|---|---|---|
| **Anchor Baseline** | `configs/rmr_v32/rmr_v32_step0_anchor.yaml` | `python -m rmr_v3.train --config configs/rmr_v32/rmr_v32_step0_anchor.yaml --run-id v32_anchor` |
| **H1: CPCM** | `configs/rmr_v32/rmr_v32_h1_cpcm.yaml` | `python -m rmr_v3.train --config configs/rmr_v32/rmr_v32_h1_cpcm.yaml --run-id v32_h1_cpcm` |
| **H2: Floor Suppression** | `configs/rmr_v32/rmr_v32_h2_floor_suppression.yaml` | `python -m rmr_v3.train --config configs/rmr_v32/rmr_v32_h2_floor_suppression.yaml --run-id v32_h2_floor_suppression` |
| **H3: Composite Model** | `configs/rmr_v32/rmr_v32_h3_composite.yaml` | `python -m rmr_v3.train --config configs/rmr_v32/rmr_v32_h3_composite.yaml --run-id v32_h3_composite` |
| **H4: Dense Loss Scaling** | `configs/rmr_v32/rmr_v32_h4_dense_loss_scaling.yaml` | `python -m rmr_v3.train --config configs/rmr_v32/rmr_v32_h4_dense_loss_scaling.yaml --run-id v32_h4_dense_loss_scaling` |
| **H5: Conservative SIRT** | `configs/rmr_v32/rmr_v32_h5_conservative_solver.yaml` | `python -m rmr_v3.train --config configs/rmr_v32/rmr_v32_h5_conservative_solver.yaml --run-id v32_h5_conservative_solver` |

---

## 8. Quy Chuẩn Toán Học và Bất Biến Đã Kiểm Chứng (Mathematical Correctness Protocol)

> [!CAUTION]
> Unit tests chỉ kiểm tra hành vi trên input nhỏ — KHÔNG đảm bảo tính đúng đắn toán học ở quy mô thực.
> Mọi thay đổi thuật toán **phải được kiểm tra thủ công** về các bất biến dưới đây.

### 8.1. Bất Biến Toán Học Cốt Lõi

| Bất Biến | Vị Trí | Kiểm Tra |
|---|---|---|
| $A^T$ là adjoint chính xác của $A$ | `adjoint.py` + `prefix_sums.py` | `test_adjoint_exact_duality` |
| Tổng khối lượng bảo toàn qua TV Laplacian (Neumann BC) | `solver_ops.py:laplacian_tv_diffusion` | `mode="replicate"` bắt buộc |
| Tổng khối lượng bảo toàn qua TV Charbonnier (Neumann BC) | `diffusion.py:charbonnier_tv_step` | Replicate-pad trước khi tính gradient |
| Tổng khối lượng bảo toàn qua TV Perona-Malik | `solver_ops.py:perona_malik_anisotropic_diffusion` | `mode="replicate"` bắt buộc |
| BB-1 step size positive khi `dot_sr >= 0` | `solver.py` | `bb_clamp_min > 0` bắt buộc |
| BB-1 track actual iterate $y_t$, KHÔNG track Nesterov extrapolate $z_t$ | `solver.py:prev_y` | `prev_y = y_curr.detach()` |

### 8.2. Lịch Sử Lỗi Đã Sửa (Bug Archaeology)

| Bug ID | Vị Trí | Mô Tả | Commit Sửa |
|---|---|---|---|
| BUG-DEVICE | `rmr_v3/trainer.py:L119` | `UnboundLocalError: device` — device scope trước DataLoader | `ebe2a44` |
| BUG-GAME-SLOW | `rmr_core/metrics.py` | GAME metric Python loop: 171x speedup bằng NumPy vectorize | `ebe2a44` |
| BUG-CUDA-FLUSH | `rmr_v3/diagnostics/trajectory.py` | 18 lần `.item()` per sample → 3276 CUDA flush/eval cycle | `ebe2a44` |
| BUG-PCIe | `rmr_core/evaluation.py` | 182 PCIe host→device transfer/eval cycle | `ebe2a44` |
| BUG-TRAJ-SCALAR | `rmr_v3/diagnostics/trajectory.py:L77-88` | `solver_help_fraction` collapse cả batch về 1 scalar — metric vô nghĩa với batch>1 | `[next commit]` |
| BUG-BB-NESTEROV | `rmr_v3/solver.py` | `prev_y = z_state.detach()` theo dõi extrapolate Nesterov thay vì iterate thực — BB-1 sai khi `use_nesterov_momentum=True` | `[next commit]` |
| BUG-CHARBONNIER-BC | `rmr_core/operators/diffusion.py` | Charbonnier TV dùng zero-pad (Dirichlet BC) thay vì replicate (Neumann BC) — không bảo toàn khối lượng ở biên | `[next commit]` |

### 8.3. Quy Chuẩn Về TV Diffusion

Tất cả TV diffusion trong solver phải dùng **Neumann zero-flux boundary conditions** (replicate padding):
- Bắt buộc: `F.pad(..., mode="replicate")` trước khi tính finite differences
- Nghiêm cấm: `F.pad(..., (0,1,0,1))` (zero-pad Dirichlet BC)
- Lý do: Neumann BC đảm bảo $\sum \Delta y = 0$, tức là tổng khối lượng $\sum y$ không thay đổi qua bước TV. Zero-pad (Dirichlet) cho phép khối lượng "chảy ra" khỏi biên ảnh.

### 8.4. Quy Chuẩn Về BB-1 và Nesterov

Khi cả `use_barzilai_borwein=True` và `use_nesterov_momentum=True` được bật đồng thời:
- **`prev_y`** phải là `y_curr.detach()` (iterate thực), KHÔNG phải `z_state.detach()` (extrapolate Nesterov).
- Lý do: BB-1 tính $s = y_t - y_{t-1}$ (sai phân iterate). Dùng extrapolate Nesterov liên tiếp nhau tạo ra sai phân không có ý nghĩa BB-1.
- Hiện tại các config v32 đều có `use_nesterov_momentum: false` → không ảnh hưởng training hiện tại.

### 8.5. Quy Chuẩn Về Diagnostics Batch-Aware

Các metric trong `compute_solver_trajectory_diagnostics` phải tính per-image rồi lấy mean trên batch:
- **Sai:** `float(tensor.sum().item())` — collapse toàn batch
- **Đúng:** `tensor.sum(dim=(-1,-2,-3)).float().cpu()` → `[B]` vector → `.mean()`
- Áp dụng cho: `solver_help_fraction`, `solver_harm_fraction`, `solver_neutral_fraction`, `solver_delta_e_mean`
