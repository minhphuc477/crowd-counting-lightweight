# RMR Agent Architecture & Developer Guide (Cẩm Nang Kiến Trúc & Ngữ Cảnh Cho AI Agents)

**Phiên bản:** 1.0 (Áp dụng từ RMR-v32 trở đi)  
**Phạm vi:** Toàn bộ hệ thống `rmr_core/`, `rmr_v3/`, `configs/`, `data/`, và các bộ kiểm thử `tests/`.  
**Mục tiêu:** Cung cấp đầy đủ ngữ cảnh lý thuyết, toán học, kiến trúc phần mềm và các bất biến thực nghiệm để các AI Agents trong tương lai phát triển mô hình mà không bao giờ bị quên ngữ cảnh, không tái phạm lỗi cũ và bảo đảm code luôn clean, bảo trì dài hạn.

---

## 1. Tuyên Ngôn Sứ Mệnh & Bài Toán Nghiên Cứu (Mission Statement)

* **Bài toán:** Đếm đám đông siêu nhẹ (Ultra-Lightweight Crowd Counting) với ngân sách tham số **$\le 105,000$ parameters** trên tập chuẩn học thuật **ShanghaiTech Part A**.
* **Nguyên tắc khoa học:** Tự lực cánh sinh hoàn toàn (**Zero Knowledge Distillation**, Zero External Teacher, Zero Pseudo-labels trong các nhánh chính). Mọi đột phá phải xuất phát từ bản chất vật lý và toán học của bài toán nghịch đảo.
* **Cơ sở toán học cốt lõi:** Mô hình hóa đám đông dưới dạng **Bài toán nghịch đảo trên Không gian Độ đo Radon** (Continuous-Discrete Inverse Problem on Radon Measures - Bredies & Pikkarainen 2013, Candes & Fernandez-Granda 2014 Beurling LASSO):
  $$\min_{y \ge 0} \frac{1}{2} \| W^{1/2} (A y - b) \|_2^2 + \lambda_{\text{TV}} \operatorname{TV}(y) + \tau \|y\|_1$$
  Trong đó:
  - $y \in \mathcal{M}_+(\Omega)$ là độ đo mật độ người (continuous measure field).
  - $A$ là toán tử tích phân hình chữ nhật trên các vùng vĩ mô (macro-regions).
  - $b$ là bằng chứng đếm vùng do mạng nơ-ron dự đoán.
  - $W$ là ma trận trọng số độ tin cậy được suy biến từ phương sai phân phối Negative-Binomial.

---

## 2. Các Bất Biến Cốt Lõi Tuyệt Đối Không Được Phá Vỡ (Cardinal Invariants)

Mọi agent khi chỉnh sửa hoặc phát triển module mới **BẮT BUỘC** phải tuân thủ 7 bất biến sau:

### Invariant 1: Phân Vùng Dữ Liệu Chuẩn Mực (Zero Ad-Hoc Split Policy)
* **Quy tắc:** Bắt buộc sử dụng 100% dữ liệu chính thức:
  - **Train:** Đúng **300 ảnh** (`data/sha_a_train_all.jsonl`).
  - **Test / Val:** Đúng **182 ảnh** (`data/sha_a_test.jsonl`).
* **Cấm tuyệt đối:** Không bao giờ tạo hay sử dụng các file chia nhỏ như `sha_a_train.jsonl` (270 ảnh) hay `sha_a_val.jsonl` (30 ảnh). Bất kỳ thí nghiệm nào chạy trên 270 ảnh đều bị coi là invalid.

### Invariant 2: Ngân Sách Tham Số Siêu Nhẹ ($\le 105,000$ Params)
* Tổng số tham số có thể học (`requires_grad=True`) phải **$\le 105,000$**.
* Mức chuẩn hiện tại:
  - Anchor Baseline: **104,441** parameters.
  - Composite Model (CPCM + Floor Suppression): **104,753** parameters.
* Mỗi tham số thêm vào phải có lý giải toán học chính đáng và nằm trong trần ngân sách.

### Invariant 3: Giới Hạn Chiều Dài Tệp Code ($\le 450$ Dòng)
* Mọi tệp mã nguồn Python (`.py`) trong toàn bộ thư mục `rmr_core/` và `rmr_v3/` **phải tuyệt đối $\le 450$ dòng**.
* Trước khi kết thúc turn, agent phải chạy script kiểm tra tự động:
  ```powershell
  python -c "import os; [print(f'{p}: {len(open(p).readlines())}') for r,d,fs in os.walk('rmr_v3') for f in fs if f.endswith('.py') and len(open(os.path.join(r,f)).readlines())>450]"
  ```

### Invariant 4: Nghiêm Cấm Tự Ý Sinh Script Shell / Runner
* **Tuyệt đối không tạo** các file script `.sh`, `.ps1`, `.bat` hay runner suites tự động nếu người dùng chưa ra lệnh rõ ràng.
* Mọi chỉ dẫn chạy thử phải được cung cấp dưới dạng câu lệnh CLI Python trực tiếp (`python -m rmr_v3.train ...`).

### Invariant 5: Triệt Tiêu CUDA Host-Sync Stalls (Zero-Stall Pipeline)
* Không được gọi `.item()`, `.tolist()`, `.cpu()`, hoặc rẽ nhánh boolean (`if tensor > 0:`) bên trong vòng lặp batch huấn luyện `train_one_epoch`.
* `LossTracker` tích lũy tensor trực tiếp trên GPU bằng `v.detach()`.
* `DiagnosticTracker` chỉ lấy mẫu snapshot ở batch cuối cùng của epoch.

### Invariant 6: Bảo Vệ Phân Kỳ Số Học Thuần Tensor (SEC-03)
* Mọi cơ chế xử lý ngoại lệ số học NaN/Inf trong bộ giải SIRT phải sử dụng phép toán tensor thuần túy:
  ```python
  y_next = torch.where(torch.isfinite(y_next), y_next, y_curr)
  ```

### Invariant 7: Nạp Checkpoint An Toàn (SEC-01)
* Sử dụng `safe_torch_load(path, map_location="cpu", weights_only=True)` với danh sách kiểu NumPy an toàn đã đăng ký trong `register_safe_globals()`.

---

## 3. Bản Đồ Module & Trách Nhiệm Kiến Trúc (Architectural Map)

```
lightweightcrcn/
├── rmr_core/                      # Thư viện nền tảng độc lập phiên bản
│   ├── data.py                    # CrowdManifestDataset với RAM Caching (Train & Eval)
│   ├── backbones.py               # MobileNetV4 / Timm pyramid backbone với offline fallback
│   ├── evaluation.py              # evaluate_dataset với async non_blocking DMA & GT check
│   ├── metrics.py                 # Vectorized Bootstrap CI, GAME(0..3), NAE, MAE, RMSE
│   ├── training.py                # safe_torch_load, safe_torch_save, seed_everything
│   ├── types.py                   # Dataclasses & types
│   ├── necks/                     # Additive FPN, RepFPN, BiFPN, Dilated Multi-scale
│   └── operators/                 # Toán tử toán học: prefix2d, regional_sum, regional_adjoint
│
├── rmr_v3/                        # Phiên bản nghiên cứu đỉnh cao hiện tại (RMR-v3 -> v32)
│   ├── model/                     # RMRv3 model class, configuration, heads, CPCM
│   ├── solver.py                  # Unrolled SIRT Solver (BB-1, Morozov, Energy Reuse)
│   ├── solver_ops.py              # Anscombe VST, TV diffusion, proximal shrinkage
│   ├── regional_head.py           # Probabilistic Negative-Binomial Regional Evidence
│   ├── losses/                    # Loss orchestration, dual-supervision, target router
│   ├── engine.py                  # train_one_epoch, evaluate_v3 (Zero-stall loops)
│   ├── tracking.py                # Fast GPU LossTracker, Dynamic DiagnosticTracker
│   ├── trainer.py                 # run_training_loop, CheckpointManager, EMAManager
│   ├── kd.py                      # Density map knowledge distillation
│   └── config/                    # Schema dataclass, config hash, resume validation
│
├── configs/rmr_v32/               # 12 tệp cấu hình thí nghiệm chuẩn hóa v32
├── data/                          # sha_a_train_all.jsonl (300) & sha_a_test.jsonl (182)
└── tests/                         # Bộ kiểm thử toàn diện (>550 tests PASS 100%)
```

---

## 4. Các Bài Học Kinh Nghiệm Sống Còn (Forensic Lessons Learned)

Các agent mới phải đọc kỹ các thất bại lịch sử sau để không lặp lại:

1. **Bẫy Khởi Tạo Bão Hòa (Activation Saturation):**
   - *Bài học:* Khởi tạo $\alpha_0 = -8.0$ khiến $\operatorname{softplus}(-8.0) \approx 3.35 \times 10^{-4}$ với đạo hàm $3.35 \times 10^{-4}$, làm đóng băng hoàn toàn gradient của tham số curvature trong suốt 1000 epoch!
   - *Chuẩn mực:* Luôn khớp giá trị khởi tạo với kỳ vọng tiên nghiệm ($m_0 \approx 0.01576 \implies \text{init} \approx -4.0$).
2. **Bẫy Ngưỡng Mật Độ Đám Đống (Crop vs Full-Image Distribution):**
   - *Bài học:* Đặt ngưỡng dense scaling là 400 dựa trên thống kê toàn ảnh (> 500 người). Tuy nhiên khi train trên crop $512 \times 512$, chỉ có 21.5% số crop đạt trên 400 người, khiến 78.5% dữ liệu không nhận được loss boost!
   - *Chuẩn mực:* Đo đạc trực tiếp trên crop huấn luyện ($p50 = 181, p75 = 357 \implies$ ngưỡng chuẩn là 250).
3. **Bẫy Bỏ Đói GPU (DataLoader & Host Sync Stalls):**
   - *Bài học:* Quên `persistent_workers=True` làm tiêu tốn 4,000 chu kỳ spawn process. Gọi `.item()` ở từng mini-batch làm tăng thời gian epoch từ 2.8 giây lên 30.3 giây.
   - *Chuẩn mực:* Luôn dùng `persistent_workers=True`, `pin_memory=True`, `non_blocking=True`, và RAM caching.
4. **Bẫy Nhân Đôi Trọng Số Loss Trong Phân Cấp (Loss Scaling Orthogonality):**
   - *Bài học:* Outer loop nhân trọng số mẫu $w_i$, inner loss lại nhân tiếp $w_i$, dẫn đến gradient bị khuếch đại thành $w_i^2$, làm sụp đổ hội tụ của mô hình.
   - *Chuẩn mực:* Giữ đúng nguyên tắc đơn nhiệm (Single Responsibility): Điều phối viên vòng ngoài áp dụng trọng số mẫu; hàm loss bên trong giữ nguyên dạng chuẩn.
5. **Bẫy Tin Tưởng Mù Quáng Vào Test Đơn Thuần (The Blind-Test Trap & Vectorized Metrics):**
   - *Bài học:* Unit test với 5 điểm tọa độ pass hoàn toàn, nhưng khi đánh giá trên ảnh thực với 2,000 điểm, vòng lặp `for pt in pts:` trong `game_physical_image` thực thi 1.45 triệu lần lặp Python, làm mỗi lần eval mất 8.41 giây (tổng cộng stall GPU hơn 28 phút qua 200 lần eval)! Tương tự, gọi 18 lần `.item()` trên từng mẫu trong `trajectory.py` gây ra 3,276 lần GPU sync flush.
   - *Chuẩn mực:* Mọi phép tính metric trên mảng tọa độ phải được vector hóa với `np.add.at` / `np.searchsorted` (tăng tốc 171x). Trích xuất chẩn đoán phải dùng bulk array sang NumPy (tăng tốc 3.6x), tuyệt đối không gọi `.item()` trên từng mẫu hoặc từng mini-batch. Luôn tuân thủ kỹ năng `.agents/skills/training-performance-and-engineering-audit/SKILL.md`.

---

## 5. Quy Trình Kiểm Thử Chuẩn Trước Khi Bàn Giao

Mỗi khi hoàn thành một thay đổi trong codebase, agent phải chạy tuần tự các lệnh sau:

```powershell
# 1. Kiểm tra unit test nền tảng và bảo mật
python -m pytest tests/audit/ tests/e2e_v32/ -q

# 2. Kiểm tra bộ test toàn vẹn RMR-v3
python -m pytest tests/rmr_v3/ -q

# 3. Kiểm tra số dòng code (tất cả file .py <= 450 dòng)
python -c "import os; [print(f'ERROR: {os.path.join(r,f)} has {len(open(os.path.join(r,f)).readlines())} lines > 450') for r,d,fs in os.walk('.') if not any(x in r for x in ('.venv','.git','__pycache__','legacy','hpc','tools')) for f in fs if f.endswith('.py') and len(open(os.path.join(r,f)).readlines())>450]"

# 4. Kiểm tra số tham số mô hình (<= 105,000)
python -c "import yaml; from rmr_v3.engine import make_model; cfg=yaml.safe_load(open('configs/rmr_v32/rmr_v32_step0_anchor.yaml')); m,_=make_model(cfg); p=sum(x.numel() for x in m.parameters() if x.requires_grad); assert p<=105000, f'Params {p}>105000'; print(f'Trainable params: {p} (PASS)')"
```
