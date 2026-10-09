# RMR CODEBASE ENTRYPOINTS & OPERATIONAL MANUAL
**Radon Measure Recovery (RMR) Architecture Framework**

Tài liệu này ghi lại toàn bộ các điểm truy cập (Entrypoints), quy ước thực thi CLI, cấu trúc module và cách vận hành chuẩn xác của codebase. Mọi agent, nhà nghiên cứu và kỹ sư bắt buộc phải tham chiếu tài liệu này trước khi chạy bất kỳ câu lệnh nào.

---

## 1. CÁC ENTRYPOINT CHÍNH THỨC (CANONICAL CLI ENTRYPOINTS)

> [!IMPORTANT]
> **Tuyệt đối không có các file giả định như `train_rmr.py` hay `eval_rmr.py` ở root directory!**  
> Mọi tác vụ huấn luyện và đánh giá đều được thực thi qua các module chính thức trong gói `rmr_v3`.

### 1.1 Huấn Luyện Mô Hình (Model Training)
Có 2 cách gọi tương đương 100%:
1. **Dạng module PyTorch (Khuyến nghị chuẩn):**
   ```bash
   python -m rmr_v3.trainer -c <PATH_TO_CONFIG_YAML> [OPTIONS]
   ```
2. **Dạng đường dẫn script trực tiếp:**
   ```bash
   python rmr_v3/trainer.py -c <PATH_TO_CONFIG_YAML> [OPTIONS]
   ```
*(Lưu ý: `rmr_v3/train.py` và `python -m rmr_v3.train` cũng là module hợp lệ, nhưng `rmr_v3/trainer.py` chứa toàn bộ logic điều phối `run_training_loop` cốt lõi).*

#### Các cờ dòng lệnh (CLI Flags) của `rmr_v3.trainer`:
| Cờ Lệnh | Viết Tắt | Kiểu | Mô Tả & Khuyến Nghị Thực Chiến |
| :--- | :---: | :---: | :--- |
| `--config` | `-c` | `str` | **Bắt buộc:** Đường dẫn tới file cấu hình YAML (ví dụ: `configs/rmr_research/rmr_v35_purified_104k.yaml`). |
| `--overwrite` | `-o` | `flag` | **Rất quan trọng:** Tự động xóa và ghi đè thư mục `output_dir` nếu đã tồn tại. Tránh lỗi ném `RuntimeError` khi chạy lại. |
| `--resume` | `-r` | `str` | Đường dẫn checkpoint `.pt` hoặc thư mục run để khôi phục quá trình huấn luyện (tự tìm `last.pt` hoặc `best_val_mae.pt`). |
| `--device` | | `str` | Thiết bị tính toán (ví dụ: `cuda`, `cuda:0`, `cpu`). Mặc định tự phát hiện CUDA. |
| `--output-dir` | | `str` | Ghi đè đường dẫn thư mục lưu trữ kết quả được định nghĩa trong YAML config. |
| `--epochs` | | `int` | Ghi đè tổng số epochs từ YAML config. |
| `--eval-every` | | `int` | Chu kỳ đánh giá validation (ví dụ: `5` epochs một lần). |
| `--workers` | | `int` | Số tiến trình worker DataLoader (khuyến nghị: `4` trên server). |
| `--num-threads` | | `int` | Số luồng CPU intra-op của PyTorch (khuyến nghị: `2` khi chạy song song nhiều runs). |
| `--deterministic` | | `flag` | Bật chế độ tất định nghiêm ngặt (`seed_everything`). |
| `--non-deterministic`| | `flag` | Tắt chế độ tất định để tối ưu tốc độ tối đa qua cuDNN benchmark. |

---

### 1.2 Đánh Giá Mô Hình & Checkpoint (Model Evaluation)
Có 2 cách gọi tương đương 100%:
1. **Dạng module PyTorch (Khuyến nghị chuẩn):**
   ```bash
   python -m rmr_v3.eval --checkpoint <PATH_TO_CHECKPOINT_PT> [OPTIONS]
   ```
2. **Dạng đường dẫn script trực tiếp:**
   ```bash
   python rmr_v3/eval.py --checkpoint <PATH_TO_CHECKPOINT_PT> [OPTIONS]
   ```
*(Lưu ý: `rmr_v3/evaluate.py` là alias bridge trỏ trực tiếp tới `rmr_v3.eval.main`).*

#### Các cờ dòng lệnh (CLI Flags) của `rmr_v3.eval`:
| Cờ Lệnh | Viết Tắt | Kiểu | Mô Tả & Khuyến Nghị Thực Chiến |
| :--- | :---: | :---: | :--- |
| `--checkpoint` | `--ckpt` | `str` | **Bắt buộc:** Đường dẫn file checkpoint `.pt` (ví dụ: `runs/sha_a/rmr_v35_purified_104k/best_val_mae.pt`) hoặc thư mục run. |
| `--manifest` | | `str` | Đường dẫn manifest JSONL đánh giá (mặc định tự lấy `val_manifest` từ config trong checkpoint, e.g. `data/sha_a_test.jsonl`). |
| `--output-dir` | | `str` | Thư mục lưu kết quả báo cáo đánh giá, CSV chi tiết và tóm tắt JSON. |
| `--device` | | `str` | Thiết bị chạy đánh giá (`cuda`, `cuda:0`, `cpu`). |
| `--tta` | `--enable-tta` | `flag` | Kích hoạt Test-Time Augmentation (Horizontal Flip) bảo toàn khối lượng L1. |
| `--no-tiling` | | `flag` | Tắt chế độ chia tile, suy luận trực tiếp toàn bộ ảnh nguyên bản. |
| `--use-live-weights`| | `flag` | Đánh giá trọng số trực tiếp (live weights) thay vì trọng số EMA (mặc định ưu tiên EMA nếu có trong checkpoint). |

---

## 2. BẢN ĐỒ CẤU TRÚC CODEBASE CHI TIẾT (CODEBASE ATLAS)

```text
F:\lightweightcrcn\ (Repository Root)
├── configs/
│   └── rmr_research/       # Các file cấu hình YAML nghiên cứu thực nghiệm
│       ├── rmr_v35_purified_104k.yaml  # Run 1: Baseline 104k thanh lọc (Cosine 1000ep)
│       ├── rmr_v35_scaled_165k.yaml    # Run 2: Mở rộng dung lượng 174k params
│       ├── rmr_v35_wsd_104k.yaml       # Run 3: Baseline 104k với WSD schedule
│       └── rmr_v35_bayesian_104k.yaml  # Run 4: Baseline 104k với Adaptive Bayesian loss
├── data/
│   ├── sha_a_train_all.jsonl # Đúng 300 ảnh huấn luyện ShanghaiTech Part A
│   └── sha_a_test.jsonl      # Đúng 182 ảnh kiểm thử ShanghaiTech Part A
├── rmr_core/                 # 27 tệp lõi toán học và primitives (đã audit <= 386 dòng)
│   ├── data.py               # Dataset, train_transform, padding, half-pixel coordinates
│   ├── operators.py          # Toán tử tích phân hộp A, liên hợp A*, transfer matrix H
│   ├── backbones.py          # MobileNetV4 backbone trích xuất C4, C8, C16
│   ├── necks.py              # HDC-Lite và ASPP-Lite neck (cộng tĩnh 1:1)
│   ├── heads.py              # Fine Carrier Head (z0 -> y0) và Regional Evidence Head
│   ├── training.py           # Tiện ích training: seed, RNG state, save/load an toàn
│   └── evaluation.py         # Hàm evaluate_dataset, tính MAE, RMSE, NAE, Bias
├── rmr_v3/                   # 55 tệp động cơ nghiên cứu và tối ưu hóa
│   ├── trainer.py            # [ENTRYPOINT HUẤN LUYỆN CHÍNH] run_training_loop & main()
│   ├── train.py              # CLI wrapper và arg parsing cho training
│   ├── eval.py               # [ENTRYPOINT ĐÁNH GIÁ CHÍNH] load checkpoint, evaluate & metrics
│   ├── evaluate.py           # CLI bridge alias trỏ sang rmr_v3.eval
│   ├── model.py              # Lớp mạng RMRv3 hợp nhất (Encoder + Neck + Heads + Solver)
│   ├── solver.py             # Bộ giải SIRT không cuộn (T=2) với Barzilai-Borwein & Morozov
│   ├── losses/               # Hệ thống hàm loss đa tầng đã thanh lọc
│   │   ├── config.py         # Cấu hình siêu tham số loss (RMRv3LossConfig)
│   │   ├── orchestration.py  # Hàm compute_rmr_v3_losses điều phối loss
│   │   ├── allocation.py     # Flat DM16 và Dirichlet-Multinomial loss
│   │   ├── point_supervision.py # Bayesian Loss chuẩn hóa điểm Dirac
│   │   └── cell.py           # Cell loss cân bằng mật độ
│   └── optim/                # Bộ điều phối Optimizer và LR Scheduler
│       ├── builder.py        # build_v3_optimizer và build_v3_scheduler
│       ├── wsd_scheduler.py  # Warmup-Stable-Decay scheduler
│       └── safe_prodigy.py   # Khoảng cách thích ứng an toàn
├── tests/                    # 1,176 unit và integration tests (100% passed)
│   ├── rmr_v3/
│   │   ├── test_v35_experimental_suite.py       # Test 4 config thực nghiệm mới
│   │   └── test_v35_purification_and_scaling.py # Test thanh lọc loss & 175k scaling
└── docs/                     # Tài liệu nghiên cứu, pháp y lỗi và quy tắc vận hành
```

---

## 3. CÁC QUY TẮC BẤT BIẾN KHI THỰC THI (OPERATIONAL INVARIANTS)

1. **Tuyệt đối không chạy full training 1000 epochs trên máy cục bộ Windows:**  
   Máy Windows chỉ dùng để viết code, chạy test (`pytest`) và dry-run 1 epoch để kiểm tra tính đúng đắn trước khi commit. Mọi lượt huấn luyện chính thức phải chạy trên server Ubuntu (`tan@tanpc-u`).
2. **Luôn sử dụng cờ `-o` (`--overwrite`) khi khởi động run mới:**  
   Nếu thư mục `output_dir` đã tồn tại do lần chạy thử trước, `rmr_v3.trainer` sẽ chủ động chặn và báo lỗi nếu không có cờ `-o` hoặc `-r`.
3. **Luôn trỏ đúng module với cờ `-m`:**  
   Ví dụ: `python -m rmr_v3.trainer -c configs/... -o`  
   Việc dùng `-m` đảm bảo Python tự động thêm thư mục gốc của repository vào `sys.path`, loại bỏ 100% các lỗi `ModuleNotFoundError` hoặc `ImportError`.
