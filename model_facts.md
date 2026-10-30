# Model Facts: Technical Inventory and Code-Level Ground Truth

Tài liệu này được biên soạn bởi vai trò **Dissector** (Reverse Engineer). Mục đích duy nhất là trích xuất **sự thật kỹ thuật có thể kiểm chứng từ mã nguồn** của repository `lightweightcrcn` tại nhánh `RMR`. Mọi tuyên bố kỹ thuật đều được truy vết trực tiếp đến tệp tin và số dòng cụ thể (`path/to/file.py:Lxx-Lyy`). Tài liệu không sử dụng ngôn từ tiếp thị hoặc suy diễn chủ quan, và phân tách hoàn toàn sự thật trong mã nguồn khỏi các tuyên bố narrative trước đây.

---

## Architecture Table

Bảng liệt kê tất cả các module trong forward pass của mô hình canonical `RMRv3` theo đúng thứ tự thực thi trong mã nguồn (`rmr_v3/model/architecture.py:L243-L334`), cấu hình theo `configs/rmr_canonical/rmr_canonical_104k_seed42.yaml`.

Giả định tensor đầu vào chuẩn: ảnh RGB kích thước $[B, C, H, W] = [1, 3, 512, 512]$.

| Bước | Module / Hàm | Vị trí định nghĩa (File & Dòng) | Tham số học được (Learnable) | Tham số cố định & Buffers | Kích thước Tensor đầu vào | Kích thước Tensor đầu ra |
|---|---|---|---:|---:|---|---|
| 0 | `pad` (Zero-Padding) | `rmr_v3/model/architecture.py:L252-L256` | 0 | 0 | `[1, 3, 512, 512]` | `[1, 3, 512, 512]` (divisible by 16) |
| 1 | `encoder` (`TimmPyramidBackbone`) | `rmr_core/backbones.py:L7-L96` | 87,568 | 3,352 | `[1, 3, 512, 512]` | `c4: [1, 32, 128, 128]`<br>`c8: [1, 32, 64, 64]`<br>`c16: [1, 64, 32, 32]` |
| 1.1 | `encoder.backbone` (`MobileNetV3Features`) | `rmr_core/backbones.py:L55-L91` | 87,568 | 3,352 (BN running stats) | `[1, 3, 512, 512]` | Như trên (cắt tại stage `blocks.2`) |
| 2 | `fusion` (`ASPPLiteFPNNeck`) | `rmr_core/necks/aspp.py:L10-L98` | 10,784 | 0 | `c4, c8, c16` từ encoder | `p4: [1, 32, 128, 128]`<br>`p8: [1, 32, 64, 64]`<br>`p16: [1, 32, 32, 32]` |
| 2.1 | `fusion.lat4` (`ConvGNAct`) | `rmr_core/necks/aspp.py:L45` | 576 | 0 | `[1, 32, 128, 128]` | `l4: [1, 32, 128, 128]` |
| 2.2 | `fusion.lat8` (`ConvGNAct`) | `rmr_core/necks/aspp.py:L46` | 1,088 | 0 | `[1, 32, 64, 64]` | `l8: [1, 32, 64, 64]` |
| 2.3 | `fusion.lat16` (`ConvGNAct`) | `rmr_core/necks/aspp.py:L47` | 1,600 | 0 | `[1, 64, 32, 32]` | `l16: [1, 32, 32, 32]` |
| 2.4 | `fusion.aspp_dw` (`ModuleList`: 3x DW) | `rmr_core/necks/aspp.py:L50-L52` | 1,056 | 0 | `l16: [1, 32, 32, 32]` | 3x `[1, 32, 32, 32]` (dilations 2, 4, 8) |
| 2.5 | `fusion.aspp_gap` (`AdaptiveAvgPool`+`Linear`) | `rmr_core/necks/aspp.py:L54-L61` | 1,056 | 0 | `l16: [1, 32, 32, 32]` | `[1, 32]` -> `[1, 32, 1, 1]` |
| 2.6 | `fusion.aspp_proj` (`ConvGNAct`) | `rmr_core/necks/aspp.py:L62` | 1,088 | 0 | `[1, 32, 32, 32]` | `[1, 32, 32, 32]` |
| 2.7 | `fusion.ref16` (`DSResidual`) | `rmr_core/necks/aspp.py:L65` | 1,440 | 0 | `[1, 32, 32, 32]` | `p16: [1, 32, 32, 32]` |
| 2.8 | `fusion.ref8` (`DSResidual`) | `rmr_core/necks/aspp.py:L66` | 1,440 | 0 | `[1, 32, 64, 64]` | `p8: [1, 32, 64, 64]` |
| 2.9 | `fusion.ref4` (`DSResidual`) | `rmr_core/necks/aspp.py:L67` | 1,440 | 0 | `[1, 32, 128, 128]` | `p4: [1, 32, 128, 128]` |
| 3 | `scale_router` (`ScaleRoutingHead`) | `rmr_core/scale_routing.py:L23-L85` | 483 | 0 | `p4: [1, 32, 128, 128]` | `scale_weights: [1, 3, 128, 128]` |
| 3.1 | `scale_router.dw` (`Conv2d` DW 3x3) | `rmr_core/scale_routing.py:L51-L58` | 320 | 0 | `[1, 32, 128, 128]` | `[1, 32, 128, 128]` |
| 3.2 | `scale_router.norm` (`GroupNorm(8, 32)`) | `rmr_core/scale_routing.py:L59` | 64 | 0 | `[1, 32, 128, 128]` | `[1, 32, 128, 128]` |
| 3.3 | `scale_router.act` (`ReLU`) | `rmr_core/scale_routing.py:L60` | 0 | 0 | `[1, 32, 128, 128]` | `[1, 32, 128, 128]` |
| 3.4 | `scale_router.pw` (`Conv2d` PW 1x1) | `rmr_core/scale_routing.py:L61-L66` | 99 | 0 | `[1, 32, 128, 128]` | `[1, 3, 128, 128]` (softmax across dim 1) |
| 4 | `fine_head` (`FineMeasureHead`) | `rmr_core/heads.py:L25-L200` | 1,473 | 0 | `p4: [1, 32, 128, 128]` | `z0: [1, 1, 128, 128]`<br>`y0: [1, 1, 128, 128]` |
| 4.1 | `fine_head.body[0]` (`ConvGNAct` DW 3x3) | `rmr_core/heads.py:L60` | 352 | 0 | `[1, 32, 128, 128]` | `[1, 32, 128, 128]` |
| 4.2 | `fine_head.body[1]` (`ConvGNAct` PW 1x1) | `rmr_core/heads.py:L61` | 1,088 | 0 | `[1, 32, 128, 128]` | `[1, 32, 128, 128]` |
| 4.3 | `fine_head.body[2]` (`Conv2d` 1x1) | `rmr_core/heads.py:L62` | 33 | 0 | `[1, 32, 128, 128]` | `z0: [1, 1, 128, 128]` |
| 4.4 | `_smooth_floor` (Activation transform) | `rmr_core/density_activation.py:L38-L45` | 0 | 0 | `z0: [1, 1, 128, 128]` | `y0: [1, 1, 128, 128]` (`tau=0.003`) |
| 5 | Region Partitioning (`_regions`) | `rmr_v3/model/architecture.py:L111-L143` | 0 | 0 | `grid_h=128, grid_w=128, stride=4` | `RegionSet`: 1,235 hộp (scales 32, 64, 128 px) |
| 6 | `region_head` (`ProbabilisticRegionalEvidenceHead`) | `rmr_v3/regional_head.py:L31-L270` | 4,082 | 0 | `p4, p8, p16` + `RegionSet` (1,235 hộp) | `mu_count: [1, 1, 1235]`<br>`dispersion: [1, 1, 1235]`<br>`rate: [1, 1, 1235]` |
| 6.1 | `_collect_region_features` | `rmr_v3/regional_head.py:L136-L235` | 0 | 0 | `p4, p8, p16` + `RegionSet` | `pooled: [1, 1235, 33]` (mean pooling + log_scale) |
| 6.2 | `region_head.trunk` (`MLP`: 33->48->48) | `rmr_v3/regional_head.py:L93-L98` | 3,984 | 0 | `[1, 1235, 33]` | `h: [1, 1235, 48]` |
| 6.3 | `region_head.mean_head` (`Linear(48, 1)`) | `rmr_v3/regional_head.py:L100` | 49 | 0 | `[1, 1235, 48]` | `mu_count: [1, 1, 1235]` |
| 6.4 | `region_head.log_dispersion_head` (`Linear(48, 1)`) | `rmr_v3/regional_head.py:L102` | 49 | 0 | `[1, 1235, 48]` | `log_dispersion: [1, 1, 1235]` |
| 6.5 | `reliability_from_nb` | `rmr_v3/regional_head.py:L273-L364` | 0 | 0 | `mu_count, dispersion` | `b_solver: [1, 1, 1235]`<br>`weight_solver: [1, 1, 1235]` |
| 7 | `solver` (`closed_form_mcbpro_transport`) | `rmr_core/operators/mcbpro.py:L45-L221` | 0 | 0 | `y0: [1, 1, 128, 128]`<br>`b_solver, weight_solver: [1, 1, 1235]`<br>`scale_weights: [1, 3, 128, 128]` | `y: [1, 1, 128, 128]`<br>`s_field: [1, 1, 128, 128]` |
| 8 | Model Buffer: `_laplace_kernel` | `rmr_v3/model/architecture.py:L97-L98` | 0 | 9 | `[1, 1, 3, 3]` kernel | Dùng cho carrier energy (nếu bật) |
| **Tổng** | **Toàn bộ mô hình `RMRv3`** | `rmr_v3/model/architecture.py:L40-L106` | **104,390** | **3,361** | `[1, 3, 512, 512]` | `RMRModelOutput` (`y: [1, 1, 128, 128]`) |

### Đối chiếu tham số với các tuyên bố và trần ngân sách (Parameter Reconciliation):
- **Trần ngân sách kiến trúc (Architecture Ceiling)**: `max_trainable_params: 104441` (`configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L44`).
- **Tổng số tham số học được thực tế trong mã nguồn**: $87,568 + 10,784 + 483 + 1,473 + 4,082 = \mathbf{104,390}$ tham số. Độ lệch so với trần ngân sách: $-51$ tham số (nằm hoàn toàn dưới trần).
- **Đối chiếu với tuyên bố "Sub-100k"**: Con số thực tế là $104,390 > 100,000$. Mô hình không thuộc phân khúc sub-100k mà thuộc phân khúc $\le 105k$ (hoặc sub-150k).
- **Tổng số buffer cố định không tối ưu (non-learnable buffers)**: $3,352$ (Batch Normalization running statistics từ backbone MobileNetV4) + $9$ (`_laplace_kernel` cố định) = $3,361$ giá trị float.

---

## Mechanism Audit

Dưới đây là việc kiểm toán từng cơ chế đặc biệt trong codebase, trả lời đầy đủ 5 câu hỏi dựa trên các dòng mã nguồn.

### 1. Toán tử hồi phục độ đo MC-BPRO (`closed_form_mcbpro_transport`)
- **Q1. Có được bật trong cấu hình tạo kết quả chính không?**
  - **Track A** (`configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L79-L80`): **BẬT** (`enable_solver: true`, `solver_mode: mcbpro`).
  - **Track B** (`configs/rmr_canonical/rmr_canonical_stride2_seed42.yaml:L87-L88`): **TẮT** (`enable_solver: false`, `solver_mode: direct`).
- **Q2. Có tham số học được không hay là công thức đóng? Bao nhiêu tham số?**
  - Công thức đóng hoàn toàn (closed-form mathematical operator).
  - Số tham số học được: **0 tham số** (`rmr_core/operators/mcbpro.py:L45-L221`).
- **Q3. Được gọi ở đâu trong forward pass, tần suất bao nhiêu lần mỗi lần suy luận?**
  - Được gọi trong `solve_inverse_measure` (`rmr_v3/model/solver_step.py:L171-L187`), được kích hoạt từ `RMRv3.forward` (`rmr_v3/model/architecture.py:L310-L316`).
  - Tần suất: Đúng **1 lần duy nhất** cho mỗi lần gọi forward inference. Không sử dụng unrolled loop lặp lại.
- **Q4. Nếu tắt đi, mã nguồn có cơ chế fallback không? Fallback là gì?**
  - Có cơ chế fallback rõ ràng tại `rmr_v3/model/solver_step.py:L107-L115`.
  - Fallback: Trả về trực tiếp bản đồ mật độ mịn tiên nghiệm `y = y0`, đặt `solver_strength = 0.0`, `iterates = [y0]`.
- **Q5. Có phụ thuộc vào điều kiện runtime nào không?**
  - Có. Phụ thuộc vào:
    1. Stride và tỷ lệ co giãn (`rmr_v3/model/solver_step.py:L157-L166`): Nếu `is_stride2` là true, `reg_threshold` và `tau_peak` tự động chia 4 (`* 0.25`), và `peak_kernel_size` đổi từ 5 sang 7.
    2. Kích thước lưới $(H, W)$: Yêu cầu `regions.boxes` phải khớp với tọa độ không gian của $y_0$.
    3. Chế độ train/eval: Hành vi toán học không đổi, nhưng tensor làm việc được ép kiểu về `work_dtype` (`rmr_core/operators/mcbpro.py:L88-L91`).

### 2. Hàm mất mát Radon-OT Sliced Wasserstein (`radon_sliced_wasserstein_loss`)
- **Q1. Có được bật trong cấu hình tạo kết quả chính không?**
  - **Track A Canonical** (`configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L101, L110`): **BẬT** (`allocation_loss_type: dual_bayesian_radon_ot`, `lambda_radon_ot: 0.85`).
  - **Ablation 3** (`configs/rmr_canonical/ablations/rmr_abl_no_radon_ot.yaml:L102`): **TẮT** (`lambda_radon_ot: 0.0`).
- **Q2. Có tham số học được không hay là công thức đóng? Bao nhiêu tham số?**
  - Là hàm tính khoảng cách Wasserstein-1 phân tích 1D dạng đóng (analytical 1D Sliced Wasserstein).
  - Số tham số học được: **0 tham số** (`rmr_v3/losses/radon_ot.py:L24, L36-L145`).
- **Q3. Được gọi ở đâu trong forward pass/training, tần suất bao nhiêu lần?**
  - Không nằm trong forward pass suy luận (inference). Chỉ được gọi trong hàm huấn luyện tại `compute_single_allocation` (`rmr_v3/losses/allocation.py:L86-L91`).
  - Tần suất: Gọi **2 lần** trong mỗi bước huấn luyện khi sử dụng giám sát đối ngẫu (`dm_target: dual` giám sát cả $y$ và $y_0$, theo `rmr_v3/losses/allocation.py:L190-L200`), hoặc 1 lần nếu giám sát đơn.
- **Q4. Nếu tắt đi, mã nguồn có cơ chế fallback không? Fallback là gì?**
  - Có. Khi `lambda_radon_ot: 0.0` hoặc `allocation_loss_type: bayesian`, thành phần $l_{rot}$ nhân với 0.0 hoặc bị bỏ qua, toàn bộ trọng số giám sát phân bổ chuyển sang `bayesian_loss` (`rmr_v3/losses/allocation.py:L67-L82`).
- **Q5. Có phụ thuộc vào điều kiện runtime nào không?**
  - Có phụ thuộc:
    1. Số điểm ground-truth $N$: Nếu ảnh không có người ($N=0$), hàm bỏ qua phép chiếu Wasserstein và trả về 0.0 (`rmr_v3/losses/radon_ot.py:L82-L85`).
    2. Số góc chiếu `num_projections`: Cố định 4 góc trong config (`configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L111`).
    3. Stride: Tọa độ điểm chuẩn được chia cho stride (`rmr_v3/losses/radon_ot.py:L121`).

### 3. Hàm mất mát điểm Bayesian (`bayesian_loss`)
- **Q1. Có được bật trong cấu hình tạo kết quả chính không?**
  - **Track A Canonical** (`configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L101, L119`): **BẬT** (`allocation_loss_type: dual_bayesian_radon_ot`, `lambda_bayesian: 0.15`).
  - **Ablation 2** (`configs/rmr_canonical/ablations/rmr_abl_no_bayesian.yaml:L108`): **TẮT** (`lambda_bayesian: 0.0`).
- **Q2. Có tham số học được không hay là công thức đóng? Bao nhiêu tham số?**
  - Công thức xác suất hậu nghiệm Gaussian chuẩn hóa với autograd tùy chỉnh (`_BayesianPersonErrorFunction`).
  - Số tham số học được: **0 tham số** (`rmr_v3/losses/point_supervision.py:L10-L125`).
- **Q3. Được gọi ở đâu trong forward pass/training, tần suất bao nhiêu lần?**
  - Chỉ gọi trong quá trình huấn luyện tại `compute_single_allocation` (`rmr_v3/losses/allocation.py:L69-L80`).
  - Tần suất: Gọi 2 lần mỗi bước train nếu `dm_target: dual` (giám sát $y$ và $y_0$).
- **Q4. Nếu tắt đi, mã nguồn có cơ chế fallback không? Fallback là gì?**
  - Có. Khi `lambda_bayesian: 0.0`, code gán `l_bay = torch.zeros(...)` (`rmr_v3/losses/allocation.py:L81-L82`), toàn bộ mất mát phân bổ phụ thuộc vào Radon-OT.
- **Q5. Có phụ thuộc vào điều kiện runtime nào không?**
  - Có phụ thuộc:
    1. $N=0$ (ảnh không có người): Trả về tổng mật độ dự đoán làm hàm phạt background (`rmr_v3/losses/point_supervision.py:L158-L160`).
    2. `adaptive_sigma`: Tính khoảng cách k-NN để co giãn $\sigma$ từng điểm (`rmr_v3/losses/point_supervision.py:L169-L215`).

### 4. Đầu ước lượng bằng chứng vùng xác suất (`ProbabilisticRegionalEvidenceHead`)
- **Q1. Có được bật trong cấu hình tạo kết quả chính không?**
  - **Track A Canonical** (`configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L58-L68`): **BẬT** (`region_head_hidden: 48`, `native_scale_pooling: true`).
- **Q2. Có tham số học được không hay là công thức đóng? Bao nhiêu tham số?**
  - Mạng nơ-ron học được (Learned Neural Network):
    - `trunk`: 3,984 tham số (`rmr_v3/regional_head.py:L93-L98`).
    - `mean_head`: 49 tham số (`rmr_v3/regional_head.py:L100`).
    - `log_dispersion_head`: 49 tham số (`rmr_v3/regional_head.py:L102`).
  - Tổng số tham số học được: **4,082 tham số**.
- **Q3. Được gọi ở đâu trong forward pass, tần suất bao nhiêu lần?**
  - Được gọi trong `_extract_regional_evidence` (`rmr_v3/model/architecture.py:L292-L300` và `rmr_v3/model/evidence.py:L31-L75`).
  - Tần suất: Đúng **1 lần duy nhất** cho mỗi lần suy luận.
- **Q4. Nếu tắt đi, mã nguồn có cơ chế fallback không? Fallback là gì?**
  - Không có cờ tắt trực tiếp trong kiến trúc RMRv3 chuẩn, vì module này là nguồn cung cấp $b_{solver}$ cho solver. Nếu `enable_solver: false` (như Track B), đầu này vẫn chạy trong forward để cung cấp độ đo vùng giám sát cho loss `lambda_region_nb`, nhưng đầu ra $b_{solver}$ không được truyền vào solver (`rmr_v3/model/solver_step.py:L107-L115`).
- **Q5. Có phụ thuộc vào điều kiện runtime nào không?**
  - Có. Phụ thuộc vào danh sách vùng `RegionSet` (được cache theo kích thước ảnh $(H, W)$ tại `rmr_v3/model/architecture.py:L120-L143`). Nếu kích thước ảnh thay đổi, cache sinh lại hộp vùng.

### 5. Đầu điều hướng tỷ lệ không gian (`ScaleRoutingHead`)
- **Q1. Có được bật trong cấu hình tạo kết quả chính không?**
  - **Track A Canonical** (`configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L69-L71`): **BẬT** (`dynamic_scale_routing: true`, `scale_router_temperature: 1.0`).
- **Q2. Có tham số học được không hay là công thức đóng? Bao nhiêu tham số?**
  - Mạng nơ-ron tích chập học được:
    - Depthwise conv: 320 tham số (`rmr_core/scale_routing.py:L51-L58`).
    - GroupNorm: 64 tham số (`rmr_core/scale_routing.py:L59`).
    - Pointwise conv: 99 tham số (`rmr_core/scale_routing.py:L61-L66`).
  - Tổng số tham số học được: **483 tham số**.
- **Q3. Được gọi ở đâu trong forward pass, tần suất bao nhiêu lần?**
  - Gọi tại `_route_scales` trong `RMRv3.forward` (`rmr_v3/model/architecture.py:L262-L264`).
  - Tần suất: Đúng **1 lần duy nhất** mỗi lần suy luận.
- **Q4. Nếu tắt đi, mã nguồn có cơ chế fallback không? Fallback là gì?**
  - Có. Nếu `dynamic_scale_routing: false`, builder trả về `None` (`rmr_v3/model/builder.py:L126`). Trong `_route_scales`, trả về `scale_weights = None` (`rmr_v3/model/architecture.py:L170`). Khi đó fine head và solver áp dụng trọng số đồng nhất (uniform weighting across scales) (`rmr_core/operators/mcbpro.py:L39-L42`).
- **Q5. Có phụ thuộc vào điều kiện runtime nào không?**
  - Không phụ thuộc vào kích thước động (hoạt động tích chập 100% fully-convolutional). Không phụ thuộc train/eval (sử dụng GroupNorm độc lập batch size).

### 6. Khử sương mù nền bậc hai trơn C1 (`_smooth_floor`)
- **Q1. Có được bật trong cấu hình tạo kết quả chính không?**
  - **Track A Canonical** (`configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L95`): **BẬT** (`floor_tau: 0.003`).
- **Q2. Có tham số học được không hay là công thức đóng? Bao nhiêu tham số?**
  - Công thức đóng (spline bậc hai $C^1$).
  - Số tham số học được: **0 tham số** (`rmr_core/density_activation.py:L38-L45`).
- **Q3. Được gọi ở đâu trong forward pass, tần suất bao nhiêu lần?**
  - Được gọi bên trong `_density_activate` (`rmr_core/density_activation.py:L115-L116`) bởi `FineMeasureHead.activate` (`rmr_core/heads.py:L163-L179`), và trong `ProbabilisticRegionalEvidenceHead.forward` (`rmr_v3/regional_head.py:L247-L249`).
  - Tần suất: 1 lần cho bản đồ mật độ mịn $y_0$ và 1 lần cho ước lượng tỷ lệ vùng.
- **Q4. Nếu tắt đi, mã nguồn có cơ chế fallback không? Fallback là gì?**
  - Khi `floor_tau: 0.0`, hàm trả về trực tiếp giá trị $y_{base}$ không sửa đổi (`rmr_core/density_activation.py:L41-L42`).
- **Q5. Có phụ thuộc vào điều kiện runtime nào không?**
  - Hoạt động point-wise trên từng phần tử tensor, không phụ thuộc vào kích thước hay chế độ chạy.

---

## Reproducibility Check

Phân tích tính khả tái lập dựa trên việc đối chiếu cấu hình chạy và mã nguồn huấn luyện.

### 1. Cơ chế cố định Seed
- **Mã nguồn thực thi**: `rmr_v3/trainer.py:L40-L48` gọi `seed_everything(seed, deterministic=deterministic)` từ `rmr_core/reproducibility.py:L11-L35`.
- **Cấu hình thực tế**: Trong tệp `configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L153`:
  ```yaml
  deterministic: false
  ```
- **Hệ quả kỹ thuật trong mã nguồn**:
  Khi `deterministic: false`:
  1. `torch.backends.cudnn.deterministic = False` (`rmr_core/reproducibility.py:L29`).
  2. `torch.backends.cudnn.benchmark = True` (`rmr_core/reproducibility.py:L30` và `rmr_v3/trainer.py:L168`).
  3. `torch.use_deterministic_algorithms(False)` (`rmr_core/reproducibility.py:L32`).
  4. Trình tạo số ngẫu nhiên của DataLoader: `gen = None` (`rmr_v3/trainer.py:L171`).

### 2. Hành vi DataLoader và Xáo trộn Dữ liệu (Data Shuffling)
- **Mã nguồn thực thi**: `rmr_v3/trainer.py:L172-L184`:
  ```python
  gen = make_generator(seed) if deterministic else None
  train_loader = DataLoader(
      train_ds,
      batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
      shuffle=True,
      num_workers=workers,  # workers: 4 trong config
      pin_memory=pin_mem,
      persistent_workers=bool(workers > 0),
      collate_fn=collate_train,
      drop_last=True,
      worker_init_fn=seed_worker,
      generator=gen,
  )
  ```
- **Sự thật từ mã nguồn**: Vì `deterministic: false`, biến `gen` bằng `None`. Dù `seed_worker` được cung cấp (`rmr_v3/trainer.py:L182`), việc thiếu generator độc lập cho DataLoader kết hợp với 4 worker tiến trình bất đồng bộ (`workers: 4`) và `cudnn.benchmark = True` khiến thứ tự nạp batch và phân bổ tác vụ giữa các epoch có yếu tố dao động thời gian thực (timing/scheduling nondeterminism).

### 3. Các phép toán phụ thuộc thời điểm chạy và phần cứng (Hardware & Precision Nondeterminism)
- **TensorFloat-32 (TF32)**:
  `rmr_v3/trainer.py:L165-L166`:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```
  Việc cho phép TF32 trên các GPU kiến trúc Ampere/Ada (ví dụ RTX 3090, 4090, A100) làm giảm độ chính xác phần định trị (mantissa) từ 23 bit xuống 10 bit trong các phép nhân ma trận và tích chập, dẫn đến sai số làm tròn tích lũy không xác định giữa các luồng xử lý song song.
- **Tăng cường dữ liệu ngẫu nhiên (Stochastic Data Augmentations)**:
  Trong `rmr_core/data.py:CrowdManifestDataset`:
  - `RandomErasing` (`random_erasing_prob: 0.35`, `configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L31`).
  - Biến đổi tỷ lệ ảnh `scale_range: [0.60, 1.40]`.
  - Biến đổi độ sáng, tương phản, độ bão hòa, gamma.

### 4. Bằng chứng thực nghiệm về độ phân tán liên-seed (Cross-Seed Variance)
Cùng một cấu hình mã nguồn và siêu tham số, chạy trên 2 seed khác nhau cho kết quả thực tế như sau:

| Chỉ số đánh giá | Seed 42 (`rmr_canonical_104k_seed42`) | Seed 7 (`rmr_canonical_104k_seed7`) | Mức chênh lệch ($\Delta = |\text{S42} - \text{S7}|$) | Trung bình $\pm$ Độ lệch chuẩn |
|---|---:|---:|---:|---:|
| **Test MAE (Uniform)** | 65.62 | 72.20 | **+6.58** | **68.91 $\pm$ 4.65** |
| **Test MAE (TTA)** | 65.69 | 72.72 | **+7.03** | **69.21 $\pm$ 4.97** |
| **Test RMSE** | 109.16 | 125.45 | **+16.29** | **117.31 $\pm$ 11.52** |
| **Test Bias** | -19.45 | -24.87 | **-5.42** | **-22.16 $\pm$ 3.83** |
| **Sparse MAE** | 15.63 | 36.40 | **+20.77** | **26.02 $\pm$ 14.69** |
| **Dense MAE** | 127.67 | 145.84 | **+18.17** | **136.76 $\pm$ 12.85** |

*(Nguồn dữ liệu: `runs/canonical/rmr_canonical_104k_seed42/eval_test_uniform/summary.json` và `runs/canonical/rmr_canonical_104k_seed7/eval_test_uniform/summary.json`)*.

**Kết luận về tính khả tái lập**: Mức độ nhạy cảm với seed là $\Delta = 6.58$ MAE. Không thể lấy một lần chạy đơn lẻ (Seed 42) để đại diện cho hiệu năng tổng thể của mô hình.

---

## Claim-vs-Code Diff

Bảng đối chiếu toàn diện giữa các tuyên bố trong các báo cáo / tài liệu trước đây và sự thật kỹ thuật kiểm chứng được từ mã nguồn và dữ liệu chạy thực nghiệm.

| # | Tuyên bố trong tài liệu / báo cáo trước | Mã nguồn và dữ liệu thực tế tương ứng | Kết luận Dissector |
|---|---|---|---|
| 1 | *"Kỷ lục thế giới phân khúc sub-100k params (World record sub-100k: 65.62 MAE)"* | - Số tham số học được thực tế là **104,390 params** (`rmr_v3/model/architecture.py:L101-L106`, `configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L5`). $104,390 > 100,000$.<br>- Trong văn hiến khoa học, các mô hình phân khúc nhỏ hơn đã đạt kết quả vượt trội: **EBC-ZIP-S (84k params, 57.2 MAE)**, **FFNet (48.3 MAE)**, **HMSTUNet (49.1 MAE)**. | **KHÔNG KHỚP**<br>(Mô hình vượt quá 100k params; kết quả kém 11.71 MAE so với SOTA sub-100k). |
| 2 | *"Đánh bại SANet (67.0) và sánh ngang TEDnet (64.2)"* | - Seed 42 đạt 65.62 MAE (`summary.json:L2`).<br>- Seed 7 đạt 72.20 MAE (`summary.json:L2`).<br>- Giá trị trung bình 2 seed là **68.91 MAE**.<br>- $68.91 > 67.0$ (kém SANet) và $68.91 > 64.2$ (kém TEDnet). | **KHÔNG KHỚP TRÊN TRUNG BÌNH**<br>(Tuyên bố cherry-picking trên Seed 42 đơn lẻ; trung bình kém cả SANet và TEDnet). |
| 3 | *"Ablation 3 chứng minh tại sao cần Radon-OT (loại bỏ Radon-OT làm hỏng mô hình)"* | - Ablation 3 (`rmr_abl_no_radon_ot`, `lambda_radon_ot: 0.0`) đạt **65.74 MAE**, **108.27 RMSE**, **123.74 Dense MAE** (`eval_val/summary.json:L2-L31`).<br>- Mô hình Full (Seed 42) đạt **65.62 MAE**, **109.16 RMSE**, **127.67 Dense MAE**.<br>- Ablation 3 có MAE tương đương ($\Delta = 0.12$), nhưng **RMSE tốt hơn** (108.27 vs 109.16) và **Dense MAE tốt hơn** (123.74 vs 127.67). Điểm duy nhất Radon-OT cải thiện là Sparse MAE (15.63 vs 26.23) và Bias (-19.45 vs -24.80). | **KHÔNG KHỚP VỚI NARRATIVE**<br>(Mã nguồn và số liệu chứng minh Radon-OT không cải thiện MAE tổng hay RMSE, chỉ cải thiện bias vùng thưa). |
| 4 | *"Định lý Radon Null-Space Degeneracy (Radon-OT thuần túy bị suy biến không gian hạt nhân)"* | - Tuyên bố dựa trên 1 lần chạy đơn lẻ của Ablation 2 (`rmr_abl_no_bayesian`, `lambda_bayesian: 0.0`) đạt 79.97 MAE (`eval_val/summary.json:L2`).<br>- Không có định lý hình thức, bổ đề, hoặc chứng minh toán học nào được viết trong mã nguồn hay phụ lục toán học. | **KHÔNG KIỂM CHỨNG ĐƯỢC TỪ TOÁN HỌC**<br>(Chỉ là một quan sát thực nghiệm từ đúng 1 seed đơn lẻ; gọi là định lý là sai lệch học thuật). |
| 5 | *"MC-BPRO là toán tử hồi phục độ đo closed-form 0 tham số O(1)"* | - Định nghĩa tại `rmr_core/operators/mcbpro.py:L45-L221`.<br>- Toàn bộ hàm thực hiện các phép chiếu đại số tensor (tích phân hộp, trần deadband, chiếu Poisson dạng đóng).<br>- Không chứa bất kỳ `nn.Parameter` nào (0 tham số học được). Không lặp solver lặp (O(1) graph complexity). | **KHỚP**<br>(Khẳng định hoàn toàn chính xác theo mã nguồn). |
| 6 | *"Track B Stride 2 là Direct Neural Operator không cần solver"* | - Cấu hình tại `configs/rmr_canonical/rmr_canonical_stride2_seed42.yaml:L87-L88`: `enable_solver: false`, `solver_mode: direct`.<br>- Trong mã nguồn `rmr_v3/model/solver_step.py:L107-L115`: trả về trực tiếp `y = y0`.<br>- Kết quả thực nghiệm Seed 42 đạt **67.20 MAE**, **110.70 RMSE**, **Bias -10.91** (`eval_val/summary.json:L2-L5`). | **KHỚP TRÊN SEED 42**<br>(Mã nguồn hoàn toàn bỏ solver và thực nghiệm đạt kết quả cạnh tranh). |
| 7 | *"Hệ thống hoàn toàn bảo vệ được trước mọi hội đồng (Bulletproof Defense)"* | - Báo cáo cũ loại bỏ hoàn toàn số liệu Seed 7 (72.20) và giấu việc Ablation 3 có RMSE tốt hơn Full.<br>- Bất kỳ reviewer nào yêu cầu nạp log, mã nguồn, hoặc chạy multi-seed sẽ phát hiện sự không đồng nhất giữa narrative và dữ liệu. | **KHÔNG KHỚP**<br>(Tuyên bố không có cơ sở thực tế; tiềm ẩn nguy cơ quy kết trình bày sai lệch - misrepresentation). |

---

## Dependency Map

Sơ đồ quan hệ phụ thuộc giữa các module trong forward pass và training pipeline. Sơ đồ phân định rõ:
- `[M]`: Module bắt buộc (Mandatory) — không thể tắt mà không thay đổi bản chất luồng dữ liệu.
- `[O]`: Module tùy chọn (Optional / Toggable) — có cờ bật/tắt trong config kèm cơ chế fallback.

```
                    Input Image x [B, 3, H, W]
                                |
                                v
               +----------------------------------+
               | [M] Zero-Padding (divisible by 16)| (architecture.py:L252)
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | [M] TimmPyramidBackbone (MNv4)   | (backbones.py:L7)
               |     Params: 87,568               |
               +----------------------------------+
                                |
                    +-----------+-----------+
                    | c4        | c8        | c16
                    v           v           v
               +----------------------------------+
               | [M] ASPPLiteFPNNeck              | (aspp.py:L10)
               |     Params: 10,784               |
               +----------------------------------+
                                |
                    +-----------+-----------+
                    | p4        | p8        | p16
                    |           |           |
        +-----------+           |           |
        |                       |           |
        v                       |           |
+----------------------+        |           |
| [O] ScaleRoutingHead |        |           |
|     Params: 483      |        |           | (scale_routing.py:L23)
+----------------------+        |           |
        |                       |           |
        | scale_weights         |           |
        v                       |           |
+----------------------+        |           |
| [M] FineMeasureHead  |<-------+           | (heads.py:L25)
|     Params: 1,473    |                    |
+----------------------+                    |
        |                                   |
        | z0, y0                            |
        v                                   |
+----------------------+                    |
| [M] Multiscale       |                    |
|     Region Partition |                    | (architecture.py:L111)
|     (RegionSet: 1235)|                    |
+----------------------+                    |
        |                                   |
        +-------------------+---------------+
                            |
                            v
               +----------------------------------+
               | [M] ProbabilisticRegionalEvidence| (regional_head.py:L31)
               |     Params: 4,082                |
               +----------------------------------+
                            |
                            | b_solver, weight_solver (derived via reliability_from_nb)
                            v
               +----------------------------------+
               | [O] MC-BPRO Transport Solver     | (mcbpro.py:L45)
               |     Params: 0 (Closed-Form)      |
               |     Fallback: y = y0             | (solver_step.py:L107)
               +----------------------------------+
                            |
                            v
                    RMRModelOutput (y, y0)
                            |
           +----------------+----------------+
           | (Inference: y)                  | (Training: y, y0)
           v                                 v
   Final Prediction               +----------------------------------+
   Count = sum(y)                 | [O] Dual-Allocation Loss Router  | (allocation.py:L66)
                                  +----------------------------------+
                                             |
                         +-------------------+-------------------+
                         |                                       |
                         v                                       v
        +---------------------------------+     +---------------------------------+
        | [O] Bayesian Point Loss         |     | [O] Radon-OT SWD Loss           |
        |     Params: 0                   |     |     Params: 0 (Analytical 1D)   |
        |     Fallback: l_bay = 0         |     |     Fallback: l_rot = 0         |
        |     (point_supervision.py:L121) |     |     (radon_ot.py:L36)           |
        +---------------------------------+     +---------------------------------+
```

### Cơ chế Fallback khi tắt các module tùy chọn (`[O]`):
1. **`scale_router` bị tắt (`dynamic_scale_routing: false`)**:
   - `build_scale_router` trả về `None` (`rmr_v3/model/builder.py:L126`).
   - `_route_scales` trả về `scale_weights = None` (`rmr_v3/model/architecture.py:L170`).
   - Fallback: `fine_head` và solver tự động chuyển về cơ chế adjoint đồng nhất không theo tỷ lệ (`rmr_core/operators/mcbpro.py:L39-L42`).
2. **`solver` bị tắt (`enable_solver: false` hoặc `solver_mode: "direct"`)**:
   - `solve_inverse_measure` bắt điều kiện ở dòng đầu (`rmr_v3/model/solver_step.py:L107-L115`).
   - Fallback: Trả về trực tiếp bản đồ mật độ mịn $y = y_0$, bỏ qua hoàn toàn việc tính toán bù trừ độ đo. Đây chính là cấu hình vận hành của Track B Stride 2.
3. **`radon_ot` bị tắt (`lambda_radon_ot: 0.0`)**:
   - Trong `compute_single_allocation` (`rmr_v3/losses/allocation.py:L89-L91`), $l_{rot}$ nhân với hệ số 0.0.
   - Fallback: Toàn bộ gradient phân bổ không gian được dẫn truyền bởi `bayesian_loss` ($l_{bay}$). Đây chính là cấu hình của Ablation 3.
4. **`bayesian` bị tắt (`lambda_bayesian: 0.0`)**:
   - Trong `compute_single_allocation` (`rmr_v3/losses/allocation.py:L81-L82`), $l_{bay} = 0$.
   - Fallback: Toàn bộ gradient phân bổ không gian phụ thuộc vào $l_{rot}$ của Radon-OT. Đây chính là cấu hình của Ablation 2.
