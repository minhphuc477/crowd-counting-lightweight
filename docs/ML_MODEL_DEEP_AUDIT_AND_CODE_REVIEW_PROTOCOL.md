# Giao Thức Chuẩn Quốc Tế: Deep Architectural Audit & Systematic Code Review Trong Machine Learning & Computer Vision

> **Tiêu chuẩn học thuật & công nghiệp tích hợp:** Google SE4ML (Breck et al.), Andrej Karpathy's Neural Net Recipe, Meta FAIR Engineering Standards, NeurIPS Reproducibility Checklist, OpenAI Research Guides.

---

## 1. Bối Cảnh & Sự Khác Biệt Nền Tảng: SE vs. ML Code Review

Trong kỹ nghệ phần mềm truyền thống (Software Engineering - SE), một lỗi lập trình thường biểu hiện công khai và rõ ràng (**Explicit Failure**): ném ra ngoại lệ (`TypeError`, `IndexError`), làm sập chương trình, hoặc làm fail ngay lập tức các bài test đơn vị (Unit Tests).

Ngược lại, trong Machine Learning và Computer Vision, hầu hết các lỗi nghiêm trọng lại là **Lỗi Im Lặng (Silent Failures)**:
- Code thực thi trơn tru, đồ thị Autograd không báo lỗi.
- Loss function vẫn giảm đều qua các epoch.
- Mô hình có vẻ đang "học", nhưng thực chất đang học các tương quan giả tạo (Spurious Correlations), giải ra các nghiệm suy biến (Degenerate Solutions), hoặc bị sụp đổ không gian biểu diễn (Representation Collapse).

### Bảng So Sánh Chi Tiết

| Đặc Tính | Software Engineering Truyền Thống | Deep Learning & Computer Vision Systems |
| :--- | :--- | :--- |
| **Bản Chất Lỗi** | **Fail-Fast**: Ném ngoại lệ, dừng chương trình ngay khi phát sinh lỗi logic. | **Silent Failure**: Code chạy êm, loss giảm nhưng gradient triệt tiêu, ma trận suy biến. |
| **Tính Tất Định** | **Deterministic**: Cùng input luôn sinh ra cùng output $f(x) == y$. | **Stochastic & Numerical**: Bị ảnh hưởng bởi CUDA non-deterministic kernels, floating point roundoff. |
| **Độ Phủ Kiểm Thử** | **Branch Coverage**: Phủ hết các nhánh `if/else` là đảm bảo an toàn. | **Manifold Coverage**: 100% test pass vẫn có thể train một model hỏng nếu tensor tự động broadcast sai. |
| **Công Cụ Phát Hiện** | Static Linters (Flake8, Pylint, Black), Static Type Checkers (Mypy). | Mathematical Audit, Invariant Assertions, Spectral Diagnostics, Gradient Surgery. |

---

## 2. Hệ Thống Phân Cấp Mức Độ Nghiêm Trọng (Severity Hierarchy)

Mọi đánh giá mã nguồn ML/DL phải phân loại lỗi theo 4 cấp độ nghiêm trọng nghiêm ngặt:

```
[SEVERITY HIERARCHY]
├── BLOCKER (P0)  : Phá hủy tính đúng đắn toán học hoặc rò rỉ dữ liệu (Vô hiệu hóa kết quả nghiên cứu)
├── CRITICAL (P1) : Làm suy biến hội tụ, nứt gãy đồ thị Autograd, bất ổn số học FP16/BF16
├── MAJOR (P2)    : Gây nghẽn công suất mô hình (Bottleneck), xung đột gradient đa nhiệm, lệch receptive field
└── MINOR/NIT (P3): Thiếu kiểm soát định dạng tensor, thiếu assertion kích thước, code style
```

### 2.1. Nhóm P0: Blockers (Từ Chối Hợp Nhất Ngay Lập Tức)
1. **Rò rỉ dữ liệu (Data Leakage)**:
   - Tính toán giá trị chuẩn hóa (Mean, Std, Min, Max, Quantiles) trên toàn bộ dataset trước khi chia tách tập Train / Val / Test. Thống kê bắt buộc phải được fit độc quyền trên tập Train.
   - Các vùng cắt (Spatial Crops) của cùng một bức ảnh gốc hoặc cùng một camera góc nhìn xuất hiện đồng thời ở cả Train và Test.
2. **Silent Tensor Broadcasting Trap**:
   - Hai tensor khác biệt về mặt ngữ nghĩa toán học tự động ép kiểu theo luật trailing dimensions của PyTorch (ví dụ: `[B, 1] - [M]` sinh ra `[B, M, M]` thay vì bắn lỗi).
3. **Vi Phạm Tính Độc Lập Mẫu (Batch Independence Violation)**:
   - Dự đoán của mẫu $x_i$ khi chạy đơn lẻ ($B=1$) bị thay đổi khi chạy chung trong batch ($B > 1$).
4. **Vi Phạm Bảo Toàn Khối Lượng Đo (Mass Conservation Invariant)**:
   - Trong bài toán hồi quy mật độ hoặc đếm đám đông: Phóng to hoặc thu nhỏ ảnh mà không điều chỉnh mật độ theo hệ số bình phương diện tích ô lưới ($S^{-2}$).

### 2.2. Nhóm P1: Critical Issues (Bắt Buộc Khắc Phục Trước Khi Huấn Luyện)
1. **Lỗ Hổng Số Học & Bất Ổn Float16/AMP**:
   - Dùng `torch.sqrt(x)` không có epsilon tại điểm $x=0$: $\frac{d}{dx}\sqrt{0} = \frac{1}{2\sqrt{0}} = +\infty \implies \text{NaN}$. Bắt buộc dùng `torch.sqrt(x + 1e-8)` hoặc `torch.clamp_min(x, 1e-12)`.
   - Dùng `torch.log(x)` không có clamping: Tại $x \le 0 \implies -\infty \implies \text{NaN}$. Bắt buộc dùng `torch.log(torch.clamp_min(x, 1e-12))` hoặc `torch.log1p(x)`.
   - Phép chia cho biến mẫu số mà không có `clamp_min(eps)` hoặc `+ eps`.
2. **Rò Rỉ Đồ Thị Autograd & Biến Đổi In-Place**:
   - Biến đổi in-place (`+=`, `.add_()`) trên tensor được lưu để tính đạo hàm backward.
   - Quên gọi `.detach()` giữa các bước lặp của unrolled optimization solver, làm đồ thị tính toán phình to vô hạn và bùng nổ gradient qua thời gian.
   - Gọi nhầm `.detach()` trên intermediate feature representation, làm đóng băng toàn bộ backbone mà không hề có cảnh báo.
3. **Lệch Pha Chế Độ Mô Hình (Model Mode Misalignment)**:
   - Đánh giá hoặc suy luận mà không gọi `model.eval()`, khiến `BatchNorm` cập nhật sai thống kê từ tập test và `Dropout` tiếp tục ngắt kết nối ngẫu nhiên.
   - Quên bọc pha inference bằng `with torch.no_grad():` hoặc `with torch.inference_mode():`.

### 2.3. Nhóm P2: Major Architectural Defects (Điểm Nghẽn Kiến Trúc)
1. **Xung Đột Gradient Đa Nhiệm (Gradient Conflict / Negative Transfer)**:
   - Góc Cosine giữa gradient của các hàm mất mát thành phần âm: $\cos(\theta) = \frac{\langle g_1, g_2 \rangle}{\|g_1\| \|g_2\|} < 0$.
   - Mất cân bằng độ lớn gradient: $\|g_1\| > 10^3 \|g_2\|$, triệt tiêu hoàn toàn nhánh phụ.
2. **Sụp Đổ Chiều Biểu Diễn (Representation Rank Collapse)**:
   - Effective Rank $\text{erank}(F) \ll D$ tại các tầng nghẽn (bottlenecks), chứng minh phần lớn số kênh bị tê liệt.
3. **Lệch Vùng Nhìn Hiệu Dụng (Receptive Field Mismatch)**:
   - Vùng nhìn thực tế (ERF) nhỏ hơn đáng kể so với kích thước vật thể cần nhận diện, hoặc hiệu ứng kẻ sọc (Gridding Artifacts) do dilation quá lớn không có cascade bù trừ.
4. **Lệch Nửa Pixel (Half-Pixel Coordinate Misalignment)**:
   - Không bù trừ tọa độ tâm ô lưới $(i + 0.5, j + 0.5)$ khi ánh xạ giữa tọa độ điểm liên tục và ô pixel rời rạc.

---

## 3. Các Công Cụ Chẩn Đoán Nội Tại Mô Hình (Intrinsic Diagnostics)

### 3.1. Phổ SVD & Effective Rank (`erank`)
Cho ma trận đặc trưng $F \in \mathbb{R}^{N \times D}$ tại bottleneck layer (với $N = B \times H \times W$ điểm ảnh, $D$ số kênh):
1. Khử kỳ vọng: $\tilde{F} = F - \mu_F$.
2. Phân tích giá trị kỳ dị (SVD): $\tilde{F} = U \Sigma V^\top \implies \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_D \ge 0$.
3. Phân phối năng lượng phổ: $p_k = \frac{\sigma_k}{\sum_{j=1}^D \sigma_j}$.
4. Shannon Entropy của phổ: $H(p) = - \sum_{k=1}^D p_k \ln(p_k)$.
5. Effective Rank:
   $$\text{erank}(F) = \exp(H(p)) \in [1, D].$$

**Tiêu Chuẩn Đánh Giá:**
- **Tối ưu:** $\text{erank}(F) \ge 0.35 D$.
- **Cảnh báo suy biến:** $\text{erank}(F) < 0.20 D$.
- **Sụp đổ hoàn toàn (Rank Collapse):** $\text{erank}(F) < 0.10 D$ (ví dụ mạng có 32 channels nhưng chỉ hoạt động trên không gian con 2-3 chiều).

### 3.2. Effective Receptive Field (ERF)
Độ nhạy gradient của một điểm ảnh trung tâm $y_{\text{center}}$ đối với các điểm ảnh đầu vào $x_{i,j}$:
$$\text{ERF}(i, j) = \left| \frac{\partial y_{\text{center}}}{\partial x_{i,j}} \right|.$$
Theo định lý giới hạn trung tâm (Luo et al., NeurIPS 2016), ERF hội tụ về một hàm Gauss 2D:
$$\text{ERF}(i, j) \propto \exp\left( -\frac{i^2 + j^2}{2 \sigma_{\text{eff}}^2} \right).$$
Năng lượng hiệu dụng tập trung chủ yếu trong bán kính $2\sigma$ ($95.4\%$ năng lượng), nhỏ hơn nhiều so với Theoretical Receptive Field (TRF) hình chữ nhật.

### 3.3. Đo Góc Cosine & Phẫu Thuật Gradient (PCGrad Protocol)
Khi tối ưu đồng thời $\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \lambda \mathcal{L}_2$:
- Vector gradient: $g_1 = \nabla_\theta \mathcal{L}_1$, $g_2 = \nabla_\theta \lambda \mathcal{L}_2$.
- Góc xung đột:
  $$\cos(\theta) = \frac{\langle g_1, g_2 \rangle}{\|g_1\|_2 \|g_2\|_2}.$$
- **Nếu $\cos(\theta) < 0$**: Hai hàm loss đang kéo trọng số ngược chiều nhau. Cập nhật theo $g_1$ trực tiếp làm tăng giá trị $\mathcal{L}_2$.
- **Giải pháp PCGrad (Yu et al., NeurIPS 2020)**: Chiếu trực giao vector gradient:
  $$g_1' = g_1 - \frac{\langle g_1, g_2 \rangle}{\|g_2\|_2^2} g_2, \quad \text{khi } \langle g_1, g_2 \rangle < 0.$$

---

## 4. Chẩn Đoán Đặc Thù Cho Inverse Problems & Crowd Counting

### 4.1. Tính Co Thắt Lipschitz trong Unrolled Solvers
Trong các bộ giải bài toán nghịch đảo mở rộng qua $T$ bước lặp (Unrolled SIRT, Landweber, ISTA):
$$Y^{(t+1)} = \mathcal{T}(Y^{(t)}, b).$$
Theo định lý điểm bất động Banach, toán tử $\mathcal{T}$ phải là một **Ánh Xạ Co (Contraction Mapping)** với hệ số Lipschitz $L < 1$:
$$\|\mathcal{T}(u) - \mathcal{T}(v)\| \le L \|u - v\|, \quad L < 1.$$

**Quy Tắc Kiểm Tra Tính Đơn Điệu:**
Residual sai số quan sát phải suy giảm đơn điệu:
$$r^{(t+1)} = \|A Y^{(t+1)} - b\|_2 \le r^{(t)} + \epsilon_{\text{tol}}.$$
Nếu sai số residual tăng đột biến qua các bước lặp ($r^{(t+1)} \gg r^{(t)}$), solver đang vi phạm điều kiện hội tụ và sẽ phân kỳ số học khi tăng số bước unrolling.

### 4.2. Bẫy Zero-Support & Carrier Sóng Mang
Trong bài toán khôi phục độ đo Dirac / Radon measure:
- Nhãn là các điểm rời rạc $\mu = \sum_i \delta_{x_i}$.
- Nếu mô hình dự đoán trường sóng mang $\hat{Y}(x) = 0$ trên một vùng rộng quanh điểm $x_i$, gradient của các hàm loss đối xứng (Bayesian, OT, PMLoss) sẽ bị triệt tiêu ($\nabla_{\hat{Y}} \mathcal{L} = 0$).
- Mô hình bị khóa cứng trong bẫy không hỗ trợ (Zero-Support Trap) và không thể bù đắp khối lượng đã mất.
- **Biện pháp bắt buộc**: Luôn duy trì sàn sóng mang tối thiểu:
  $$Y_{\text{safe}} = \text{clamp\_min}(Y, \tau_{\text{floor}}), \quad \tau_{\text{floor}} \ge 0.005.$$

---

## 5. Quy Trình Vận Hành Chuẩn (SOP) Cho Code Review

```
[BƯỚC 1: XÁC ĐỊNH BÀI TOÁN & RÀNG BUỘC KIẾN TRÚC]
   ├── Xác định loại bài toán (Measure Recovery, Regression, Detection).
   ├── Kiểm tra ngân sách tham số (Parameters Ceiling).
   └── Kiểm tra ràng buộc huấn luyện (Zero Knowledge Distillation, v.v.).

[BƯỚC 2: RÀ SOÁT TĨNH (STATIC CODE AUDIT - P0 & P1)]
   ├── Kiểm tra tensor shapes, luật broadcasting, in-place modifications.
   ├── Kiểm tra các hàm sqrt, log, div, lgamma có đủ bảo vệ số học.
   └── Kiểm tra model.eval() và torch.no_grad() ở toàn bộ inference paths.

[BƯỚC 3: KIỂM THỬ ĐỘNG BẢO TOÀN ĐỊNH LÝ (DYNAMIC INVARIANT TESTS)]
   ├── Chạy Batch Independence test: Max discrepancy < 1e-5.
   ├── Chạy L1 Mass Conservation test dưới phép biến đổi lật ngang/crop.
   ├── Chạy Monotonic Residual Dissipation test trên solver.
   └── Chạy Effective Rank test trên bottleneck representations.

[BƯỚC 4: ĐIỀU PHỐI LOSS & TỐI ƯU HÓA]
   ├── Tính toán ma trận góc Cosine giữa các gradient thành phần.
   ├── Đo tỉ số cân bằng độ lớn gradient (Gradient Balance Ratio).
   └── Kiểm tra Warmup, Cosine Annealing, và GradScaler.

[BƯỚC 5: TỔNG HỢP BÁO CÁO & ĐỀ XUẤT MÃ NGUỒN CỤ THỂ]
   ├── Phân nhóm kết quả theo P0, P1, P2, P3.
   ├── Cung cấp diff/patch code sửa chữa trực tiếp.
   └── Chạy toàn bộ test suite để đảm bảo Zero Regression.
```

---

## 6. Bộ Test PyTorch Tự Động (Production Pytest Suite)

Bộ kiểm thử sau có thể nhúng trực tiếp vào hệ thống CI/CD hoặc `tests/`:

```python
"""Automated ML Invariant Suite."""
import pytest
import torch
import torch.nn as nn


def test_batch_sample_independence_invariant(model, input_shape=(1, 3, 128, 128)):
    """P0 Check: Output of sample i must not depend on sample j."""
    model.eval()
    x1 = torch.randn(input_shape)
    x2 = torch.randn(input_shape)
    with torch.no_grad():
        out1_single = model(x1)
        batch_out = model(torch.cat([x1, x2], dim=0))
    diff = (out1_single - batch_out[0:1]).abs().max().item()
    assert diff < 1e-5, f"Batch independence failed! Max diff: {diff}"


def test_autograd_graph_and_numerical_safety(model, input_shape=(2, 3, 128, 128)):
    """P1 Check: Forward and backward passes must produce finite numbers without NaN."""
    model.train()
    x = torch.randn(input_shape, requires_grad=True)
    out = model(x)
    y = out["y"] if isinstance(out, dict) else out
    assert torch.isfinite(y).all(), "NaN or Inf detected in model forward output!"
    loss = y.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "NaN or Inf detected in gradients!"


def test_bottleneck_effective_rank(features: torch.Tensor, min_ratio: float = 0.20):
    """P2 Check: Representation must not collapse into degenerate subspace."""
    c = features.shape[1]
    f_flat = features.permute(0, 2, 3, 1).reshape(-1, c).float()
    f_centered = f_flat - f_flat.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(f_centered, full_matrices=False)
    s = s[s > 1e-7]
    p = s / s.sum()
    erank = torch.exp(-(p * torch.log(p)).sum()).item()
    ratio = erank / c
    assert ratio >= min_ratio, f"Rank collapse detected: erank={erank:.1f}/{c} ({ratio:.1%})"


def test_l1_mass_conservation_under_flip(model, input_shape=(1, 3, 128, 128)):
    """Domain Invariant: Count measure must be invariant to horizontal flip."""
    model.eval()
    x = torch.randn(input_shape)
    x_flip = torch.flip(x, dims=[-1])
    with torch.no_grad():
        out = model(x)
        out_flip = model(x_flip)
        y = out["y"] if isinstance(out, dict) else out
        y_flip = out_flip["y"] if isinstance(out_flip, dict) else out_flip
    c1, c2 = y.sum().item(), y_flip.sum().item()
    rel_err = abs(c1 - c2) / max(c1, c2, 1e-6)
    assert rel_err < 1e-3, f"Mass conservation violated! Relative error: {rel_err:.2e}"
```
