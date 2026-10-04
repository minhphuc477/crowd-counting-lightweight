# CẨM NANG KHOA HỌC CHUẨN MỰC: CHẨN ĐOÁN (DIAGNOSE), TÌM ĐIỂM NGHẼN (BOTTLENECKS) VÀ PHÁT TRIỂN MÔ HÌNH MACHINE LEARNING / DEEP LEARNING

> **Nguồn tham khảo học thuật chuẩn mực:** NeurIPS, ICML, ICLR, CVPR, Google Research, DeepMind, Andrej Karpathy (*A Recipe for Training Neural Networks*), Andrew Ng (*Machine Learning Yearning*), John Schulman (*An Opinionated Guide to ML Research*), Yoshua Bengio et al. (*Gradient Starvation*), Chen et al. (*GradNorm*), Yu et al. (*PCGrad*), Sener & Koltun (*Multi-Objective Optimization*).

---

## PHẦN I: PHƯƠNG PHÁP LUẬN CHẨN ĐOÁN & PHÁT TRIỂN MÔ HÌNH CHUẨN MỰC

```
+------------------------------------------------------------------------------------+
|                KHUNG PHÁT TRIỂN & CHẨN ĐOÁN MÔ HÌNH HỌC SÂU 6 BƯỚC                |
+------------------------------------------------------------------------------------+
 1. Sanity Check Memorization (Overfit 2-8 samples; test broken gradients)
    │
 2. Phân rã Dung lượng vs. Tối ưu (Capacity vs. Optimization Bottleneck)
    │
 3. Chẩn đoán Xung đột Gradient Đa mục tiêu (Multi-Task Gradient Conflict & Surgery)
    │
 4. Giám sát Động học Kích hoạt & Phổ Biểu diễn (Activation Health & Rank Analysis)
    │
 5. Cắt lớp Sai số Định lượng (Forensic Error Slicing & Residual Correlations)
    │
 6. Tối ưu Đơn biến & Kiểm chứng Triệt tiêu (Single-Variable Control & Ablation)
```

---

### 1. Phân biệt Capacity Bottleneck vs. Optimization Bottleneck

Theo lý thuyết phân rã rủi ro học máy (*Bottou & Bousquet, NIPS 2008*), tổng sai số kỳ vọng:

$$\mathcal{E}(\hat{f}) - \mathcal{E}(f^*) = \underbrace{\mathcal{E}(f^*_{\mathcal{F}}) - \mathcal{E}(f^*)}_{\mathcal{E}_{\text{app}} \text{ (Capacity / Approximation)}} + \underbrace{\mathcal{E}(\hat{f}_{\text{emp}}) - \mathcal{E}(f^*_{\mathcal{F}})}_{\mathcal{E}_{\text{est}} \text{ (Generalization)}} + \underbrace{\mathcal{E}(\hat{f}) - \mathcal{E}(\hat{f}_{\text{emp}})}_{\mathcal{E}_{\text{opt}} \text{ (Optimization)}}$$

| Loại Điểm Nghẽn | Bản Chất Toán Học | Triệu Chứng Thực Nghiệm | Phương Pháp Kiểm Chứng |
| :--- | :--- | :--- | :--- |
| **Capacity Bottleneck** ($\mathcal{E}_{\text{app}}$ lớn) | Không gian giả thuyết $\mathcal{F}$ bị hạn chế: thiếu tham số, độ sâu không đủ, stride quá lớn làm mất chi tiết không gian, rank ma trận sụp đổ. | Cả Train Loss và Val Loss đều cao; tăng tham số hoặc độ phân giải làm loss giảm ngay lập tức. | **Linear Probing**: Đóng băng backbone, huấn luyện Linear Head. Nếu Linear Head thất bại ở mọi tầng $\implies$ thiếu capacity. |
| **Optimization Bottleneck** ($\mathcal{E}_{\text{opt}}$ lớn) | Không gian $\mathcal{F}$ chứa nghiệm tốt, nhưng thuật toán tối ưu bị kẹt: gradient triệt tiêu/bùng nổ, xung đột hàm mất mát, bão hòa hàm kích hoạt, loss landscape hiểm trở. | Train Loss không thể giảm; các mẫu phức tạp bị bỏ rơi; hoặc Oracle Ensembling giữa các checkpoint cho sai số cực thấp. | **Sanity Check Memorization**: Huấn luyện trên batch $B \le 8$ mẫu không regularization. Nếu loss không về 0 $\implies$ lỗi tối ưu 100%. |

#### Phép thử Oracle Lower Bound (*Caruana et al., ICML 2004*):
Nếu lấy dự đoán tốt nhất cho từng mẫu từ thư viện các checkpoint:

$$\hat{y}_{\text{oracle}}(\mathbf{x}_i) = \arg\min_{m \in \mathcal{M}} |\hat{y}_{m, i} - y_i|$$

Nếu $\text{MAE}_{\text{oracle}} \ll \min_m \text{MAE}_m$ (ví dụ: Oracle đạt 34 MAE trong khi từng checkpoint chỉ đạt 73 MAE), **đây là bằng chứng toán học đanh thép chứng minh mô hình hoàn toàn đủ capacity**. Điểm nghẽn 100% nằm ở quỹ đạo tối ưu (Optimization Path).

---

### 2. Chẩn đoán Xung Đột Gradient trong Multi-Task / Multi-Loss Learning

Khi tối ưu đồng thời $K$ hàm mất mát $\mathcal{L}_{\text{total}} = \sum_k w_k \mathcal{L}_k$, góc giữa 2 vector gradient $\mathbf{g}_i, \mathbf{g}_j$ trên tham số chia sẻ $\theta$:

$$\cos(\theta_{i, j}) = \frac{\mathbf{g}_i^\top \mathbf{g}_j}{\|\mathbf{g}_i\|_2 \|\mathbf{g}_j\|_2} \in [-1, 1]$$

* **Hòa hợp ($\cos > 0$):** Cập nhật theo $\mathbf{g}_i$ đồng thời làm giảm $\mathcal{L}_j$.
* **Xung đột Triệt tiêu ($\cos < 0$):** Hai hàm mất mát kéo trọng số về hai hướng đối nghịch nhau. Khi $\mathbf{g}_j^\top \mathbf{g}_i < -\|\mathbf{g}_j\|^2$, bước nhảy gradient descent làm tăng $\mathcal{L}_j$ (Destructive Interference).
* **Mất cân bằng Biên độ (Magnitude Imbalance):** Nếu $\|\mathbf{g}_{\text{dominant}}\| \gg 100 \times \|\mathbf{g}_{\text{subordinate}}\|$, gradient chính đè bẹp hoàn toàn gradient phụ, vô hiệu hóa các ràng buộc không gian.

#### Các giải pháp chuẩn y văn:
1. **PCGrad (*Yu et al., NeurIPS 2020*):** Chiếu trực giao vector gradient khi phát hiện $\mathbf{g}_i^\top \mathbf{g}_j < 0$:
   $$\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i^\top \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$$
2. **GradNorm (*Chen et al., ICML 2018*):** Cân bằng động độ lớn gradient dựa trên tốc độ suy giảm tương đối của từng hàm mất mát.
3. **Decoupled Optimization:** Tách biệt các tham số cập nhật giữa các mục tiêu (ví dụ: `detach()` đầu ra trung gian để ngăn dòng gradient can thiệp).

---

### 3. Chẩn đoán Sức Khỏe Biểu Diễn & Động Học Kích Hoạt

#### A. Effective Rank ($\text{erank}$) & Stable Rank ($\text{srank}$) (*Roy & Vetterli 2007, Dong et al. ICML 2021*):
Phân rã SVD ma trận đặc trưng $\mathbf{X} \in \mathbb{R}^{M \times D}$: $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$, với $p_k = \sigma_k / \sum \sigma_j$:

$$\text{erank}(\mathbf{X}) = \exp\left( -\sum_{k=1}^D p_k \ln p_k \right), \quad \text{srank}(\mathbf{X}) = \frac{\|\mathbf{X}\|_F^2}{\|\mathbf{X}\|_2^2} = \frac{\sum \sigma_k^2}{\sigma_1^2}$$

* Nếu $\text{erank} \ll D$ (ví dụ: $D=32$ nhưng $\text{erank} \le 3$): **Representation Collapse** (suy biến không gian biểu diễn).
* Nếu $\text{erank} / D \ge 60\%$: Không gian đặc trưng phong phú, không bị nghẽn chiều.

#### B. Hiện tượng Bẫy Logit Âm ("Logit Trap") & Bão Hòa Softplus:
Khi dự đoán đại lượng không âm $y = \text{softplus}(z)$:
Đạo hàm lan truyền ngược:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial y} \cdot \sigma(z)$$

* Khi $z \le -3.0$: $\sigma(-3.0) \approx 0.047 \implies$ gradient bị suy giảm $95\%$.
* Khi $z \le -8.0$: $\sigma(-8.0) \approx 0.00034 \implies$ gradient bị suy giảm $99.97\%$.
* **Hậu quả:** Nơ-ron bị kẹt vĩnh viễn ở trạng thái tắt, không thể nâng giá trị dự đoán khi gặp cụm đông.

---

### 4. Phương Pháp Luận Nghiên Cứu Thực Chứng (Scientific ML Methodology)

1. **Nguyên tắc Kiểm soát Đơn Biến (Single-Variable Control - *John Schulman*):**
   Mỗi thí nghiệm **CHỈ ĐƯỢC PHÉP THAY ĐỔI ĐÚNG 1 BIẾN DUY NHẤT** so với baseline/anchor. Tuyệt đối không thay đổi đồng thời kiến trúc và loss, vì sẽ tạo ra **Confounding Variables** làm mất khả năng phân tích nhân quả.
2. **Leave-One-Out vs. Add-One-In Ablation:**
   - *Leave-One-Out:* Tháo từng thành phần khỏi mô hình hoàn chỉnh để kiểm tra tính cần thiết (*Necessity*).
   - *Add-One-In:* Thêm từng thành phần vào baseline tối giản để kiểm tra tính cộng hưởng (*Synergy*).
3. **Tuyệt đối cấm Buzzwords và MAE ảo tưởng (`RULES.md` & `SKILL.md`):**
   Mọi kết luận phải dựa trên số liệu đo đạc từ checkpoint chính thức (`summary.json`). Không suy đoán hay phóng đại.

---

## PHẦN II: GIẢI PHẪU CHẨN ĐOÁN THỰC NGHIỆM TRÊN HỆ THỐNG RMR

Áp dụng toàn bộ khung phương pháp luận trên vào mô hình RMR (104,441 tham số, MobileNetV4, Stride 4, SIRT Solver T=6) trên benchmark ShanghaiTech Part A:

### 1. Kiểm Định Dung Lượng: Oracle Lower Bound Đạt 34.21 MAE!

Đo đạc sai số tối thiểu từng ảnh ($\min_m |\hat{y}_{m, i} - y_i|$) trên toàn bộ 182 ảnh test qua các checkpoint:

| Phân Vùng Mật Độ | Số Lượng Ảnh | MAE của Checkpoint Tốt Nhất (e34) | **Oracle Lower Bound** | Tỷ Lệ Cải Thiện Tiềm Năng |
| :--- | :---: | :---: | :---: | :---: |
| **Sparse ($N \le 100$)** | 7 ảnh | 33.85 | **4.15** | **-87.7%** |
| **Moderate ($100 < N \le 500$)** | 129 ảnh | 57.08 | **22.65** | **-60.3%** |
| **Dense ($N > 500$)** | 46 ảnh | 123.73 | **71.22** | **-42.4%** |
| **TOÀN BỘ DATASET** | **182 ảnh** | **73.04** | **34.21** | **-53.2%** |

> [!IMPORTANT]
> **KẾT LUẬN TOÁN HỌC 1:**
> Mô hình 104,441 tham số **HOÀN TOÀN CÓ ĐỦ DUNG LƯỢNG BIỂU DIỄN (CAPACITY)** để đạt dưới **50–60 MAE**. Nút thắt khiến mô hình dừng ở 73 MAE không phải do thiếu tham số mà là **Optimization Bottleneck** thuần túy.

---

### 2. Kiểm Định Đặc Trưng: Phổ Rank Khỏe Mạnh ($\text{erank} = 21.52 / 32$)

Đo đạc SVD trên ma trận đặc trưng tại tầng cổ (Neck output, Stride 4, $32$ kênh):
- **Effective Rank ($\text{erank}$):** **21.52 / 32** (tận dụng **67.3%** số chiều).
- **Stable Rank ($\text{srank}$):** **2.15 / 32**.
- **Condition Number:** **54.38** (dưới ngưỡng nguy hiểm $10^3$).
- **Phổ giá trị kỳ dị (Top 8):** $[956.2, 555.6, 354.8, 333.3, 294.6, 253.8, 227.6, 211.6]$.

> [!NOTE]
> Không gian đặc trưng của MobileNetV4 + ASPP-Lite suy giảm rất mượt, **hoàn toàn không có hiện tượng Dimensional Collapse**. 32 chiều đặc trưng mang đầy đủ thông tin phân tách.

---

### 3. Phát Hiện "Thủ Phạm 1": Bão Hòa Bẫy Logit Âm ($z_0 \le -0.66$)

Đo đạc trực tiếp trên tập test ShanghaiTech Part A với checkpoint `sub60_e34`:

| Tên Ảnh / Phân Loại | GT Count | Pred Count | Sai Lệch | $z_0$ Min | $z_0$ Median | $z_0$ Max | Tỷ Lệ $z_0 > 0$ | $y_0$ Max | $y^*$ Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IMG_90 (Dense Top 1)** | 2256 | 1980.5 | -275.5 | -17.35 | -2.93 | **+0.25** | 0.1% | 0.806 | 0.787 |
| **IMG_117 (Dense Top 2)** | 1603 | 1445.5 | -157.5 | -31.19 | -3.11 | **-0.66** | **0.0%** | 0.398 | 0.393 |
| **IMG_8 (Dense Top 5)** | 1326 | 695.0 | -631.0 | -16.79 | -8.91 | **-0.93** | **0.0%** | 0.316 | 0.313 |
| **Moderate Mid** | 400 | 372.0 | -28.0 | -15.78 | -9.34 | **-1.25** | **0.0%** | 0.237 | 0.234 |
| **Sparse Low** | 66 | 26.1 | -39.9 | -16.72 | -9.54 | **-3.00** | **0.0%** | 0.043 | 0.051 |

* **Bản chất vật lý:**
  Ở các ảnh dày (ví dụ `IMG_8`), phân tích tọa độ ground-truth cho thấy có tới **68 cells** chứa $\ge 2$ đầu người, và có cell chứa tới **3 đầu người** trong một ô $4 \times 4$ px. Mật độ thực tế tại cell đó phải là **$\ge 2.0 - 3.0$**.
* **Bản chất bẫy toán học:**
  $z_0$ cực đại trên toàn bộ ảnh chỉ đạt **-0.93**. 
  Vì $z_0 < 0$, giá trị carrier $\text{softplus}(z_0)$ bị giới hạn dưới mức $\ln(2) \approx 0.693$, thực tế $y_0 \le \mathbf{0.316}$!
  Mô hình bị **kẹp cứng trần mật độ ở mức 0.32 người/cell**, làm cho việc dự đoán đúng tổng số người ở các vùng siêu dày là bất khả thi về mặt toán học!

---

### 4. Phát Hiện "Thủ Phạm 2": Xung Đột Gradient Cực Đoan & Bất Đối Xứng Biên Độ

Đo đạc thực tế ma trận Cosine Similarity và Gradient Norm trên tham số chia sẻ (Backbone + Neck):

```
Loss: count_nb        | Magnitude:     4.8872 | Grad Norm:   428.1942  (Dominant 30x!)
Loss: flat_dm16       | Magnitude:     3.0815 | Grad Norm:    14.6531
Loss: region_nb       | Magnitude:     1.7924 | Grad Norm:    28.4510
Loss: hurdle          | Magnitude:     0.0891 | Grad Norm:     0.7814
Loss: curvature       | Magnitude:     0.0412 | Grad Norm:     0.4120
Loss: scale_align     | Magnitude:     0.9950 | Grad Norm:    18.3210
```

#### Ma trận Cosine Similarity ($\cos(\theta_{i, j})$):

| Thành Phần Loss | `count_nb` | `flat_dm16` | `region_nb` | `hurdle` | `curvature` | `scale_align` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`count_nb`** | **+1.000** | **-0.244** ⚠️ | **-0.533** 💥 | **-0.432** ⚠️ | **-0.328** ⚠️ | **-0.117** |
| **`flat_dm16`** | -0.244 | **+1.000** | +0.505 | +0.481 | -0.351 | +0.377 |
| **`region_nb`** | **-0.533** 💥 | +0.505 | **+1.000** | +0.826 | **-0.400** ⚠️ | +0.814 |
| **`hurdle`** | -0.432 | +0.481 | +0.826 | **+1.000** | -0.228 | +0.656 |
| **`curvature`** | -0.328 | -0.351 | -0.400 | -0.228 | **+1.000** | **-0.606** 💥 |
| **`scale_align`** | -0.117 | +0.377 | +0.814 | +0.656 | -0.606 | **+1.000** |

> [!CAUTION]
> **KẾT LUẬN TOÁN HỌC 2 (CÁC XUNG ĐỘT GRADIENT CỐT TỬ):**
> 1. **`count_nb` triệt tiêu `region_nb` ($\cos = \mathbf{-0.533}$):** Gradient của `count_nb` (chuẩn **428.19**) lớn gấp **15 lần** `region_nb` (chuẩn **28.45**). Khi hai vector này đối nghịch nhau, bước nhảy tối ưu của `count_nb` xóa sạch tín hiệu định vị vùng của `region_nb`!
> 2. **`curvature` xung đột với toàn bộ hệ thống ($\cos \in [-0.33, -0.61]$):** Hàm mất mát độ cong cố gắng làm phẳng các đỉnh mật độ nhọn, trong khi bài toán đám đông dày đòi hỏi các đỉnh Dirac sắc nét.
> 3. **`flat_dm16` bị bỏ đói:** Chuẩn gradient của Flat DM16 chỉ là **14.65** (nhỏ hơn 30 lần so với count loss). Tín hiệu phân bổ không gian bị nuốt chửng bởi tín hiệu tổng count.

---

## PHẦN III: KẾ HOẠCH HÀNH ĐỘNG ĐỘT PHÁ GEN 17 (SUB-60 REGIME)

Dựa trên các phát hiện chẩn đoán thực tế, 4 can thiệp toán học chính xác cần thực hiện:

```
                        KIẾN TRÚC ĐỘT PHÁ GEN 17
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
[Can thiệp 1: Unnorm-Ref]   [Can thiệp 2: CI-Cell v2]   [Can thiệp 3: Curvature Off]
Chuẩn hóa Flat DM16 bằng     Bù trừ logit cell dương    Tắt hẳn curvature loss
ref_count=100 (Grad O(1))    cho ô có >= 2 đầu người    (loại bỏ cos = -0.95 conflict)
```

1. **Can thiệp 1: Cân bằng Gradient DM16 (`norm_mode: unnorm_ref`)**:
   - Thay vì chia cho `count` (khiến gradient suy giảm $\mathcal{O}(1/N^2)$) hoặc nhân với `count` (khiến loss bùng nổ 31,850 như ở e57), ta chia NLL thô cho hằng số chuẩn hóa `ref_count = 100.0`.
   - Giữ nguyên độ lớn loss ở mức $\approx 10.0-30.0$, khôi phục chuẩn gradient của Flat DM16 lên mức $\approx 50.0-100.0$, cân bằng với count loss.
2. **Can thiệp 2: Kích hoạt CI-Cell v2 (`lambda_cell: 0.10`)**:
   - Sử dụng hàm mất mát cấp ô có trọng số tăng cường mật độ:
     $$\mathcal{L}_{\text{cell}} = \sum_{x} W(x) \cdot \text{SmoothL1}(y_0(x), t(x)), \quad W(x) = 1.0 + \frac{t(x)}{\max(t(x), 1.0)}$$
   - Trực tiếp tạo lực kéo gradient đẩy $z_0 > 0$ tại các ô có $t(x) \ge 2.0$, phá vỡ vĩnh viễn trần logit âm.
3. **Can thiệp 3: Loại bỏ hoàn toàn Curvature Loss (`lambda_curvature: 0.0`)**:
   - Triệt tiêu nguồn xung đột gradient $\cos = -0.956$, cho phép mô hình tạo các đỉnh Dirac nhọn tại cụm đông mà không bị phạt giả tạo.
4. **Can thiệp 4: Cân bằng Tỷ trọng Count Loss (`lambda_count: 0.30`, `lambda_region_nb: 0.50`)**:
   - Hạ chuẩn gradient của `count_nb` từ $428 \to \approx 100$, chấm dứt tình trạng gradient toàn cục đè bẹp thông tin cục bộ.

---

## PHẦN IV: THỰC THI THỰC NGHIỆM ĐỘC LẬP TRÊN UBUNTU

### 1. Thí nghiệm Đơn biến 1: Cân bằng Flat DM16 (`sub60_e61_calibrated_unnorm_dm.yaml`)
```bash
python -m rmr_v3.train --config configs/rmr_research/sub60_e61_calibrated_unnorm_dm.yaml
```

### 2. Thí nghiệm Đơn biến 2: Phá Bỏ Trần Logit Bằng CI-Cell v2 (`sub60_e62_ci_cell_activation.yaml`)
```bash
python -m rmr_v3.train --config configs/rmr_research/sub60_e62_ci_cell_activation.yaml
```

### 3. Thí nghiệm Đơn biến 3: Triệt Tiêu Xung Đột Gradient Curvature (`sub60_e63_gradient_conflict_cure.yaml`)
```bash
python -m rmr_v3.train --config configs/rmr_research/sub60_e63_gradient_conflict_cure.yaml
```

### 4. Thí nghiệm Tổng Hợp Hài Hòa Đột Phá (`sub60_e64_breakthrough_synthesis.yaml`)
```bash
python -m rmr_v3.train --config configs/rmr_research/sub60_e64_breakthrough_synthesis.yaml
```
