# RMR-Count: Lịch Sử Kiến Trúc, Phân Tích Cơ Chế, Stage C và Hướng RMR-v3

**Dự án**: Ultra-Lightweight Crowd Counting  
**Ngân sách mục tiêu**: Khoảng dưới 105k tham số cho graph trainable chính  
**Dataset trọng tâm hiện tại**: ShanghaiTech Part A  
**Trạng thái**: Stage C validation đã hoàn tất cho B0, B1, B2, B3a, B5-P; B3b là learned-projector control cần hoàn tất cùng protocol trước khi đóng causal claim.  
**Test set policy**: Phải giữ **FROZEN** tuyệt đối cho đến khi architecture, loss, config, checkpoint-selection rule và commit được freeze hoàn toàn.

---

## 1. Mục Đích Tài Liệu

Tài liệu này tổng hợp toàn bộ quá trình phát triển kiến trúc từ:
1. **MICF** (Multi-scale Integral Context Flow)
2. **NTPC** (Neural Tree Pólya Count)
3. **RMR Pilot** (Stage A / Stage B Pilots)
4. **RMR-v2 / Stage C** (Projected SIRT & Causal Matched Matrix)
5. **Định hướng RMR-v3** (Reliability-Weighted Regional Measure Reconciliation)

Mục tiêu cốt lõi không phải tạo một narrative “mỗi thế hệ sau đều tốt hơn thế hệ trước”, mà là xác định rõ ràng, khách quan và minh bạch về mặt khoa học:
- Từng hypothesis đã được kiểm tra như thế nào;
- Hypothesis nào bị bác bỏ (falsified);
- Lesson nào được chắt lọc và giữ lại;
- Failure mechanism nào đã được quan sát trong thực nghiệm;
- Component nào của RMR hiện tại thực sự có bằng chứng thực nghiệm bảo chứng;
- Claim nào chưa đủ bằng chứng để tuyên bố;
- Bước nâng cấp tiếp theo (RMR-v3) phải giải quyết trúng failure pattern hiện tại.

---

## 2. Provenance và Nguyên Tắc Diễn Giải

### 2.1. Các nhánh chính trong repository
Lineage thực tế của codebase:
```
master
  ↓
test-1
  ↓
feat/ntpc-neural-tree-polya
  ↓
MICF
  ↓
RMR (HEAD)
```
Các nhánh này không phải là các project hoàn toàn tách biệt. Chúng là các thế hệ kế tiếp nhau, kế thừa backbone, loss formulation, data pipeline và evaluation suite.

### 2.2. Cảnh báo quan trọng về so sánh lịch sử
Không được coi mọi con số giữa MICF, NTPC và RMR là một benchmark hoàn toàn matched. Các thế hệ có sự khác biệt lớn về:
- **Crop size**: Crop 256 (NTPC) vs Crop 512 (RMR Stage C).
- **Hàm mất mát (Loss)**: Đơn thuần DM NLL vs Rebalanced Multi-loss (Count + DM16 + Cell + Region).
- **Backbone implementation**: Truncation depth, GroupNorm vs BatchNorm, feature reduction mapping.
- **Pretrained vs Scratch**: Scratch trong RMR Pilot vs Pretrained ImageNet-1k trong NTPC & Stage C.
- **Quy tắc chọn Checkpoint**: Test-loss selection (NTPC cũ) vs Held-out Validation MAE selection (Stage C).
- **Giao thức Direct / Tiled**: Direct full-image vs Local sliding-window tiling.
- **Runtime solver**: Không có (MICF/NTPC) vs Latent Softplus solver (Pilot) vs Projected Measure-Space SIRT (Stage C).
- **Chính sách Learning Rate**: Uniform LR 1e-4 vs Differential LR (Backbone 1e-5, Main/Neck 1e-4).

> [!IMPORTANT]
> **Nguyên tắc khoa học**: So sánh lịch sử dùng để phân tích mechanism và design evolution, không phải bảng SOTA matched protocol. Bảng causal duy nhất để chứng minh giá trị của RMR là **Stage C matched matrix**.

---

## 3. Tổng Quan Design Evolution: Ba Bài Học Lớn

Toàn bộ quá trình nghiên cứu qua 4 thế hệ có thể đúc kết thành 3 bài học nền tảng:

### 3.1. Lesson từ MICF
**Không dùng một representation có identity phụ thuộc mạnh vào extent toàn ảnh nếu training chủ yếu bằng crop.**
$$\text{global extent-dependent cumulative state} \implies \text{crop/full/tile representation mismatch} \implies \text{direct-tiled inconsistency}$$
*Bài học được giữ lại trong RMR*:
- Loại bỏ hoàn toàn full-image region trong main setup;
- Chỉ dùng các vùng có kích thước vật lý cố định (32, 64, 128 px);
- Hình học không phụ thuộc vào tọa độ crop cục bộ;
- Solver chỉ hoạt động trên local finite-support measurements.

### 3.2. Lesson từ NTPC
**Hierarchy không tự động tốt hơn flat allocation.**
$$\text{Flat-DM16} > \text{Hierarchical Multinomial} \ge \text{Hierarchical Dirichlet-Multinomial}$$
*Bài học được giữ lại trong RMR*:
Nếu việc phân bổ không gian mịn (fine spatial allocation) có thể được giám sát bằng một hàm mục tiêu xác suất phẳng sạch (Flat-DM16), không nên ép mạng học một cấu trúc cây phân cấp phức tạp chỉ vì nó nghe có vẻ hợp lý trên lý thuyết.
RMR-v2 vì vậy giữ: **Flat-DM16 làm training-only spatial allocation supervision**, và loại bỏ Tree-Pólya khỏi cơ chế suy luận chính.

### 3.3. Lesson từ RMR
**Thông tin vùng (regional information) chỉ thực sự có ích khi ta phân biệt rõ ba cấp độ:**
1. *Regional supervision*: Giám sát $A Y$ qua loss trong quá trình huấn luyện (Model B1).
2. *Regional prediction*: Dự đoán nhánh phụ $b$ độc lập như một auxiliary task (Model B2).
3. *Runtime regional correction*: Dùng $b$ để hiệu chỉnh trực tiếp $Y$ tại thời điểm suy luận (Model B5-P).

Stage C chứng minh ba cấp độ này không hề tương đương: B1 và B2 cho kết quả tương đương (~87.7 - 87.8 MAE), và chỉ khi bước sang B5-P (Runtime reconciliation), bước nhảy vọt mới xuất hiện (**79.38 MAE**).

---

## 4. Gen 1 — MICF (Multi-scale Integral Context Flow)

### 4.1. Ý tưởng cốt lõi
MICF cố gắng mô hình hóa crowd count bằng biểu diễn tích phân 2D (cumulative/integral field) $C \in \mathbb{R}^{H_o \times W_o}$ thay vì chỉ dự đoán local density map:
$$C(u, v) = \sum_{i=0}^u \sum_{j=0}^v Y(i, j) \quad \forall (u, v) \in [0, H_o-1] \times [0, W_o-1]$$
Khối lượng cục bộ được khôi phục bằng toán tử vi phân rời rạc (Finite Mixed Difference):
$$\Delta_{xy} C(u, v) = C(u, v) - C(u-1, v) - C(u, v-1) + C(u-1, v-1)$$
Ý tưởng này hấp dẫn vì về bản chất, count là một độ đo cộng tính (additive measure), và góc cuối cùng $C(H_o-1, W_o-1) \equiv N$ đại diện cho tổng số người.

### 4.2. Global Cumulative Formulation và Điểm nghẽn Extent-Dependence
Vấn đề chí mạng của trường tích phân toàn ảnh là giá trị tại tọa độ $(u, v)$ không chỉ phụ thuộc vào đặc trưng thị giác tại vùng đó, mà còn phụ thuộc vào **toàn bộ phần ảnh nằm phía trên và bên trái nó**.
Với cùng một local patch kích thước $256 \times 256$:
- Trong training crop, nó nằm ở tọa độ $(0, 0) \implies C$ nhỏ.
- Trong full image, nó nằm ở góc dưới-phải $\implies C$ cực lớn.
- Trong sliding tile, tọa độ tích phân lại thay đổi hoàn toàn.

Hệ quả: Biểu diễn $C$ bị ràng buộc mật thiết vào extent của ảnh, phá hủy tính bất biến tịnh tiến (translation invariance).

### 4.3. B5b và Failure Pattern
Số liệu lịch sử của B5b (Global Extent-Aware):
- **Tiled MAE**: 118.31
- **Direct MAE**: 209.63
- **Direct-Tiled Gap**: **+91.32**
- **RMSE**: 308.52
- **Bias**: +137.81
- **Negative Cell Violation Rate**: 20.8% ($\Delta_{xy} C < 0$)

Điểm quan trọng nhất không chỉ là MAE cao, mà là $\text{Direct MAE} \gg \text{Tiled MAE}$ (chênh lệch tới 91.32 điểm), chứng minh hiện tượng representation drift nghiêm trọng theo kích thước ảnh.

### 4.4. Finite-Horizon MICF (B8)
Model B8 cố gắng giải quyết lỗi extent bằng cách cắt nhỏ trường tích phân thành các khối cục bộ hữu hạn (Finite-Horizon $K=4$, khối $4 \times 4$):
- **Tiled MAE**: 105.98
- **Direct MAE**: 115.22
- **Direct-Tiled Gap**: **9.24** (giảm ngoạn mục từ 91.32 của B5b)
- **RMSE**: 188.76
- **Bias**: -73.00

*Kết luận thiết kế*: Ràng buộc hình học cục bộ hữu hạn (finite/local geometry) khắc phục được phần lớn hiện tượng extent inconsistency. Tuy nhiên, độ chính xác tuyệt đối vẫn bị giới hạn và under-count nặng (Bias -73.0).

### 4.5. Kết luận MICF
MICF không nên được hồi sinh nguyên vẹn trong RMR-v3. Bài học cốt tử được giữ lại:
$$\boxed{\text{Local finite-support representation} \gg \text{Global extent-dependent cumulative identity}}$$
RMR-v2 và v3 hấp thụ bài học này bằng các vùng cục bộ 32/64/128 px độc lập và loại bỏ hoàn toàn ràng buộc full-image.

---

## 5. Gen 2 — NTPC (Neural Tree Pólya Count)

### 5.1. Kiến trúc
NTPC xây dựng một backbone siêu nhẹ dựa trên họ MobileNetV4 cắt ngắn đến tầng C16:
- Tổng tham số: **96,593 tham số** (sub-0.1M envelope).
- Backbone chiếm hơn 90% tham số; task-specific head chỉ khoảng ~9,000 tham số.
- Output: Bản đồ khối lượng dương (positive mass map) ở Stride-4.
- Đóng góp cốt lõi của NTPC là **mục tiêu phân bổ xác suất (probabilistic allocation formulation)** thông qua phân bố Dirichlet-Multinomial.

---

## 6. NTPC Experimental Family (Ablation History)

Số liệu thực nghiệm lịch sử từ nhánh `feat/ntpc-neural-tree-polya` (`runs/ablation_study_summary.json`):

| Run ID | Cơ chế | MAE | RMSE | Bias | Dense MAE (>500) | Ghi chú kiến trúc |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **R0** | Exact / Direct Regression | 84.89 | 143.51 | -10.72 | 277.97 | Baseline không phân rã không gian |
| **R1** | SDC Deterministic Allocation | 86.17 | 158.16 | -27.32 | 314.46 | Phân rã tất định không có độ mềm dẻo |
| **R2** | **Flat-DM16** | **74.09** | **123.49** | **-1.85** | **224.44** | **Chiến thắng áp đảo; đơn giản nhất** |
| **R3** | Multinomial Quadtree | 82.63 | 146.65 | -30.52 | 290.86 | Thiếu overdispersion; under-count nặng |
| **R4** | Neural DTM Quadtree (Tree-Pólya) | 77.69 | 144.49 | -21.94 | 288.51 | Cây phân cấp thua R2 |
| **R5** | Full Adaptive NTPC | 77.71 | 145.74 | -21.27 | 296.75 | Phức tạp hóa không tăng hiệu năng |
| **R6** | NPAC | 80.00 | 141.17 | -11.96 | 269.72 | Không cải thiện vùng đông |

*(Lưu ý provenance: Các kết quả trên thuộc giao thức crop-256 lịch sử của NTPC, không chạy dưới giao thức Stage C hiện tại).*

---

## 7. Tại Sao Flat-DM16 Là Bài Học Quan Trọng Nhất Từ NTPC?

Mô hình R2 (Flat-DM16) đánh bại hoàn toàn các biến thể cây phân cấp R3, R4, R5 trên mọi chỉ số:
- MAE thấp hơn: **74.09 vs 77.69 (R4) và 82.63 (R3)**.
- RMSE thấp hơn vượt trội: **123.49 vs 144.49 (R4)**.
- Bias tiệm cận 0: **-1.85 vs -21.94 (R4)**.
- Dense MAE tốt hơn: **224.44 vs 288.51 (R4)**.

Điều này chính thức **bác bỏ giả thuyết khoa học của Tree-Pólya**: Giả định rằng việc phân rã xác suất theo cây phân cấp (Quadtree) sẽ mang lại inductive bias tốt hơn mô hình phẳng là sai. Ngược lại, **Flat-DM16 (phân rã xác suất phẳng trực tiếp trên các khối 16px)** vừa đơn giản, vừa sạch, vừa tối ưu hóa dễ dàng hơn rất nhiều.

---

## 8. Hiện Tượng Xung Đột Gradient Cây (Tree Gradient Conflict)

Chẩn đoán chuyên sâu trong file `runs/objective_audit/audit_v2_full_test.json` của nhánh NTPC đã vạch trần cơ chế thất bại của Tree-Pólya:
- Gradient giữa tầng $64 \to 32$ và tầng $32 \to 16$ có **cosine similarity âm (-0.18)**.
- Hai tầng cây liên tục triệt tiêu lẫn nhau (destructive interference), gây xung đột gradient nội tại trong quá trình lan truyền ngược.

> [!CAUTION]
> **Diễn đạt an toàn cho bài báo**: Không nên tuyên bố "RMR triệt tiêu hoàn toàn gradient cancellation". Cách diễn đạt chuẩn xác là: *RMR loại bỏ các mục tiêu cây phân cấp từng được quan sát là có tương tác gradient triệt tiêu trong NTPC. RMR vẫn bao gồm nhiều thành phần loss và vẫn có thể tồn tại các dạng xung đột gradient khác.*

---

## 9. Kết Luận Thế Hệ NTPC

- **Thành phần giữ lại cho RMR**:
  - Carrier MobileNetV4 siêu nhẹ ImageNet-pretrained;
  - Positive stride-4 mass map;
  - Flat-DM16 spatial allocation supervision.
- **Thành phần loại bỏ vĩnh viễn**:
  - Tree-Pólya, deep quadtree hierarchy, adaptive tree depth, multi-level hierarchical likelihood.

---

## 10. Gen 3 — RMR Pilot

### 10.1. RMR Pilot không dùng cùng backbone với Stage C
Một số thực nghiệm pilot cũ của RMR sử dụng encoder siêu nhỏ huấn luyện từ đầu (`TinyLocalEncoder`, train from scratch), không phải MobileNetV4 pretrained.
Do đó, không được kết luận sai lệch rằng "pilot thất bại vì MobileNetV4 bị phá bởi learning rate". Nguyên nhân thực sự ở pilot cũ là: **Observer visual quá yếu** và **công thức solver chưa được chuẩn hóa**.

### 10.2. Kết quả Pilot Lịch sử (100 Epochs)
- **Pilot B2**: Best Val MAE ≈ 239.25 (Final: 340.16)
- **Pilot B3b**: Best Val MAE ≈ 237.77 (Final: 412.90)
- **Pilot B5-P**: Best Val MAE ≈ 235.78 (Final: 313.93)

Dù kết quả tuyệt đối lúc đó rất kém do observer yếu, chẩn đoán solver của B5-P đã bắt đầu cho thấy tín hiệu sửa sai không gian có giá trị.

---

## 11. Cơ Chế Thất Bại Của Bộ Giải Latent Cũ (Old Latent RMR Failure)

Trong các thiết kế RMR đời đầu, mật độ tế bào được định nghĩa qua biến ẩn (latent variable) $z$:
$$Y = \operatorname{softplus}(z)$$
Với khởi tạo thưa (sparse initialization) $z \approx -4.6$, ta có $\sigma(z) \approx 0.01$.  
Nếu cập nhật xảy ra trong không gian ẩn:
$$z_{t+1} = z_t - \eta \, r$$
hoặc qua Jacobian-gating:
$$z_{t+1} = z_t - \eta \, \sigma(z) \, r$$
thì sự thay đổi thực tế trong không gian khối lượng (measure space) là:
$$\Delta Y \approx \sigma(z) \Delta z \approx 0.01 \times \Delta z \approx 0$$
Hệ quả: **Bộ giải latent hầu như không thể sinh ra khối lượng mới ở các ô trống (empty cells)**, làm tê liệt khả năng sửa sai của solver.

---

## 12. Thí Nghiệm Oracle Của Projected Solver

Một thí nghiệm Oracle lịch sử đã được tiến hành để so sánh trực tiếp cơ chế cập nhật:
- Khởi tạo ban đầu: $Y_0$ MAE ≈ 236
- Latent Solver ($T=2$ với Oracle $b$): MAE ≈ **233** (hầu như không sửa được gì)
- **Projected Measure-Space Solver ($T=2$ với Oracle $b$)**: MAE vọt xuống **24**!

Thí nghiệm này là bằng chứng toán học đanh thép:
$$\boxed{\text{Toán tử chiếu trong không gian độ đo (Projected Solver) sở hữu năng lực sửa sai cực lớn nếu bằng chứng vùng } b \text{ đủ chính xác}}$$

---

## 13. Chuyển Đổi Từ Pilot Sang RMR-v2

RMR-v2 được tái thiết kế dựa trên sự phân công trách nhiệm rành mạch:
1. **Backbone**: Phải có năng lực thị giác mạnh (pretrained) để nhìn rõ người.
2. **FlatDM16**: Chịu trách nhiệm phân bổ không gian mịn (allocation).
3. **Regional Head**: Đo lường khối lượng thô đa tỷ lệ độc lập.
4. **RMR Operator**: Thực hiện hòa giải mâu thuẫn (reconciliation) trong không gian độ đo không âm.
*Nguyên tắc vàng: Tuyệt đối không bắt solver phải gánh vác nhiệm vụ cứu vãn một observer thị giác yếu kém.*

---

## 14. Kiến Trúc RMR-v2

### 14.1. Backbone
- **Model**: `mobilenetv4_conv_small_050.e3000_r224_in1k` (timm pretrained ImageNet-1k).
- Trích xuất đặc trưng theo reduction tỷ lệ thay vì hard-code layer indices: $C_4$ (stride 4), $C_8$ (stride 8), $C_{16}$ (stride 16).

### 14.2. Additive Feature Pyramid Network (FPN)
Chiếu tất cả các tầng về cùng số kênh $C=32$ bằng tích chập $1 \times 1$:
$$L_4 = W_4 C_4, \quad L_8 = W_8 C_8, \quad L_{16} = W_{16} C_{16}$$
Hòa nhập top-down bằng phép cộng thuần túy (Additive Fusion) kết hợp Bilinear Upsampling:
$$P_8 = L_8 + \uparrow P_{16}$$
$$P_4 = L_4 + \uparrow P_8$$
*Triết lý tối giản*: Không concatenation, không Transformer, không Self-Attention, giữ tham số tối thiểu.

---

## 15. Fine Measure Head

Fine observer dự đoán trực tiếp trường khối lượng trên lưới Stride-4:
$$z_0 = h_f(P_4), \quad Y_0 = \operatorname{softplus}(z_0)$$
- $Y_0$ là độ đo khối lượng người trên từng ô (count-per-cell measure).
- Không dùng làm bia mật độ Gaussian (Gaussian density target) bị nhòe.
- Ground truth được rasterize chính xác từ tọa độ điểm thực tế vào các ô stride-4, bảo toàn 100% tổng số người.

---

## 16. Empirical Initialization Prior

Để tránh hiện tượng dự đoán ban đầu bị overcount hoặc undercount cực đoan, RMR-v2 tính toán khối lượng tế bào trung bình từ tập huấn luyện:
$$m_0 = \frac{\sum_i N_i}{\sum_i H_{o,i} W_{o,i}}$$
Thiết lập bias khởi tạo cho lớp tích chập cuối:
$$b_0 = \operatorname{softplus}^{-1}(m_0) = \log(e^{m_0} - 1)$$
Giúp mạng bắt đầu quá trình huấn luyện ở điểm cân bằng thống kê tự nhiên của tập dữ liệu.

---

## 17. Scale-Matched Regional Evidence Head

Dự đoán bằng chứng vùng $b_R$ trên 3 tỷ lệ vật lý độc lập:
- **32 px**: Trích xuất từ đặc trưng $P_4$
- **64 px**: Trích xuất từ đặc trưng $P_8$
- **128 px**: Trích xuất từ đặc trưng $P_{16}$

Tất cả các feature map được đưa về hệ tọa độ hỗ trợ $P_4$ khi cần để tránh sai số lượng tử hóa hộp (box quantization error).  
Mật độ vùng dự đoán:
$$\rho_R = \operatorname{softplus}(h_r(u_R, \log s_R)), \quad b_R = |R| \cdot \rho_R$$

---

## 18. Toán Tử Vùng (Regional Operator Formulations)

Cho $Y$ là bản đồ mật độ tế bào mịn kích thước $H_g \times W_g$.
- **Toán tử thuận (Forward Operator $A$)**:
  $$(A Y)_R = \sum_{p \in R} Y_p$$
- **Toán tử liên hợp (Adjoint Operator $A^\top$)**:
  $$(A^\top r)_p = \sum_{R \ni p} r_R$$
- **Ma trận đường chéo diện tích (Area Diagonal $D_a$)**: $D_a(R, R) = |R|$
- **Ma trận đường chéo độ phủ (Coverage Diagonal $D_c$)**: $D_c(p, p) = \#\{R \mid p \in R\}$

---

## 19. Projected SIRT (RMR-P)

Quy trình lặp hòa giải nghiệm giải tích:
1. **Tính sai số tỷ lệ vùng (Residual Rate)**:
   $$e_t = D_a^{-1} (A Y_t - b)$$
2. **Chiếu ngược chuẩn hóa độ phủ (Normalized Backprojection)**:
   $$r_t = D_c^{-1} A^\top e_t$$
3. **Cập nhật và chiếu không âm (Projected Update)**:
   $$Y_{t+1} = \Pi_{\ge 0} \left( Y_t - \omega \, r_t \right)$$

Trong cấu hình Stage C chính thức:
$$T = 2, \quad \omega = 1.0$$
Hoàn toàn phi tham số trong quá trình chiếu, không dùng learned preconditioner hay learned step-size.

---

## 20. Flat-DM16 Trong RMR-v2

Flat-DM16 được kế thừa từ bài học NTPC nhưng được áp dụng riêng biệt trên:
$$\boxed{Y_0}$$
chứ không áp dụng trên kết quả cuối cùng $Y_T$.  
Sự phân công vai trò:
$$\text{Flat-DM16} \implies \text{Nâng cao chất lượng phân bổ không gian của Observer ban đầu}$$
$$\text{Projected SIRT} \implies \text{Hòa giải và sửa sai tại thời điểm suy luận (Runtime Correction)}$$

---

## 21. Count-Normalized FlatDM16 Loss

NLL của phân bố Dirichlet-Multinomial trên toàn bộ 1024 khối tế bào có thể tăng vọt theo số người trong ảnh. Do đó, RMR-v2 chuẩn hóa loss theo từng ảnh:
$$L_{\text{DM16}} = \frac{-\log p(y \mid \alpha)}{\max(N, 1)}$$
Loss mang ý nghĩa xác suất: **Allocation negative log-likelihood per person**.  
Nhờ đó scale của loss giảm từ ~800 xuống còn ~3.5 nats/person, giúp quá trình tối ưu hóa ổn định hoàn hảo.

---

## 22. Hàm Mục Tiêu Huấn Luyện (Training Objective)

$$L_{\text{total}} = \lambda_c L_{\text{count}} + \lambda_{dm} L_{\text{DM16}} + \lambda_y L_{\text{cell}} + \lambda_b L_{\text{region}}$$
Các trọng số cố định trong Stage C:
$$\lambda_c = 1.0, \quad \lambda_{dm} = 1.0, \quad \lambda_y = 0.25, \quad \lambda_b = 0.20$$
$L_{\text{count}}$ sử dụng Negative Binomial crop-count objective để xử lý hiện tượng phân tán dư (overdispersion).

---

## 23. Ma Trận Thực Nghiệm Đối Chứng Stage C (Stage C Matched Matrix)

| Model ID | Variant | Bản chất cơ chế đối chứng |
|:---:|:---|:---|
| **B0** | `direct` | Baseline thuần túy, không có cơ chế vùng |
| **B1** | `region_loss` | Chỉ giám sát vùng qua hàm mất mát lúc huấn luyện ($A Y$) |
| **B2** | `region_aux` | Có head dự đoán vùng $b$ nhưng không dùng để sửa sai lúc suy luận |
| **B3a** | `local_refine` | Tinh chỉnh lặp bằng mạng tích chập cục bộ (Recurrent DW-Conv $T=2$) |
| **B3b** | `learned_project` | Phân bổ sai số vùng bằng mạng nơ-ron học phân bổ cục bộ (Learned Softmax) |
| **B5-P** | `rmr_projected` | **Hòa giải bằng toán tử liên hợp giải tích $A^\top$ trong không gian độ đo** |

---

## 24. Kết Quả Thực Nghiệm Validation Stage C (1000 Epochs, Seed 42)

Dữ liệu chính thức trích xuất từ các file `eval_val/summary.json` trên tập validation ShanghaiTech Part A:

| Model ID | Kiến trúc | Params | Val MAE ↓ | Val RMSE ↓ | Val NAE ↓ | Val Bias | GAME1 ↓ | GAME2 ↓ | GAME3 ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | Direct Baseline | 97,682 | 95.72 | 137.73 | 0.1966 | +15.88 | 120.97 | 146.02 | 189.65 |
| **B1** | Region Loss Supervision | 97,682 | 87.82 | 121.19 | 0.1823 | -9.52 | 105.86 | 132.72 | 183.50 |
| **B2** | Region Aux Feature Sharing | 101,714 | 87.74 | 121.13 | 0.1932 | -21.15 | 118.23 | 147.70 | 189.19 |
| **B3a** | Local Recurrent Refine ($T=2$) | 100,690 | 84.50 | 116.01 | 0.1820 | -15.12 | 99.90 | 127.59 | 179.08 |
| **B5-P** | **Projected SIRT ($T=2$) [Ours]** | **101,714** | <mark>**79.38**</mark> | <mark>**106.17**</mark> | **0.1907** | <mark>**-1.36**</mark> | **102.14** | **128.16** | <mark>**177.57**</mark> |
| **B3b** | Learned Projector ($T=2$) | 104,756 | *(Running E62)* | — | — | — | — | — | — |

*(B3b cần hoàn tất cùng protocol trước khi đóng hoàn toàn claim so sánh giữa exact-adjoint và learned-projector).*

---

## 25. Phân Tích B0 → B1: Giá Trị Của Regional Supervision

Khi chuyển từ B0 sang B1:
$$95.72 \to 87.82 \quad (\Delta\text{MAE} = -7.90)$$
B1 chỉ đơn thuần bổ sung supervision đo lường vùng trên $A Y$. Kết quả này chứng minh rằng việc giám sát tỷ lệ mật độ đa tỷ lệ giúp observer học được cách phân bổ khối lượng tốt hơn.  
Tuy nhiên, B1 hoàn toàn không có cơ chế suy luận vùng tại runtime, do đó không được dùng B1 để minh chứng cho runtime reconciliation.

---

## 26. Phân Tích B1 → B2: Auxiliary Head Không Tự Động Tạo Ra Giá Trị

So sánh giữa B1 và B2:
- B1: MAE **87.82**, RMSE **121.19**
- B2: MAE **87.74**, RMSE **121.13**

Hai mô hình cho kết quả gần như trùng khớp. Điều này chứng minh một nguyên lý quan trọng:
$$\boxed{\text{Dự đoán bằng chứng vùng (Predicting Regional Evidence) } \neq \text{ Sử dụng bằng chứng vùng (Using Regional Evidence)}}$$
Một nhánh phụ dự đoán vùng tốt không tự động cải thiện chất lượng đếm nếu nó chỉ tồn tại như một passive auxiliary task.

---

## 27. Phân Tích B2 → B5-P: Bằng Chứng Nhân Quả Cốt Lõi (Main Causal Result)

So sánh giữa B2 và B5-P:
- B2: MAE **87.74** | RMSE **121.13**
- B5-P: MAE **79.38** | RMSE **106.17**
- Cải thiện: $\Delta\text{MAE} = -8.36$ (-9.5%), $\Delta\text{RMSE} = -14.96$ (-12.4%).

Cả B2 và B5-P sở hữu **chính xác 101,714 tham số**.  
Dưới cùng protocol Stage C cố định, đây là bằng chứng thực nghiệm mạnh mẽ cho thấy việc hòa giải mâu thuẫn hình học lúc suy luận mang lại giá trị vượt trội so với chia sẻ đặc trưng thụ động.

> [!TIP]
> **Diễn đạt an toàn cho bài báo**: *"Under the matched Stage C protocol, explicit runtime regional reconciliation improves over an otherwise parameter-matched regional auxiliary model."* Không nên viết khái quát hóa rằng *"causal hypothesis is universally proven"*.

---

## 28. Phân Tích B3a → B5-P: Ràng Buộc Hình Học vs Mạng Tích Chập Tự Do

So sánh giữa B3a và B5-P:
- B3a: MAE **84.50** | RMSE **116.01**
- B5-P: MAE **79.38** | RMSE **106.17**

B3a bổ sung một mạng Depthwise-Conv hồi quy tự do để tinh chỉnh bản đồ mật độ, nhưng vẫn thua B5-P hơn 5.1 điểm MAE và 9.8 điểm RMSE.  
Điều này chứng minh rằng việc áp đặt toán tử liên hợp giải tích $A^\top$ cung cấp một inductive bias mạnh mẽ hơn nhiều so với việc chỉ tăng thêm dung lượng tích chập cục bộ tự do.

---

## 29. Hành Vi Theo Nhóm Mật Độ (Density-Bin Behavior)

| Model ID | Sparse MAE ($\le 100$) | Moderate MAE ($101 - 500$) | Dense MAE ($> 500$) |
|:---:|:---:|:---:|:---:|
| **B0** | 30.71 | 53.60 | 147.13 |
| **B1** | 27.34 | 52.75 | 131.52 |
| **B2** | 30.24 | 54.04 | 129.65 |
| **B3a** | 29.34 | 54.31 | 122.57 |
| **B5-P** | <mark>**20.47**</mark> | <mark>**70.36**</mark> | <mark>**96.80**</mark> |

---

## 30. Diễn Giải Về Sự Suy Giảm Ở Nhóm Mật Độ Vừa (Moderate-Density Degradation)

Đây là **failure pattern quan trọng nhất** của RMR Stage C hiện tại.  
Trong khi B5-P vượt trội ở nhóm thưa (20.47) và nhóm cực đông (96.80), nó lại bị thụt lùi ở nhóm mật độ vừa (70.36 so với 52 - 54 của các baseline).

*Nguyên nhân cơ chế*:
$$\boxed{\text{Sự tin cậy đồng đều (uniform trust) vào } b_R \text{ đã dẫn đến hiện tượng sửa sai quá mức (over-correction) ở một số cảnh mật độ vừa}}$$
Khi observer ban đầu $Y_0$ đã dự đoán tương đối tốt, nhưng bằng chứng vùng $b_R$ tại cảnh đó có chất lượng kém hơn, bộ giải SIRT với trọng số tin cậy đồng nhất vẫn ép $Y$ phải khớp với $b$, vô tình làm hỏng dự đoán ban đầu.  
**Đây chính là động lực khoa học trực tiếp cho sự ra đời của RMR-v3.**

---

## 31. Hành Vi Triệt Tiêu Sai Số Đuôi (Tail Error Behavior)

- **P90 AE (Phân vị sai số 90%)**: B5-P đạt **156.23** (giảm sâu so với 209.79 của B0 và 209.69 của B2).
- **Max AE (Sai số lớn nhất)**: B5-P đạt **285.71** (so với > 400 của B0 và B2).

RMR đặc biệt hiệu quả trong việc loại bỏ các sai số hệ thống cực đoan, thể hiện qua việc giảm ngoạn mục RMSE và Dense MAE.

---

## 32. Phân Tích Count Bias

B5-P đạt Bias: **-1.36** (so với -21.15 ở B2 và +15.88 ở B0).  
Đây là một kết quả cân bằng thực nghiệm (empirical calibration) xuất sắc.  
Tuy nhiên, cần lưu ý: **Đây không phải là định lý bảo toàn khối lượng toán học tuyệt đối**, vì bước cập nhật của SIRT có phép chiếu không âm $\Pi_{\ge 0}$, nên tổng khối lượng giữa hai bước $\sum Y_{t+1} \neq \sum Y_t$ nói chung.  
*Paper-safe claim*: *"B5-P exhibits near-zero empirical count bias on the validation split."*

---

## 33. Phân Tích Chỉ Số GAME và Không Gian Nullspace

- B5-P không dẫn đầu ở GAME1 và GAME2 (B3a đạt tốt hơn ở thang đo cục bộ).
- B5-P đạt kết quả tốt nhất ở **GAME3 (177.57)**.

*Bản chất cơ chế*:
$$\boxed{\text{B5-P mạnh hơn ở tính nhất quán vĩ mô/vùng (coarse/global consistency) hơn là sai số chia cắt siêu mịn}}$$
Điều này hoàn toàn khớp với lý thuyết không gian triệt tiêu (nullspace) của toán tử vùng.

---

## 34. Lý Thuyết Rank và Nullspace Của Toán Tử Vùng

Với ảnh crop 512, Stride 4:
- Lưới mịn có kích thước $128 \times 128 = 16,384$ ô tế bào ($G = 16,384$).
- Các cửa sổ vùng 32px (bước trượt 16px) chỉ tạo ra khoảng $31 \times 31 = 961$ phép đo độc lập.
- Các phép đo ở thang 64px và 128px phần lớn là tổ hợp tuyến tính của các cửa sổ nhỏ hơn.

Do đó, **Rank của toán tử $A$ nhỏ hơn rất nhiều so với số chiều $G$ của lưới tế bào**.  
Không gian nullspace lớn đồng nghĩa với việc RMR không thể khôi phục các sai số không gian tần số cao tùy ý. Nó chủ yếu triệt tiêu các bất đồng số lượng ở dải tần số thấp/trung bình (regional count disagreement). Điều này giải thích tại sao B5-P giảm mạnh RMSE, cải thiện vượt bậc Dense MAE và GAME3, nhưng không thống trị GAME1/2.

---

## 35. Động Học Hòa Giải Vùng (Regional Reconciliation Dynamics)

Độ bất đồng vùng $|A Y^{(t)} - b|$ qua các bước lặp của B5-P:

| Tỷ lệ vùng | Bước 0 ($Y_0$) | Bước 1 ($Y_1$) | Bước 2 ($Y_2$) | Mức độ suy giảm (%) |
|:---:|:---:|:---:|:---:|:---:|
| **32 px** | 0.542 | 0.229 | **0.196** | **-63.8%** |
| **64 px** | 2.136 | 0.641 | **0.501** | **-76.6%** |
| **128 px** | 7.740 | 2.095 | **1.916** | **-75.2%** |

Sai số so với ground truth cũng giảm đơn điệu (ví dụ thang 128px giảm từ $8.93 \to 4.30$). Đây là bằng chứng thực nghiệm trực tiếp về năng lực hoạt động của toán tử.

---

## 36. Giảm Năng Lượng Đơn Điệu (Energy Monotonicity)

100% mẫu ảnh validation đều thỏa mãn:
$$\mathcal{E}(Y_{t+1}) < \mathcal{E}(Y_t)$$
*Diễn đạt an toàn*: *"B5-P exhibited empirically monotone regional-energy reduction on all validation samples in the Stage C evaluation."* Tránh dùng từ *"provable convergence"* khi chưa có bài chứng minh giải tích riêng cho step-size có trọng số.

---

## 37. Nhất Quán Suy Luận Direct vs Tiled

Độ lệch chuẩn hóa Direct-vs-Tiled (MeanNorm) của B5-P là **0.0272** (95% CI: [0.0209, 0.0341]), hoàn toàn tương đồng với các mô hình baseline (0.024 - 0.028).  
Điều này khẳng định thiết kế vùng hữu hạn cục bộ đã khắc phục triệt để lỗi biên của thời kỳ MICF.

---

## 38. B3b — Learned Projector Control

B3b được thiết kế để kiểm chứng câu hỏi khoa học:
$$A^\top \quad \text{vs} \quad P_\theta$$
B3b sử dụng cùng cập nhật trực tiếp trong không gian độ đo:
$$Y_{t+1} = \Pi_{\ge 0} \left[ Y_t - \omega \, r_\theta \right]$$
với $T=2, \omega=1.0$ và bằng chứng vùng được detach như B5-P để đảm bảo tính công bằng tối đa.

---

## 39. Điểm Nghẽn Cài Đặt Của B3b (Implementation Bottleneck)

Bản cài đặt tham chiếu của B3b duyệt tuần tự qua từng vùng trong số 1,235 bounding boxes bằng vòng lặp Python:
```python
for m, box in enumerate(boxes_list):
    y1, x1, y2, x2 = map(int, box)
    pi = torch.softmax(score[:, :, y1:y2, x1:x2].flatten(-2), dim=-1)
    out[:, :, y1:y2, x1:x2] += delta[:, :, m] * pi
```
Điều này tạo ra hàng nghìn CUDA kernel launches nhỏ và đồ thị autograd cồng kềnh, khiến B3b mất ~175 giây / epoch (chậm hơn 10.7 lần so với B5-P).  
*Cảnh báo*: Không được viết "learned projection intrinsically chậm". Phải ghi rõ: *"Current explicit-region reference implementation is substantially slower"*.

---

## 40. Kết Luận Khoa Học Của RMR-v2 (Supported Claims)

Stage C hỗ trợ mạnh mẽ 4 tuyên bố khoa học sau:
- **Claim A (Regional supervision helps)**: Giám sát vùng có ích ($B1 > B0$).
- **Claim B (Passive prediction is not enough)**: Dự đoán vùng thụ động là chưa đủ ($B2 \approx B1$).
- **Claim C (Runtime exact reconciliation adds value)**: Hòa giải giải tích lúc suy luận tạo ra bước nhảy vọt ($B5\text{-P} > B2$).
- **Claim D (Regional algebra helps tail/dense errors)**: Đại số vùng triệt tiêu sai số cực đoan và giải quyết vùng cực đông.

---

## 41. Các Tuyên Bố Chưa Đủ Bằng Chứng (Unsupported Claims)

Tuyệt đối không đưa vào bài báo các tuyên bố sau:
- "Exact projection universally beats learned projection" (cần đợi B3b hoàn tất hoặc có bằng chứng matched);
- "Exact mass conservation";
- "Global mathematical convergence proof";
- "All gradient conflicts eliminated";
- "Translation invariance theorem";
- "RMR uniformly improves every density regime" (vì nhóm moderate bị giảm);
- "RMR wins every spatial metric" (vì GAME1/2 không thắng).

---

## 42. Tại Sao Chưa Nên Đổi Backbone?

Backbone hiện tại kết hợp chuẩn hóa đã đưa MAE từ ~239 xuống **79.38**.  
Failure pattern hiện tại không phải do backbone thiếu dung lượng biểu diễn, mà do **bộ giải đang tin tưởng bằng chứng vùng một cách mù quáng ngay cả khi nó không đáng tin cậy**. Việc tăng kích thước backbone lúc này sẽ làm mờ đi bản chất của giả thuyết nghiên cứu.

---

## 43. Động Lực Cho RMR-v3 (Motivation)

Bộ giải B5-P hiện tại coi mọi quan sát vùng $b_R$ đều có độ tin cậy ngang nhau sau khi chuẩn hóa diện tích.  
Tuy nhiên, chất lượng dự đoán vùng không hề đồng đều giữa các cảnh, thang đo và mức độ tắc nghẽn (occlusion). Một giá trị $b_R$ sai lệch có thể bóp méo một bản đồ mật độ $Y_0$ vốn đang chuẩn xác.  
**Mục tiêu của RMR-v3 là trang bị khả năng tự đánh giá độ tin cậy cho từng quan sát vùng.**

---

## 44. RMR-v3: Reliability-Weighted Regional Measure Reconciliation

Regional head mới sẽ dự đoán phân bố xác suất cho từng vùng thay vì một giá trị điểm:
$$(\mu_R, \sigma_R) \quad \text{hoặc} \quad (\mu_R, r_R)$$
trong đó $\mu_R$ là kỳ vọng số lượng vùng, và $\sigma_R$ (hoặc độ phân tán $r_R$) thể hiện độ bất định dự đoán (predictive uncertainty).

---

## 45. Hàm Năng Lượng Có Trọng Số Tin Cậy (Weighted Regional Energy)

Thay thế hàm năng lượng bình phương thông thường:
$$\mathcal{E}(Y) = \frac{1}{2} (A Y - b)^\top D_a^{-1} (A Y - b)$$
bằng hàm năng lượng có trọng số tin cậy:
$$\boxed{\mathcal{E}_w(Y) = \frac{1}{2} (A Y - b)^\top W D_a^{-1} (A Y - b)}$$
với ma trận trọng số đường chéo $W = \operatorname{diag}(w_R)$, trong đó $w_R$ là độ tin cậy được chuẩn hóa theo thang đo từ nghịch đảo phương sai mật độ (rate precision):
$$q_R = \frac{1}{V_R^{\text{rate}}}, \quad \bar{q}_s = \frac{1}{|S_s|} \sum_{R' \in S_s} q_{R'}, \quad w_R = \operatorname{clamp}\left(\frac{q_R}{\bar{q}_s}, \; 0.25, \; 4.0\right)$$

---

## 46. Toán Tử Cập Nhật Có Trọng Số Giải Tích (Weighted Exact Update)

Độ phủ có trọng số (Weighted Coverage Diagonal):
$$D_{c,w} = \operatorname{diag}(A^\top w)$$
Bước lặp cập nhật chính xác:
$$\boxed{r = D_{c,w}^{-1} A^\top W D_a^{-1} (A Y - b)}$$
$$Y_{t+1} = \Pi_{\ge 0} \left( Y_t - \omega \, r \right)$$
với $T=2, \omega=1.0$.

---

## 47. Bảo Toàn Tính Đồng Nhất Toán Học (Mathematical Identity Preservation)

Nếu mọi vùng có cùng một mật độ sai số dư $\delta$:
$$D_a^{-1} (A Y - b) = \delta \mathbf{1}$$
thì tử số của phép chiếu ngược là:
$$A^\top W (\delta \mathbf{1}) = \delta A^\top w$$
Mẫu số chuẩn hóa độ phủ:
$$D_{c,w} = \operatorname{diag}(A^\top w)$$
Khi chia tử số cho mẫu số:
$$r = \frac{\delta A^\top w}{A^\top w} = \delta \mathbf{1}$$
$$\boxed{\text{Mật độ sai số vùng đồng nhất } \implies \text{ Hiệu chỉnh không gian đồng nhất}}$$
Nguyên lý cân bằng không gian được bảo toàn tuyệt đối ngay cả khi các vùng có trọng số tin cậy khác nhau.

---

## 48. Thiết Kế Reliability Head

Không dùng Attention. Thiết kế siêu nhẹ dựa trên MLP chung:
```
Regional Pooled Feature (d=33)
        ↓
Shared MLP Trunk (33 → 48 → 48)
      ├── Mean Head (48 → 1, Softplus) → mu_R
      └── Uncertainty Head (48 → 1, Softplus) → sigma_R (hoặc r_R)
```
Gia tăng tham số cực nhỏ (< 3k tham số).

---

## 49. Hàm Khả Năng Xác Suất (Probabilistic Regional Likelihood)

Giả định số lượng người trong vùng tuân theo phân bố Negative Binomial:
$$N_R \sim \operatorname{NB}(\mu_R, r_R)$$
Phương sai lý thuyết theo count và rate:
$$\operatorname{Var}(N_R) = \mu_R + \frac{\mu_R^2}{r_R}, \quad V_R^{\text{rate}} = \frac{\operatorname{Var}(N_R)}{|R|^2} + \sigma_{\min}^2$$
Độ tin cậy được suy biến từ nghịch đảo phương sai (precision), chuẩn hóa trung bình bằng 1 trong từng scale family và chặn giới hạn:
$$q_R = \frac{1}{V_R^{\text{rate}}}, \quad \bar{q}_s = \frac{1}{|S_s|} \sum_{R' \in S_s} q_{R'}, \quad w_R = \operatorname{clamp}\left(\frac{q_R}{\bar{q}_s}, \; 0.25, \; 4.0\right)$$

---

## 50. Ngăn Chặn Gian Lận Độ Bất Định (Proper Likelihood Regularization)

Mạng không được huấn luyện bằng hàm mất mát kiểu $\text{confidence} \times \text{error}$ tự do vì mạng sẽ có xu hướng tăng vô hạn phương sai để triệt tiêu loss.  
Hàm mất mát Negative Binomial NLL tự động phạt độ bất định lớn:
$$-\log p(N_R \mid \mu_R, r_R)$$
ép mạng phải cân bằng chính xác giữa độ chính xác dự đoán và độ tự tin.

---

## 51. Giả Thuyết Khoa Học Trọng Tâm Của RMR-v3

$$\boxed{\textbf{Can uncertainty-calibrated regional evidence preserve RMR's dense/tail gains while preventing harmful over-correction in moderate-density scenes?}}$$

---

## 52. Kế Hoạch Thực Nghiệm Tối Thiểu (Minimal Experiment Plan)

| Biến thể | Dự đoán $\mu_R$ | Dự đoán Uncertainty | Runtime Weighted Solver | Ý nghĩa kiểm chứng |
|:---:|:---:|:---:|:---:|:---|
| **V2** | Có | Không | Không (Uniform) | Baseline B5-P hiện tại |
| **V3-A** | Có | Có | Không (Uniform) | Kiểm tra việc học thêm uncertainty có tự cải thiện feature representation không |
| **V3-B** | Có | Có | **Có ($W$)** | **Kiểm tra giá trị gia tăng của bộ giải weighted reconciliation** |

---

## 53. Bộ Chỉ Số Đánh Giá Bắt Buộc Cho RMR-v3

Không chỉ nhìn vào headline MAE. Phải theo dõi chặt chẽ:
$$\boxed{\text{Moderate MAE}} \quad \text{và} \quad \boxed{\text{Dense MAE}}$$
- Mục tiêu chính: Hạ rõ rệt Moderate MAE từ mức **70.36** xuống gần mức baseline (~55), đồng thời **giữ vững Dense MAE ở ngưỡng kỉ lục 96.80**.

---

## 54. Chẩn Đoán Độ Tin Cậy Bắt Buộc (Reliability Diagnostics)

Phải log và báo cáo:
1. Hệ số tương quan Pearson giữa predicted variance và absolute regional error;
2. Calibration error bins;
3. Trọng số trung bình theo thang đo ($32, 64, 128\text{px}$);
4. Trọng số trung bình theo nhóm mật độ (Sparse, Moderate, Dense);
5. Tỷ lệ các vùng bị đánh giá độ tin cậy thấp ($w_R \approx 0$).

---

## 55. Xử Lý Trọng Số Theo Thang Đo (Scale Reliability)

Chỉ sau khi V3-B thành công mới xem xét kết hợp thêm trọng số tĩnh theo thang đo:
$$W_R = w_R^{\text{instance}} \times w_{\text{scale}}$$
Không đưa vào ngay từ đầu để tránh gây nhiễu nguồn gốc cải thiện (confounder).

---

## 56. Danh Sách Các Thành Phần Cấm Đưa Vào RMR-v3

Tuyệt đối không đưa vào RMR-v3 các thành phần:
- Transformer, Self-Attention, C32 expansion;
- Full-image region constraint;
- Tree-Pólya, quadtree hierarchical loss;
- MICF cumulative representation;
- Learned preconditioner, learned step-size $\omega$;
- Dynamic iteration count, Mixture-of-Experts (MoE), PointMass, Voronoi, SSER.

---

## 57. Giao Thức Đóng Băng RMR-v2 (Freeze Protocol)

Toàn bộ hiện trạng Stage C được đóng băng nguyên vẹn tại Git commit [`65eba08`](https://github.com/minhphuc477/crowd-counting-lightweight/commit/65eba08722c4d356c3b7becdafbd618b3ae209de).  
Không sửa đổi bất kỳ checkpoint, config hay code nào của B5-P reference.

---

## 58. Chính Sách Quản Lý Test Set

Tập test `data/sha_a_test.jsonl` được **giữ đông lạnh 100%**.  
Chỉ tiến hành đánh giá đúng một lần duy nhất trên test set sau khi hoàn thành freeze RMR-v2 và chốt thiết kế RMR-v3.

---

## 59. Định Vị Bài Báo Khoa Học (Paper Positioning)

Đóng góp chính của công trình không phải là "một backbone mới", vì backbone là MobileNetV4 chuẩn từ cộng đồng.  
Đóng góp khoa học cốt lõi là:
1. **Independent regional evidence estimation**;
2. **Exact regional forward and adjoint operators**;
3. **Non-negative measure-space reconciliation**;
4. **Matched causal analysis against passive, local, and learned controls**;
5. *(Nếu v3 thành công)* **Uncertainty-calibrated geometric reconciliation**.

---

## 60. Tuyên Bố Trung Tâm An Toàn (Paper-Safe Central Claim)

> *"Independent fine-scale and regional count estimators can disagree. Under an ultra-light matched architecture (< 105k parameters), explicitly reconciling their disagreement through known regional count geometry improves count accuracy and reduces large systematic errors more effectively than passive auxiliary regional prediction or local iterative refinement."*

---

## 61. Tóm Tắt Toàn Bộ Tiến Trình Phát Triển

```
MICF
 │
 ├─ Phát hiện: Trường tích phân toàn cục phụ thuộc extent cực kỳ mong manh.
 └─ Giữ lại: Ràng buộc hình học cục bộ hữu hạn (Finite-support geometry).
      ↓
NTPC
 │
 ├─ Phát hiện: Cây phân cấp không tự động tốt hơn; Flat-DM16 mạnh mẽ và sạch hơn.
 └─ Giữ lại: Pretrained carrier siêu nhẹ + Flat-DM16.
      ↓
RMR Pilot
 │
 ├─ Phát hiện: Observer scratch quá yếu; bộ giải latent không thể sinh khối lượng thưa.
 └─ Giữ lại: Hình học vùng giải tích chính xác.
      ↓
RMR-v2 (Stage C)
 │
 ├─ MobileNetV4 pretrained + Flat-DM16 trên Y0.
 ├─ Scale-matched independent regional head.
 ├─ Projected measure-space SIRT (B5-P: MAE = 79.38, RMSE = 106.17).
 └─ Phát hiện failure pattern: Over-correction ở nhóm moderate density.
      ↓
RMR-v3
 └─ Reliability-Weighted Regional Measure Reconciliation (RW-RMR).
```

---

## 62. Nguyên Tắc Thiết Kế Tối Hậu (Final Design Principle)

$$\boxed{\textbf{Learn Perception} \quad + \quad \textbf{Learn Uncertainty} \quad + \quad \textbf{Keep Known Geometry Exact}}$$

Cụ thể hóa:
$$\boxed{\text{MobileNetV4} \implies Y_0}$$
$$\boxed{P_4 / P_8 / P_{16} \implies (\mu_R, \sigma_R)}$$
$$\boxed{A, A^\top, D_a, D_{c,w} \implies \text{Exact Weighted Reconciliation}}$$

---

## 63. Danh Mục Các Artifact Đã Đóng Băng Cùng Tài Liệu

Tất cả các file số liệu canonical đã được khởi tạo và lưu trữ vĩnh viễn tại thư mục [`reports/stage_c_freeze/`](file:///f:/lightweightcrcn/reports/stage_c_freeze/):
1. [`stage_c_results.csv`](file:///f:/lightweightcrcn/reports/stage_c_freeze/stage_c_results.csv): Bảng kết quả tổng hợp chính của 6 model.
2. [`stage_c_density_bins.csv`](file:///f:/lightweightcrcn/reports/stage_c_freeze/stage_c_density_bins.csv): Phân rã sai số theo 3 nhóm mật độ.
3. [`stage_c_game.csv`](file:///f:/lightweightcrcn/reports/stage_c_freeze/stage_c_game.csv): Chỉ số định vị không gian GAME(0..3).
4. [`stage_c_region_dynamics.csv`](file:///f:/lightweightcrcn/reports/stage_c_freeze/stage_c_region_dynamics.csv): Động học suy giảm bất đồng vùng qua các bước lặp.
5. [`stage_c_tiling_consistency.csv`](file:///f:/lightweightcrcn/reports/stage_c_freeze/stage_c_tiling_consistency.csv): Kiểm toán tính nhất quán Direct vs Tiled.
6. [`stage_c_profile.csv`](file:///f:/lightweightcrcn/reports/stage_c_freeze/stage_c_profile.csv): Đo lường tham số, thời gian chạy và thông lượng.
7. [`stage_c_commit.txt`](file:///f:/lightweightcrcn/reports/stage_c_freeze/stage_c_commit.txt): Biên bản ghi nhận commit SHA, branch và siêu tham số.

---

## 64. Lời Kết

Kết quả quan trọng nhất của toàn bộ chặng đường nghiên cứu không chỉ nằm ở con số **79.38 MAE**, mà nằm ở chỗ dự án đã phân tách rành mạch ba chức năng:
1. **Quan sát thị giác (Visual Observation)**
2. **Phân bổ không gian (Spatial Allocation)**
3. **Hòa giải mâu thuẫn vùng (Regional Consistency Correction)**

MICF thất bại khi biểu diễn bị hòa lẫn với hình học toàn ảnh. NTPC thất bại khi giả định cây phân cấp một cách võ đoán trong khi phân bổ phẳng lại tối ưu hơn. RMR Pilot thất bại khi bắt bộ giải phải cứu vãn một observer yếu kém.

RMR-v2 Stage C thành công vang dội vì đã đặt đúng vai trò cho từng thành phần. Và RMR-v3 sẽ giải quyết chính xác failure pattern cuối cùng: **Trang bị khả năng tự đánh giá độ tin cậy để bộ giải giải tích không sửa sai quá mức ở các cảnh mật độ vừa, đưa kiến trúc đạt tới trạng thái hoàn thiện toàn diện.**
