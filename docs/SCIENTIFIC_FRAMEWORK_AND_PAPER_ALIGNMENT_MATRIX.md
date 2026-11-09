# KHUNG KHOA HỌC & MA TRẬN CĂN CHỈNH BÀI BÁO CHUẨN MỰC
## Constrained Continuous Radon Measure Recovery for Ultra-Lightweight Crowd Counting ($\le 104\text{k}$ Parameters, 0% KD)

> **Tài liệu Định hướng Học thuật Gốc (Canonical Academic Specification)**  
> **Nguyên tắc cốt lõi (Rule 17):** Tuyệt đối không lạm dụng thuật ngữ "Pareto" trong Problem Statement, RQ và RO. Thuật ngữ "Đường biên Pareto" chỉ xuất hiện duy nhất ở phần *Thực nghiệm & Thảo luận* dưới dạng biểu đồ phân tán hậu nghiệm (Post-hoc Benchmarking Scatter Plot: MAE vs Params/FLOPs).  
> **Tính trung thực khoa học:** Không dùng từ ngữ thổi phồng (zero hype). Thừa nhận và phân tích trực diện các điểm nghẽn kỹ thuật thực tế (bias âm -19 đến -24 người/ảnh, sương mù nền chiếm 64.4% khối lượng dự đoán do bẫy bão hòa Softplus, Dense MAE vọt lên 127.67).

---

## 1. BẢN CHẤT BÀI TOÁN & CÁC ĐIỂM NGHẼN KỸ THUẬT (PROBLEM STATEMENT & ROOT CAUSES)

### 1.1. Bản Chất Toán Học của Bài Toán Đếm Đám Đông
Về mặt giải tích và thị giác máy tính, bài toán đếm đám đông không phải là bài toán tối ưu đa mục tiêu (Multi-Objective Optimization). Đây là **bài toán nghịch đảo khôi phục độ đo Radon liên tục (Constrained Radon Measure Recovery)** từ quan sát ảnh 2D bị suy thoái:
$$b = \mathcal{A}_\pi y + \eta$$
Trong đó:
- $y \in \mathcal{M}^+(\Omega)$ là độ đo mật độ thực cần khôi phục trên miền ảnh $\Omega \subset \mathbb{R}^2$.
- $\mathcal{A}_\pi$ là toán tử chiếu phối cảnh suy thoái (perspective degradation operator).
- $b$ là quan sát ảnh đầu vào có nhiễu $\eta$.

### 1.2. Ba Điểm Nghẽn Cốt Tử Khi Ràng Buộc Ngân Sách Cực Tiểu ($\le 104\text{k}$ Params, 0% KD)
1. **Sự sụp đổ cân bằng học đường tắt (Shortcut Learning Equilibrium):**
   Dưới các hàm loss đếm tổng thể và phân bổ điểm, mô hình nơ-ron nhỏ rơi vào trạng thái cân bằng thoái hóa: phân bổ khoảng $35.6\%$ khối lượng vào các vị trí đầu người thật, và làm rò rỉ tới $64.4\%$ khối lượng còn lại vào nền rỗng dưới dạng một lớp sương mù đồng nhất ($\approx 0.00124\text{ count/cell}$). Trên toàn bộ ảnh kích thước lớn, lượng rò rỉ này tích tụ thành hơn 200 "người ảo" (phantom people).
2. **Bẫy bão hòa gradient lồi của hàm kích hoạt (Convex Activation Trap):**
   Do đạo hàm của hàm Softplus suy giảm theo hàm mũ khi logit âm ($\sigma(z) \to 0$ khi $z < -6$), gradient descent bị triệt tiêu khi cố gắng đẩy logit vùng nền xuống sâu hơn dưới tác động của weight decay. Kết quả là mô hình bị kẹt ở mức logit nền khoảng $-7.7$, không thể triệt tiêu hoàn toàn sương mù nền.
3. **Hiện tượng Under-counting có hệ thống và Sụp đổ ở cảnh đông (Dense Failure):**
   Trong các bức ảnh có mật độ cực cao (hơn 1,000 người), khoảng cách giữa các đầu người nhỏ hơn bước sải đặc trưng ($d < 4\text{px}$). Các đặc trưng bị bão hòa, dẫn đến bias âm có hệ thống (mô hình đếm thiếu từ 19 đến 24 người/ảnh trên ShanghaiTech A) và sai số ở vùng đông (Dense MAE) tăng vọt lên tới $127.67$ (chênh lệch gấp 8 lần so với vùng thưa Sparse MAE $15.63$).

---

## 2. MA TRẬN CĂN CHỈNH KHOA HỌC (THE ALIGNMENT MATRIX)

Bảng căn chỉnh ánh xạ $1-1$ giữa bài toán mở thực tế, câu hỏi nghiên cứu (RQ), mục tiêu nghiên cứu (RO), phương pháp giải quyết, bằng chứng thực nghiệm và giới hạn kỹ thuật:

| Thành phần | RQ1: Giới hạn Băng thông Không gian - Tần số (Spatial Bandwidth & Sampling) | RQ2: Cơ chế Cổng Hỗ trợ & Dị phương sai Thống kê (Support Gating & Heteroscedasticity) | RQ3: Tính Thích ứng Tiên nghiệm Phân bố Đa Miền (Cross-Domain Prior Adaptation) |
| :--- | :--- | :--- | :--- |
| **Problem Statement** | Khi không sử dụng mạng Teacher chưng cất tri thức (0% KD), backbone cực nhẹ ($\approx 87\text{k}$ params) bị suy giảm độ phân giải không gian nghiêm trọng ở bước sải thô, làm mất dấu các xung điểm cá thể trong cụm đám đông đặc. | Hàm kích hoạt Softplus rơi vào bẫy bão hòa gradient lồi khiến 64.4% khối lượng bị rò rỉ vào nền (sương mù nền); đồng thời giả định sai số đồng phương sai ($L_1/L_2$) gây sai số đếm có hệ thống khi mật độ biến thiên từ $10^0$ đến $10^4$. | Phân bố mật độ và góc nhìn camera biến thiên cực lớn giữa các môi trường thực tế (chênh lệch hơn 6 lần giữa SHA và SHB/QNRF), khiến các mô hình dùng tham số tiên nghiệm cố định bị lệch phân bố nghiêm trọng. |
| **Research Question (RQ)** | Năng lực biểu diễn không gian - tần số của mạng nơ-ron $\le 104\text{k}$ tham số bị giới hạn như thế nào khi tái tạo các xung điểm rời rạc, và cấu trúc nào dung hòa được giữa trường tiếp nhận ngữ cảnh và độ phân giải xung điểm? | Làm thế nào để giải phóng bẫy bão hòa gradient lồi của hàm kích hoạt để triệt tiêu sương mù nền, đồng thời mô hình hóa được hiện tượng bùng nổ phương sai (over-dispersion) của dữ liệu đếm rời rạc? | Cơ chế thích ứng nào cho phép mô hình tự động hiệu chỉnh mật độ tiên nghiệm $m_0$ và trường nhìn phối cảnh theo chiều dọc camera mà không phụ thuộc vào các ngưỡng gán cứng thủ công? |
| **Research Objective (RO)** | Xây dựng khung phân rã độ đo liên tục kết hợp Sliced Radon-OT (1D closed-form CDF) và phân bổ điều kiện Bayes; giải phẫu ranh giới hiệu quả giữa khối cổ mạng ASPP đa tỉ lệ và phép chiếu trực tiếp $1\times 1$ Subpixel. | Thiết kế cơ chế cổng hỗ trợ phi tuyến (Decoupled Support Gating) tách biệt việc phát hiện có người và ước lượng cường độ; tích hợp hàm hợp lý Negative-Binomial phân tán dị phương sai cho giám sát số lượng vĩ mô. | Phát triển cơ chế tự động hiệu chuẩn mật độ tiên nghiệm $m_0$ động theo manifest dữ liệu; đánh giá tính khái quát hóa và độ ổn định xuyên miền trên 5 tập benchmark chuẩn quốc tế. |
| **Proposed Method** | **Dual Radon-OT + Bayesian Loss & Subpixel Reconstruction:** 4 góc chiếu Radon $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$ bảo toàn khối lượng toàn cục với chi phí $O(K \log K)$; đầu ra PixelShuffle Stride-2 giữ nguyên chi tiết điểm. | **Negative-Binomial Count Head & Support Gating:** Giám sát song song kỳ vọng $\mu(x)$ và tham số phân tán $r(x)$ qua Negative Log-Likelihood; dập tắt kích hoạt nền rỗng để ngăn rò rỉ khối lượng. | **Adaptive Manifest Normalization & Multi-Scale Pipeline:** Chuẩn hóa động kích thước ảnh $\le 1024$, tính toán $m_0$ tự động từ nhãn huấn luyện, và áp dụng crowd-aware crop để cân bằng phân bố mẫu. |
| **Empirical Evidence** | **[C]** Trên SHA: No-Neck ($1\times 1$ conv, 102k params) đạt **66.24 MAE**, vượt ASPP Neck (**74.29 MAE**). Ngược lại trên SHB: ASPP Neck đạt **10.49 MAE** vs No-Neck **10.94 MAE**. Pure Radon-OT đơn lẻ kém hơn Flagship (SHA: 84.54 vs 74.29; JHU: 74.55 vs 63.63). | **[C]** Negative-Binomial vượt trội áp đảo $L_1$ count loss trên cả 4 dataset: JHU: **63.63** vs **76.35** ($\Delta = +12.72$). QNRF: **109.80** vs **125.65** ($\Delta = +15.85$). SHA: **74.29** vs **89.33** ($\Delta = +15.04$). SHB: **10.49** vs **11.89** ($\Delta = +1.40$). | **[C]** Mô hình đạt độ ổn định cao trên 5 bộ dữ liệu: JHU Val: **61.72 MAE** (Pretrained) / **63.63 MAE** (Scratch); QNRF Test: **105.36 MAE** (Pretrained) / **109.80 MAE** (Scratch); UCF_CC_50: **$298.65 \pm 76.19$ MAE** (5-Fold CV). |
| **Limitation & Failure Mode** | Khi khoảng cách điểm $< 4\text{px}$, các bộ lọc dilated conv trong ASPP gây hiện tượng răng cưa không gian (comb-aliasing); trong khi phép chiếu $1\times 1$ lại thiếu ngữ cảnh ở ảnh góc rộng. | Khối lượng sương mù nền vẫn chiếm một phần đáng kể nếu tham số khởi tạo bias không đủ sâu; tham số phân tán $r$ cần được kẹp giá trị cẩn thận để tránh mất ổn định ở epoch đầu. | Trên ảnh có kích thước cực lớn (> 4000px) nhưng mật độ người rất thấp (nhiều tán cây, kiến trúc lặp lại), mô hình vẫn có xu hướng báo dương nhẹ nếu không có crowd-aware sampling. |

---

## 3. BA ĐÓNG GÓP KHOA HỌC CHUẨN MỰC (CORE SCIENTIFIC CONTRIBUTIONS)

1. **Khung Khôi Phục Độ Đo Liên Tục Giải Tích Dạng Đóng (Closed-Form Continuous Measure Recovery):**
   * Chúng tôi giải bài toán bảo toàn khối lượng toàn cục trong đếm đám đông bằng cách phân rã bài toán Wasserstein 2D sang tích phân các bài toán vận chuyển tối ưu 1D dọc theo các góc chiếu Radon ($\mathcal{SW}_1$). Nghiệm giải tích dạng đóng thông qua hàm phân phối tích lũy (CDF) đạt độ phức tạp $O(K \log K)$, cho phép tối ưu trực tiếp trên mạng siêu nhẹ mà không cần ma trận lặp Sinkhorn đắt đỏ và không cần làm mờ Gaussian heuristic.

2. **Cơ Chế Giám Sát Bất Định Dị Phương Sai & Phân Tách Không Gian (Heteroscedastic Uncertainty Modeling & Spatial Support):**
   * Chúng tôi chứng minh về mặt giải tích và thực nghiệm rằng hàm mất mát hồi quy $L_1/L_2$ phá vỡ cân bằng tối ưu do giả định sai về phương sai đồng nhất. Việc đưa hàm hợp lý rời rạc Negative-Binomial vào giám sát số lượng vĩ mô kết hợp với phân bổ điểm Bayes cục bộ đã triệt tiêu đáng kể hiện tượng under-counting có hệ thống, cải thiện từ $+1.4$ đến $+15.8\text{ MAE}$ đồng thời trên 4 tập dữ liệu độc lập.

3. **Khảo Sát Thực Nghiệm Đa Trung Tâm & Phân Tích Đánh Đổi Phần Cứng (Comprehensive Multi-Dataset Empirical Study):**
   * Chúng tôi cung cấp một nghiên cứu thực nghiệm có kiểm soát với 70 cấu hình trên 5 benchmark quốc tế (JHU-CROWD++, UCF-QNRF, ShanghaiTech A/B, UCF_CC_50). Trong phần Thảo luận thực nghiệm, chúng tôi phân tích mối quan hệ đánh đổi thực tế (Empirical Trade-off: MAE vs Params/FLOPs), chứng minh rằng mạng $\le 104\text{k}$ tham số huấn luyện hoàn toàn từ đầu (0% KD) đạt hiệu năng cạnh tranh với các mô hình lớn hơn nhiều lần.

---

## 4. DỮ LIỆU THỰC NGHIỆM ĐÃ KIỂM TOÁN (5 BENCHMARKS, 70 CONFIGS)

*(Trích xuất trực tiếp từ `train_log.csv` và `summary.json`, Commit `74257fa`, hạt giống cố định `seed: 42`)*

### 4.1. Bảng Tổng Hợp 5 Tập Dữ Liệu Chuẩn Quốc Tế

| Tập Dữ Liệu | Giao thức Đánh giá | Nghị quyết / Resize | Số lượng Đầu (Train / Test) | Biến thể Mô hình | Tham số | MACs (FLOPs) | MAE | RMSE | NAE | Bias |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **JHU-CROWD++** | Official Val (500 imgs) | Dynamic $\le 1024$ | 844k / 521k | Pretrained | 104,006 | ~0.11 G | **61.72** | **227.79** | 0.285 | -13.54 |
| **JHU-CROWD++** | Official Val (500 imgs) | Dynamic $\le 1024$ | 844k / 521k | Scratch (Flagship) | 104,006 | ~0.11 G | **63.63** | **254.78** | 0.259 | -30.77 |
| **UCF-QNRF** | Official Test (334 imgs)| Dynamic $\le 1024$ | 974k / 277k | Pretrained | 104,006 | ~0.11 G | **105.36** | **174.94** | 0.161 | -5.94 |
| **UCF-QNRF** | Official Test (334 imgs)| Dynamic $\le 1024$ | 974k / 277k | Scratch (Flagship) | 104,006 | ~0.11 G | **109.80** | **189.08** | 0.152 | -38.16 |
| **ShanghaiTech A**| Official Test (182 imgs)| Dynamic $\le 1024$ | 124k / 60k | No-Neck (Scratch) | **102,246** | **~0.10 G** | **66.24** | **102.27** | 0.182 | -15.62 |
| **ShanghaiTech A**| Official Test (182 imgs)| Dynamic $\le 1024$ | 124k / 60k | Pretrained | 104,006 | ~0.11 G | **67.57** | **110.77** | 0.165 | -19.74 |
| **ShanghaiTech A**| Official Test (182 imgs)| Dynamic $\le 1024$ | 124k / 60k | Scratch (Flagship) | 104,006 | ~0.11 G | **74.29** | **118.23** | 0.182 | -24.00 |
| **ShanghaiTech B**| Official Test (316 imgs)| Dynamic $\le 1024$ | 53k / 35k | Pretrained | 104,006 | ~0.11 G | **10.19** | **15.88** | 0.084 | -2.33 |
| **ShanghaiTech B**| Official Test (316 imgs)| Dynamic $\le 1024$ | 53k / 35k | Scratch (Flagship) | 104,006 | ~0.11 G | **10.49** | **16.98** | 0.088 | -3.79 |
| **ShanghaiTech B**| Official Test (316 imgs)| Dynamic $\le 1024$ | 53k / 35k | BG Ratio 0.05 | 104,006 | ~0.11 G | **9.41** | **14.12** | 0.080 | -2.52 |
| **UCF_CC_50** | 5-Fold Cross-Val | Dynamic $\le 1024$ | Total 50 imgs | Scratch (Mean $\pm$ Std) | 104,006 | ~0.11 G | **298.65 $\pm$ 76.19** | **466.66 $\pm$ 208.79** | 0.425 | -30.54 |
| **UCF_CC_50** | 5-Fold Cross-Val | Dynamic $\le 1024$ | Total 50 imgs | Pretrained (Mean $\pm$ Std) | 104,006 | ~0.11 G | **329.99 $\pm$ 90.00** | **492.00 $\pm$ 199.12** | 0.355 | -49.33 |

---

### 4.2. Bằng Chứng Đối Chứng Bất Biến Giữa Các Giả Thuyết Khoa Học

#### Giả Thuyết 1: Giám sát Số lượng Negative-Binomial vs Naive $L_1$
* **JHU:** Flagship (NB): **63.63** vs. $L_1$ Count: **76.35** ($\Delta = +12.72\text{ MAE}$)
* **QNRF:** Flagship (NB): **109.80** vs. $L_1$ Count: **125.65** ($\Delta = +15.85\text{ MAE}$) vs. No Count: **143.83** ($\Delta = +34.03\text{ MAE}$)
* **SHA:** Flagship (NB): **74.29** vs. $L_1$ Count: **89.33** ($\Delta = +15.04\text{ MAE}$)
* **SHB:** Flagship (NB): **10.49** vs. $L_1$ Count: **11.89** ($\Delta = +1.40\text{ MAE}$)

#### Giả Thuyết 2: Vai trò của Sliced Radon-OT và Bayesian Likelihood
* **JHU:** Flagship: **63.63** vs. Pure ROT: **74.55** vs. Gaussian MSE: **70.80**
* **QNRF:** Flagship: **109.80** vs. Pure ROT: **122.68** vs. Gaussian MSE: **119.51**
* **SHA:** Flagship: **74.29** vs. Pure ROT: **84.54**
* **SHB:** Flagship: **10.49** vs. Pure ROT: **13.83** vs. Gaussian MSE: **11.85**
* *Nhận xét:* Pure Radon-OT có Bias toàn cục sát 0 nhất trên mọi tập dữ liệu (JHU: $+2.60$, QNRF: $+2.31$, SHA: $+4.76$, SHB: $+4.12$), chứng minh tính bảo toàn khối lượng chính xác; nhưng bắt buộc cần thành phần Bayesian Loss để định vị điểm nhọn cục bộ.

#### Giả Thuyết 3: Phép Chiếu Trực Tiếp $1\times 1$ Subpixel vs ASPP Dilated Neck
* Trên **ShanghaiTech Part A (Đám đông siêu đặc, khoảng cách điểm $< 4\text{px}$):**
  * Mô hình **No-Neck** ($1\times 1$ conv trực tiếp, $102,246$ params): **66.24 MAE | 102.27 RMSE**
  * Mô hình **Flagship có ASPP Neck** (dilated conv, $104,006$ params): **74.29 MAE | 118.23 RMSE**
  * *Nguyên nhân:* Tích chập giãn nở làm suy hao thông tin tần số cao khi khoảng cách điểm nhỏ hơn bước nhảy giãn nở.
* Trên **ShanghaiTech Part B (Mật độ thưa, biến thiên góc nhìn sâu):**
  * Mô hình **Flagship có ASPP Neck**: **10.49 MAE | 16.98 RMSE**
  * Mô hình **No-Neck**: **10.94 MAE | 18.49 RMSE**
  * *Nguyên nhân:* Khi quy mô người thay đổi từ tiền cảnh đến hậu cảnh, trường tiếp nhận đa tỉ lệ của ASPP là cần thiết để bao quát các đầu người kích thước lớn.

---

## 5. THỰC NGHIỆM ĐÁNH ĐỔI PHẦN CỨNG & PHÒNG VỆ PHẢN BIỆN (REBUTTAL READINESS)

### 5.1. Biểu Đồ Đánh Đổi Thực Nghiệm Hậu Nghiệm (Empirical Trade-off Scatter Plot — Chương 4)
* Trong chương Thực nghiệm và Thảo luận, một biểu đồ phân tán hậu nghiệm (Post-hoc Benchmarking Scatter Plot) sẽ được sử dụng để minh họa mối quan hệ giữa MAE và số lượng tham số / FLOPs:
  - So sánh với các mô hình cùng phân khúc siêu nhẹ ($\le 200\text{k}$ tham số, Standalone, 0% KD): TinyCount (60k params, 0.16G FLOPs, SHA MAE 63.17 / RMSE 120.80), Lw-Count (71k params, 0.8G FLOPs, SHA MAE 68.20 / RMSE 113.50), LSANet (200k params, 14.4G FLOPs, SHA MAE 66.10 / RMSE 110.20).
  - So với các mô hình khổng lồ ($>20\text{M}$ tham số): Thừa nhận khoảng cách về độ chính xác (ví dụ: MAN 56.8 MAE, PET 49.3 MAE) do sự chênh lệch hơn 200 lần về số lượng tham số và năng lực bộ nhớ.

### 5.2. Lập Luận Phản Biện Sẵn Sàng Trước Reviewer
1. **Về việc không đạt MAE 50 như các mô hình SOTA nặng:**
   * *Trả lời:* Mục tiêu nghiên cứu được đặt trong điều kiện biên nghiêm ngặt: ngân sách $\le 104\text{k}$ tham số và hoàn toàn không sử dụng mạng Teacher chưng cất tri thức (0% KD). Trong văn hiến, không có bất kỳ mô hình độc lập nào dưới $1\text{M}$ tham số đạt được ngưỡng MAE 50.
2. **Về hiện tượng No-Neck tốt hơn Flagship trên ShanghaiTech A:**
   * *Trả lời:* Đây là phân tích kỹ thuật khách quan về hiện tượng răng cưa không gian (comb-aliasing) của dilated convolutions trên lưới điểm dày đặc, giải thích rõ sự cần thiết phải chọn lựa cấu trúc cổ mạng dựa trên đặc tính phân bố không gian của từng tập dữ liệu.
3. **Về số lượng 800 epoch trên JHU:**
   * *Trả lời:* Với 2,272 ảnh huấn luyện, 800 epoch tương đương $227,200$ bước cập nhật gradient (gấp hơn 4 lần so với 1,500 epoch trên 300 ảnh của ShanghaiTech). Nhật ký thực nghiệm chứng minh mô hình đã bão hòa và hội tụ ổn định từ epoch 525.
