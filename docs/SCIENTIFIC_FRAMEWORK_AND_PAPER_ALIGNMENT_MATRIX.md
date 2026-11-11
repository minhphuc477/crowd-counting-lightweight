# KHUNG KHOA HỌC & MA TRẬN CĂN CHỈNH BÀI BÁO CHUẨN MỰC
## Constrained Continuous Radon Measure Recovery for Ultra-Lightweight Crowd Counting ($\le 104\text{k}$ Parameters, 0% KD)

> **Tài liệu Định hướng Học thuật Gốc (Canonical Academic Specification)**  
> **Nguyên tắc cốt lõi:** Toàn bộ bài toán là tối ưu hóa đơn mục tiêu có ràng buộc ngân sách tham số ($\le 104\text{k}$ params, 0% KD), không phải tối ưu đa mục tiêu tiến hóa. Phần đánh giá chi phí - độ chính xác chỉ sử dụng biểu đồ đánh đổi thực nghiệm (*Empirical Complexity-Accuracy Trade-off Plot*: MAE vs Params/FLOPs).  
> **Tính trung thực khoa học:** Không dùng từ ngữ thổi phồng (zero hype). Trình bày trực diện các hạn chế đã kiểm chứng trên `runs/benchmarks/`, `results/*.json` và `results/forensics/full_182_forensic_results.json`: bias âm `-19.74` (Pretrained) / `-24.00` (Scratch) người/ảnh trên SHA, rò rỉ sương mù nền ngoài vùng $5\times 5$ quanh điểm GT chiếm **65.57%** tổng khối lượng gộp (**`53,528.18 / 81,629.66`**) và **71.27%** theo trung bình từng ảnh (trung vị **73.45%**), Dense MAE trên SHA Pretrained tăng vọt lên **132.16** (gấp **8.07x** Sparse MAE **16.38**), định vị điểm thô (`F1@4px = 0.2955`, `F1@8px = 0.5820`), nhạy cảm với scale `0.5x` (`116.75 MAE`) và `4` ảnh trống JHU cho trung bình **185.10** người ảo, cùng thành phần `radon_ot` chưa chuẩn hóa độ dài chiếu chỉ chiếm ~0.06% tổng loss.

---

## 1. BẢN CHẤT BÀI TOÁN & CÁC ĐIỂM NGHẼN KỸ THUẬT (PROBLEM STATEMENT & ROOT CAUSES)

### 1.1. Bản Chất Toán Học của Bài Toán Đếm Đám Đông
Về mặt giải tích và thị giác máy tính, bài toán đếm đám đông không phải là bài toán tối ưu đa mục tiêu. Đây là **bài toán nghịch đảo khôi phục độ đo Radon liên tục (Constrained Radon Measure Recovery)** từ quan sát ảnh 2D bị suy thoái:
$$b = \mathcal{A}_\pi y + \eta$$
Trong đó:
- $y \in \mathcal{M}^+(\Omega)$ là độ đo mật độ thực cần khôi phục trên miền ảnh $\Omega \subset \mathbb{R}^2$.
- $\mathcal{A}_\pi$ là toán tử chiếu phối cảnh suy thoái (perspective degradation operator).
- $b$ là quan sát ảnh đầu vào có nhiễu $\eta$.

### 1.2. Ba Điểm Nghẽn Cốt Tử Khi Ràng Buộc Ngân Sách Cực Tiểu ($\le 104\text{k}$ Params, 0% KD)
1. **Rò rỉ khối lượng nền (Background Fog Equilibrium):**
   Kiểm toán forensic trên 182 ảnh SHA (`results/forensics/full_182_forensic_results.json`, tổng `pred_y0 = 81,629.66`) cho thấy `mean_crowd_ratio_5 = 30.32%` và tổng khối lượng nằm ngoài lân cận $5\times 5$ quanh các điểm GT là `bg_mass_5 = 53,528.18` (**65.57%** tổng khối lượng dự đoán).
2. **Bẫy bão hòa gradient của hàm Softplus (Softplus Bias Saturation Trap):**
   Do đạo hàm hàm Softplus suy giảm theo hàm mũ ở miền logit âm ($\sigma(z) \to 0$ khi $z < -6$), gradient suy giảm mạnh dưới tác động weight decay khi cố dập tắt nền rỗng về $0$ tuyệt đối, tạo hiện tượng trì trệ bias và sương mù nền đồng mức.
3. **Hiện tượng Under-counting và Suy giảm ở cảnh siêu đông (Dense Failure):**
   Trong cảnh mật độ cực cao ($d < 4\text{px}$), khoảng cách giữa các đầu người nhỏ hơn bước sải đặc trưng đầu ra ($4\text{px}$), gây hiện tượng đếm thiếu có hệ thống: trên SHA Test (Direct Pretrained, seed 42), `Bias = -19.74`, `Sparse MAE = 16.38` ($n=7$), `Moderate MAE = 47.31` ($n=129$), còn `Dense MAE = 132.16` ($n=46$, gấp **8.07 lần** vùng thưa).

---

## 2. MA TRẬN CĂN CHỈNH KHOA HỌC (THE ALIGNMENT MATRIX)

Bảng căn chỉnh ánh xạ $1-1$ giữa bài toán thực tế, câu hỏi nghiên cứu (RQ), mục tiêu nghiên cứu (RO), phương pháp giải quyết, bằng chứng thực nghiệm và giới hạn kỹ thuật:

| Thành phần | RQ1: Giới hạn Băng thông Không gian - Tần số (Spatial Bandwidth & Sampling) | RQ2: Giám sát Bất định Dị phương sai & Rò rỉ Nền (Heteroscedastic Count & Calibration) | RQ3: Khái quát hóa Đa Miền & Thiết lập Tiên nghiệm (Cross-Domain Generalization) |
| :--- | :--- | :--- | :--- |
| **Problem Statement** | Khi không dùng mạng Teacher chưng cất tri thức (0% KD), backbone $\approx 87.6\text{k}$ params suy giảm phân giải ở tầng sải thô, trong khi dilated convolutions của ASPP có thể tạo răng cưa không gian (comb-aliasing) ở cụm điểm dày đặc ($d < 4\text{px}$). | Sương mù nền tích tụ **65.57%** khối lượng dự đoán ngoài vùng $5\times 5$ quanh điểm GT; đồng thời loss đếm $L_1$ giả định sai phương sai đồng nhất gây sai số nặng khi tổng số người biến thiên từ hàng chục đến hàng nghìn; `Radon-OT` chưa scale theo chiều dài chiếu khiến tỷ trọng thực tế chỉ ~0.06% loss. | Phân bố mật độ và độ phân giải ảnh biến thiên mạnh giữa SHA, SHB, QNRF, JHU-CROWD++ và UCF_CC_50, đòi hỏi thiết lập tiên nghiệm mật độ $m_0$ và cấu hình resize/crop phù hợp từng miền mà không dùng heuristic phụ thuộc nhãn test. |
| **Research Question (RQ)** | Cấu trúc cổ mạng nào (ASPP giãn nở `[2,4,8]` + GAP vs nối tắt đơn tỉ lệ DW $3\times 3$ `d=1` No-Neck) dung hòa tốt hơn giữa trường tiếp nhận ngữ cảnh (ảnh thưa/biến cỡ lớn) và tần số xung điểm cao (ảnh siêu đông)? | Hàm hợp lý Negative-Binomial toàn cục (với phân tán $r$) và loss phân bổ Bayes + Sliced Radon-OT cải thiện sai số đếm như thế nào so với $L_1$/MSE truyền thống, và solver lặp phân vùng (`b1`) có cải thiện bản Direct (`enable_solver=false`) hay không? | Đường ống Direct (`enable_solver=false`, $\le 104\text{k}$ params) có duy trì được thứ hạng và độ ổn định xuyên qua 5 tập benchmark chuẩn từ cả hai chế độ From-Scratch và Pretrained hay không? |
| **Research Objective (RO)** | Đo kiểm định lượng sự đánh đổi giữa **ASPP-Lite (`104,006` params)** và **No-Neck DW $3\times 3$ `d=1` (`102,246` params)** cả ở From-Scratch và Pretrained (kèm cặp đối chứng khắc phục trên SHA/QNRF/JHU), cùng ảnh hưởng bước sải (`stride=2` vs `stride=4`). | Kiểm chứng vai trò độc lập của **Global NB Count Loss (`r=50`, Direct mode)** so với $L_1$ / No-Count, tách riêng tác động của `region_head` (`4,082` params ở idle trong Direct mode; nhánh `b1` SIRT làm xấu MAE), và hiệu chỉnh thang đo `radon_ot_scale_by_length=true`. | Chuẩn hóa đánh giá không ad-hoc split trên 5 tập quốc tế (SHA, SHB, QNRF, JHU Val/Test, UCF_CC_50 5-fold) với cấu hình đồng nhất `configs/benchmarks/` và bổ sung bộ đối chứng khắc phục `configs/benchmarks/remediation/`. |
| **Proposed Method** | **Subpixel Stride-2 Direct Carrier + Neck Analysis:** Backbone `MobileNetV4-Conv-Small-050` kết hợp ASPP-Lite hoặc nối tắt đơn tỉ lệ `DW 3x3 d=1` (`102,246` params), đầu ra PixelShuffle Stride-2 tạo bản đồ mật độ $1/4$ kích thước ảnh. | **Hybrid Bayesian Point Allocation + Global NB Count:** Phân bổ điểm Bayes (với `bg_ratio` hiệu chỉnh được) + Global Negative Binomial NLL trên tổng ảnh; `Sliced Radon-OT` có công tắc `radon_ot_scale_by_length` để cân bằng thang đo CDF. | **Deterministic Manifest & Scale Protocol:** `init_m0: auto` từ manifest train, resize $\le 1024$ (SHB/SHA) hoặc $\le 2048$ (QNRF/JHU), đánh giá đa seed (`42/1337/2024` trên SHA) và 5-fold CV trên UCF_CC_50. |
| **Empirical Evidence** | Trên SHA Scratch: No-Neck (`DW 3x3 d=1`, `102,246` params) đạt **66.24 MAE**, vượt ASPP Scratch (**74.29 MAE**). Trên JHU Val Scratch: No-Neck **59.65** vs ASPP **63.63**; QNRF Test Scratch: **108.42** vs **109.80**. Ngược lại trên SHB Scratch (cảnh thưa): ASPP **10.49** tốt hơn No-Neck **10.94**. | NB vượt $L_1$ trên cả 4 dataset (Scratch): JHU **63.63 vs 72.73** ($\Delta = +9.10$); QNRF **109.80 vs 125.65** ($\Delta = +15.85$); SHA **74.29 vs 89.33** ($\Delta = +15.04$); SHB **10.49 vs 11.89** ($\Delta = +1.40$). Trong Direct mode Flagship, solver tắt (`enable_solver=false`) vì nhánh SIRT (`b1`) bị bão hòa phân tán (`disp_sat_high=0.479`, làm xấu SHA Scratch thành **77.48**, QNRF thành **117.41**). | Direct Flagship (`104,006` params) đạt trên Pretrained / Scratch: SHA **67.57** (3-seed mean **$69.46 \pm 1.66$**) / **74.29**; SHB **10.19** (`bg005`: **9.41** Scratch) / **10.49**; QNRF **105.36** / **109.80**; JHU Val **61.72** / **63.63** (JHU Test: **76.58** / **80.88**); UCF_CC_50 5-fold: **$329.99 \pm 100.63$** / **$298.65 \pm 85.18$** (`ddof=1`). |
| **Limitation & Failure Mode** | Dưới cấu hình mặc định, ASPP giữ lợi thế ở cảnh thưa biến cỡ mạnh (SHB) nhưng kém hơn No-Neck ở mật độ cao (SHA, JHU). Thiếu cặp No-Neck Pretrained trên SHA/QNRF/JHU (đã bổ sung vào `remediation/`). | 1) Direct Flagship giữ `region_head` (`4,082` params) ở trạng thái idle; 2) `radon_ot_scale_by_length=false` làm `train_radon_ot` chỉ ~`0.016-0.038` (~0.06% loss), khiến các mức $\lambda_{\text{radon}} \in \{0.2..2.0\}$ bất phân biệt; 3) `a2_pure_bayesian` trên SHA/QNRF thay đổi $\lambda_{\text{bayes}}$ thành `1.0` thay vì `0.15`. | Trên UCF_CC_50 (50 ảnh cực ít), Pretrained kém hơn Scratch (`329.99` vs `298.65`); một số biến thể Pretrained đang bỏ ngỏ đối chứng (`SHB bg005`, `JHU disp10`, `SHA Gaussian MSE`) và 8 sweep đang chạy dở trên `sweeps_train`. |

---

## 3. BA ĐÓNG GÓP KHOA HỌC CHUẨN MỰC (CORE SCIENTIFIC CONTRIBUTIONS)

1. **Phân tích Thực chứng Kiến trúc Siêu nhẹ $\le 104\text{k}$ Tham số (0% KD):**
   * Định lượng rạch ròi vai trò backbone `MobileNetV4-Conv-Small-050`, khối cổ mạng ASPP-Lite vs nối tắt `DW 3x3 d=1` (`No-Neck`, `102,246` params), và biến thể thuần Direct không mang trọng số thừa (`Pure-Direct`, `99,924` params) trong chế độ không chưng cất tri thức.
2. **Giám sát Số lượng Negative-Binomial Toàn cục & Phân tích Sai số Hỗn hợp Loss:**
   * Chứng minh hàm hợp lý rời rạc Negative-Binomial toàn cục cải thiện đồng loạt từ $+1.40$ đến $+15.85$ MAE trên 4 tập benchmark so với hồi quy $L_1$, đồng thời minh bạch hóa các chế độ thất bại của solver SIRT phân vùng (`b1`) và bẫy tỷ lệ thang đo của `Radon-OT`.
3. **Khảo sát Thực nghiệm Đa Trung Tâm & Phân tích Đánh đổi Chi phí - Sai số:**
   * Đánh giá toàn diện trên 5 tập benchmark quốc tế chuẩn (SHA 3-seed, SHB, UCF-QNRF, JHU-CROWD++ Val/Test, UCF_CC_50 5-fold) với profiling FLOPs chính xác bằng `PyTorch FlopCounterMode @ 512x512` (`0.6547 GFLOPs` = `0.327 GMACs`) thay cho ước lượng cũ.

---

## 4. DỮ LIỆU THỰC NGHIỆM ĐÃ KIỂM TOÁN (TRÊN `runs/benchmarks/`)

*(Trích xuất trực tiếp từ `runs/benchmarks/**/eval_val/summary.json` và `attribute_breakdown.json` trên nhánh `RMR`; hạt giống mặc định `seed: 42` trừ khi ghi rõ)*

### 4.1. Bảng Tổng Hợp 5 Tập Dữ Liệu Chuẩn Quốc Tế

| Tập Dữ Liệu | Giao thức Đánh giá | Resize Config | Biến thể Mô hình | Tham số | FLOPs (@ 512x512) | MAE | RMSE | NAE | Bias |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **JHU-CROWD++** | Official Val (500 imgs) | Max $\le 2048$ | Pretrained (Direct) | 104,006 | 0.655 GFLOPs (`0.327G` MACs) | **61.72** | **227.79** | 0.285 | -13.54 |
| **JHU-CROWD++** | Official Test (`attribute_breakdown.json`) | Max $\le 2048$ | Pretrained (Direct) | 104,006 | 0.655 GFLOPs | **76.58** | **295.07** | 0.415 | — |
| **JHU-CROWD++** | Official Val (500 imgs) | Max $\le 2048$ | Scratch (Flagship) | 104,006 | 0.655 GFLOPs | **63.63** | **254.78** | 0.259 | -30.77 |
| **JHU-CROWD++** | Official Test (`attribute_breakdown.json`) | Max $\le 2048$ | Scratch (Flagship) | 104,006 | 0.655 GFLOPs | **80.88** | **311.95** | 0.522 | — |
| **JHU-CROWD++** | Official Val (500 imgs) | Max $\le 2048$ | No-Neck (Scratch) | **102,246** | **0.654 GFLOPs** | **59.65** | **198.74** | 0.301 | -32.92 |
| **UCF-QNRF** | Official Test (334 imgs)| Max $\le 2048$ | Pretrained (Direct) | 104,006 | 0.655 GFLOPs | **105.36** | **174.94** | 0.161 | -5.94 |
| **UCF-QNRF** | Official Test (334 imgs)| Max $\le 2048$ | Scratch (Flagship) | 104,006 | 0.655 GFLOPs | **109.80** | **189.08** | 0.152 | -38.16 |
| **ShanghaiTech A**| Official Test (182 imgs)| Max $\le 1024$ | Pretrained (Direct, s42) | 104,006 | 0.655 GFLOPs | **67.57** | **110.77** | 0.165 | -19.74 |
| **ShanghaiTech A**| Official Test (3-Seed: 42/1337/2024)| Max $\le 1024$ | Pretrained (Mean $\pm$ Std) | 104,006 | 0.655 GFLOPs | **69.46 $\pm$ 1.66** | **107.83** | 0.176 | -13.92 |
| **ShanghaiTech A**| Official Test (182 imgs)| Max $\le 1024$ | Scratch (Flagship) | 104,006 | 0.655 GFLOPs | **74.29** | **118.23** | 0.182 | -24.00 |
| **ShanghaiTech A**| Official Test (182 imgs)| Max $\le 1024$ | No-Neck (Scratch) | **102,246** | **0.654 GFLOPs** | **66.24** | **102.27** | 0.182 | -15.62 |
| **ShanghaiTech B**| Official Test (316 imgs)| Max $\le 1024$ | Pretrained (Direct) | 104,006 | 0.655 GFLOPs | **10.19** | **15.88** | 0.084 | -2.33 |
| **ShanghaiTech B**| Official Test (316 imgs)| Max $\le 1024$ | Scratch (Flagship) | 104,006 | 0.655 GFLOPs | **10.49** | **16.98** | 0.088 | -3.79 |
| **ShanghaiTech B**| Official Test (316 imgs)| Max $\le 1024$ | BG Ratio 0.05 (Scratch) | 104,006 | 0.655 GFLOPs | **9.41** | **14.12** | 0.080 | -2.52 |
| **UCF_CC_50** | 5-Fold Cross-Val (`ddof=1`) | Max $\le 1024$ | Scratch (Mean $\pm$ Std) | 104,006 | 0.655 GFLOPs | **298.65 $\pm$ 85.18** | **466.66 $\pm$ 233.43** | 0.431 | -10.54 |
| **UCF_CC_50** | 5-Fold Cross-Val (`ddof=1`) | Max $\le 1024$ | Pretrained (Mean $\pm$ Std) | 104,006 | 0.655 GFLOPs | **329.99 $\pm$ 100.63** | **492.00 $\pm$ 222.62** | 0.355 | -49.33 |

*(Ghi chú: Độ lệch chuẩn mẫu `ddof=1` trên 5 fold UCF_CC_50 là $\pm 85.18$ [Scratch] và $\pm 100.63$ [Pretrained]; nếu tính `ddof=0` theo quần thể 5 phần tử là $\pm 76.19$ và $\pm 90.00$.)*

---

### 4.2. Bằng Chứng Đối Chứng Giữa Các Giả Thuyết Khoa Học (From-Scratch Controls)

#### Giả Thuyết 1: Giám sát Số lượng Negative-Binomial vs Naive $L_1$
* **JHU (Val):** Flagship (NB): **63.63** vs. $L_1$ Count (`b1`): **72.73** ($\Delta = +9.10\text{ MAE}$) vs. No Count (`b2`): **69.15**
* **QNRF:** Flagship (NB): **109.80** vs. $L_1$ Count (`c2`): **125.65** ($\Delta = +15.85\text{ MAE}$) vs. No Count (`c1`): **143.83** ($\Delta = +34.03\text{ MAE}$)
* **SHA:** Flagship (NB): **74.29** vs. $L_1$ Count (`b3`): **89.33** ($\Delta = +15.04\text{ MAE}$) vs. No Count (`b2`): **79.78**
* **SHB:** Flagship (NB): **10.49** vs. $L_1$ Count (`c2`): **11.89** ($\Delta = +1.40\text{ MAE}$) vs. No Count (`c1`): **10.90**

#### Giả Thuyết 2: Vai trò của Sliced Radon-OT và Bayesian Likelihood
* **JHU (Val):** Flagship: **63.63** vs. Pure Bayes (`a2`): **62.50** vs. Pure ROT (`a3`): **74.55** vs. Gaussian MSE (`a5`): **70.80**
* **QNRF:** Flagship: **109.80** vs. Pure Bayes (`a2`, $\lambda=1.0$): **121.24** vs. Pure ROT (`a3`): **122.68** vs. Gaussian MSE (`a4`): **119.51**
* **SHA:** Flagship: **74.29** vs. Pure Bayes (`a2`, $\lambda=1.0$): **77.81** vs. Pure ROT (`a3`): **84.54** vs. Gaussian MSE (`a4`): **74.60**
* **SHB:** Flagship: **10.49** vs. Pure Bayes (`a2`): **10.38** vs. Pure ROT (`a3`): **13.83** vs. Gaussian MSE (`a4`): **11.85**
* *Lưu ý kiểm toán kỹ thuật:* (1) Pure Radon-OT đạt Bias toàn cục gần 0 nhất trên mọi tập (JHU: $+2.60$, QNRF: $+2.31$, SHA: $+4.76$, SHB: $+4.12$), nhưng khi `radon_ot_scale_by_length=false` (mặc định cũ), giá trị số học của `radon_ot` chỉ đạt ~`0.016-0.038` (~`0.06%-0.14%` tổng loss); (2) `a2_pure_bayesian` trên SHA & QNRF dùng `lambda_bayesian=1.0` thay vì `0.15` nên cần cặp đối chứng sạch (`configs/benchmarks/remediation/`).

#### Giả Thuyết 3: Nối tắt Đơn tỉ lệ DW $3\times 3$ `d=1` (`No-Neck`) vs ASPP Dilated Neck
* Trên **SHA (Scratch):** `No-Neck` (`102,246` params): **66.24 MAE | 102.27 RMSE** vs. `ASPP Flagship`: **74.29 MAE | 118.23 RMSE**
* Trên **JHU Val (Scratch):** `No-Neck`: **59.65 MAE | 198.74 RMSE** vs. `ASPP Flagship`: **63.63 MAE | 254.78 RMSE**
* Trên **QNRF (Scratch):** `No-Neck`: **108.42 MAE | 184.73 RMSE** vs. `ASPP Flagship`: **109.80 MAE | 189.08 RMSE**
* Trên **SHB (Scratch, mật độ thưa):** `ASPP Flagship`: **10.49 MAE | 16.98 RMSE** vs. `No-Neck`: **10.94 MAE | 18.49 RMSE**

---

## 5. ĐỐI CHIẾU VĂN HIẾN & PHÒNG VỆ PHẢN BIỆN (LITERATURE BASELINES & REBUTTAL)

### 5.1. Biểu Đồ Đánh Đổi Thực Nghiệm (Empirical Complexity-Accuracy Trade-off — SHA)
* So sánh với các mô hình siêu nhẹ độc lập ($\le 200\text{k}$ tham số, không KD, số liệu gốc từ bài báo tương ứng trên ShanghaiTech Part A):
  - **TinyCount (JRTIP'24):** `0.060M` params, `0.16G` FLOPs, SHA MAE **78.2** (bản không chưng cất; MAE 63.17 chỉ đạt khi dùng chưng cất tri thức LSKD).
  - **Lw-Count (TCSVT'22):** `0.071M` params, SHA MAE **69.7** / RMSE **100.5**.
  - **MCNN (CVPR'16):** `0.133M` params, `2.80G` FLOPs, SHA MAE **110.2**.
  - **LSANet (Neurocomp'22):** `0.200M` params, `14.4G` FLOPs (tại độ phân giải đánh giá của tác giả), SHA MAE **66.1** / RMSE **110.2**.
* So sánh với nhóm trung bình & nặng (báo cáo đúng độ chênh tham số & FLOPs):
  - **MobileCount (ECCVW'20):** `3.40M` params, `6.15G` FLOPs, SHA MAE **89.4**.
  - **CSRNet (CVPR'18):** `16.26M` params, `68.7G` FLOPs (`@ 512`), SHA MAE **68.2**; **BL (ICCV'19):** `21.50M` params, MAE **62.8**; **DM-Count (NeurIPS'20):** `21.50M` params, MAE **59.7**; **P2PNet (ICCV'21):** `19.22M` params, `104.7G` FLOPs, MAE **52.7**; **PET (ICCV'23):** `51.65M` params, MAE **49.3**.

### 5.2. Lập Luận Phản Biện Sẵn Sàng Trước Reviewer
1. **Không thổi phồng vượt khung đánh giá:** Thừa nhận `Lw-Count` (`69.7`) và `RMR-104k Pretrained` (3-seed mean `69.46±1.66`, seed42 `67.57`) cạnh tranh trực tiếp với nhau ở ngưỡng ~70 MAE dưới điều kiện 0% KD; lợi thế của RMR nằm ở khung phân tích hàm mất mát có kiểm chứng đa tập dữ liệu (SHA/SHB/QNRF/JHU/UCF_CC_50) và bản Scratch `No-Neck` đạt `66.24` MAE trên SHA / `59.65` trên JHU.
2. **Về sự khác biệt Scratch vs Pretrained ở No-Neck và bộ thí nghiệm khắc phục:** Đã thiết kế 11 thí nghiệm khắc phục trong `configs/benchmarks/remediation/` để kiểm tra `No-Neck Pretrained`, `Pure-Direct 99k` (`enable_regional_head=false`), `Radon-OT` chuẩn hóa thang đo (`radon_ot_scale_by_length=true`), `Pure Bayesian matched 0.15`, `SHB Pretrained bg=0.05`, `JHU Pretrained disp=10`, và `SHA Pretrained Gaussian MSE` sạch/nhiễu vị trí.
3. **Về số lượng 800 epoch trên JHU:** Với 2,272 ảnh huấn luyện, 800 epoch tạo nhiều bước cập nhật hơn đáng kể so với 1,500 epoch trên 300 ảnh của SHA.
