# LƯU Ý THIẾT KẾ THÍ NGHIỆM & QUY CHUẨN HÀNH VI CHỐNG THOÁI HÓA (ANTI-REGRESSION)
*Dự án Radon Measure Recovery (RMR) - ShanghaiTech Part A Benchmark*

---

## I. BỐN THỦ PHẠM KỸ THUẬT GÂY THOÁI HÓA (THE FOUR TECHNICAL CULPRITS)

Tài liệu này ghi lại chi tiết bản chất vật lý, cơ sở toán học, triệu chứng thực nghiệm và quy tắc bắt buộc liên quan đến 4 lỗi kỹ thuật đã được mổ xẻ từ đợt thực nghiệm V65 so với mỏ neo V64 Champion (69.03 Raw / 69.19 TTA MAE).

```
                            ┌─────────────────────────────────────────────────────────┐
                            │      4 THỦ PHẠM KỸ THUẬT GÂY THOÁI HÓA (V65 LESSONS)    │
                            └─────────────────────────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┬────────────────────────┐
         ▼                         ▼                         ▼                        ▼
 ┌───────────────┐         ┌───────────────┐         ┌───────────────┐        ┌───────────────┐
 │   THỦ PHẠM 1  │         │   THỦ PHẠM 2  │         │   THỦ PHẠM 3  │        │   THỦ PHẠM 4  │
 │ use_diag=True │         │ scale_range:  │         │ Bỏ Bayesian   │        │ Ungated FED   │
 │               │         │  [0.4, 1.8]   │         │ trên Track A  │        │ Carrier       │
 └───────┬───────┘         └───────┬───────┘         └───────┬───────┘        └───────┬───────┘
         │                         │                         │                        │
         ▼                         ▼                         ▼                        ▼
  GAP phá nát cục bộ       Đầu 4px co còn 1.6px     Ô 64x64px mất định vị    Gieo người ảo vào
  (+4.96 MAE penalty)      (Aliasing Stride 4)      (Mất vector gradient)    gờ tường, lan can
```

---

### 1. Thủ phạm 1: Kích hoạt nhầm `use_diag: True` (Định tuyến thang đo hình học toàn cục)

* **Bản chất kỹ thuật & Cơ chế thất bại**:
  * DiAG sử dụng `F.adaptive_avg_pool2d(P16, 1)` (Global Average Pooling - GAP) để nén toàn bộ bản đồ đặc trưng cấp cao $P_{16}$ thành 1 vector đặc trưng toàn cục, sau đó dùng MLP để dự đoán độ lệch thang đo $\delta_{\text{scale}}$ điều chế mạng.
  * Trên benchmark ShanghaiTech Part A, phân bố đám đông mang tính **dị hướng không gian cực đoan (extreme spatial anisotropy)**: vùng đường chân trời ở phía xa cực kỳ dày đặc ($N > 1000$, cự ly đầu $3 - 6$px), trong khi cận cảnh ở tiền cảnh lại rất thưa thớt hoặc là nền gạch/cây cối rỗng (cự ly $30 - 80$px).
  * Phép nén GAP toàn cục làm ô nhiễm thông tin thang đo: nó lấy trung bình cộng đặc trưng giữa vùng dày và vùng thưa. Hệ quả là tại các vùng dày, mạng bị gán thang đo quá lớn (gây nhòe mờ đầu người li ti), còn tại vùng thưa, mạng lại gán thang đo quá nhỏ.
* **Hậu quả định lượng thực tế**:
  * Tại đợt ablation V64: mô hình `use_diag: False` đạt **69.03 MAE**, trong khi mô hình bật `use_diag: True` bị phạt nặng **73.99 MAE (+4.96 MAE penalty)**.
  * Tại V65: Việc bật nhầm `use_diag: True` trên toàn bộ config làm suy yếu độ sắc nét của nhánh `ScaleRoutingHead`.
* **Quy tắc thiết kế thí nghiệm bắt buộc (Experimental Invariant)**:
  * **MẶC ĐỊNH BẮT BUỘC: `use_diag: False`**.
  * Chỉ sử dụng định tuyến thang đo cục bộ không gian $3 \times 3$ thuần túy (`ScaleRoutingHead` dựa trên tensor đặc trưng không gian $P_4, P_8, P_{16}$ tại từng vị trí pixel, tuyệt đối không dùng Global Average Pooling toàn cảnh).

---

### 2. Thủ phạm 2: Scale Jittering quá đà `scale_range: [0.4, 1.8]`

* **Bản chất kỹ thuật & Cơ chế thất bại**:
  * Tỷ lệ co giãn $[0.4, 1.8]$ tạo ra dải biến thiên diện tích khổng lồ: $(1.8 / 0.4)^2 = 4.5^2 = \mathbf{20.25\times}$ (so với dải chuẩn $[0.6, 1.4]$ của V64 chỉ là $(1.4/0.6)^2 = \mathbf{5.44\times}$).
  * Khi một ảnh có đầu người nhỏ $4$px bị downscale với hệ số ngẫu nhiên $0.4\times$, đường kính đầu người bị co xuống chỉ còn $\mathbf{1.6}$ **pixels**.
  * Trên Track A (Stride 4): Một ô đặc trưng FPN có kích thước $4 \times 4$ pixels không gian ảnh gốc. Khi đầu người nhỏ $1.6$px, diện tích chiếm chỗ chỉ bằng $1.6^2 / 4^2 = 0.16$ (chưa tới 1/6 diện tích ô đặc trưng). Hiện tượng lấy mẫu dưới tần số Nyquist (**spatial sub-sampling aliasing**) phá hủy hoàn toàn hình thái đầu người thành nhiễu hạt.
  * Mạng nơ-ron siêu nhẹ ($\le 104k$ tham số) hoàn toàn không có dung lượng biểu diễn để học một ánh xạ bất biến trên dải biến dạng từ $1.6$px đến $72$px. Động lực học tối ưu bị chao đảo dữ dội (Val MAE tại epoch 1000 dao động ở mức 79.69 so với 71.65 của V64).
* **Hậu quả định lượng thực tế**:
  * Track A bị phá hủy nặng nề: từ 73.91 MAE tụt thẳng xuống 87.75 MAE (+13.84 MAE thoái hóa).
* **Quy tắc thiết kế thí nghiệm bắt buộc (Experimental Invariant)**:
  * **DẢI SCALE JITTER CHUẨN DUY NHẤT: `scale_range: [0.6, 1.4]`**.
  * Tuyệt đối không sử dụng `scale_range` có cận dưới $< 0.5$ đối với kiến trúc Stride 4 hoặc bất kỳ mô hình nào có tham số $\le 200,000$.

---

### 3. Thủ phạm 3: Loại bỏ Bayesian Point Loss trên Track A (`lambda_bayesian: 0.0`)

* **Bản chất kỹ thuật & Cơ chế thất bại**:
  * Ở V65 Track A, hàm loss phân bổ bị rút về chỉ dùng `flat_dm16` thuần túy (`lambda_bayesian: 0.0`).
  * Hàm mất mát Dirichlet-Multinomial `flat_dm16` chỉ giám sát việc bảo toàn tổng số lượng trên từng khối vĩ mô $16 \times 16$ ô đặc trưng.
  * Trên Track A (Stride 4), mỗi ô đặc trưng cách nhau 4 pixels ảnh gốc. Do đó, một khối vĩ mô $16 \times 16$ tương đương diện tích không gian ảnh gốc lên tới:
    $$(16 \times 4) \times (16 \times 4) = 64 \times 64 = \mathbf{4,096 \text{ pixels}^2}$$
  * Trong một vùng rộng $64 \times 64$ pixels, nếu không có hàm mất mát điểm (Bayesian Point Loss) kéo gradient vào từng tọa độ $(x, y)$ cụ thể của đầu người, đạo hàm của DM16 là đẳng hướng và bất biến vị trí bên trong khối. Mạng nơ-ron không nhận được bất kỳ vector gradient nào để biết đầu người nằm ở đâu trong 4,096 pixels đó.
* **Hậu quả định lượng thực tế**:
  * Track A mất hoàn toàn khả năng định vị điểm Dirac nhọn, sóng mang bị dàn phẳng lì, MAE vọt lên 84.70.
* **Quy tắc thiết kế thí nghiệm bắt buộc (Experimental Invariant)**:
  * **TRACK A (Stride 4) BẮT BUỘC DÙNG DUAL SUPERVISION**:
    `allocation_loss_type: dual_bayesian_dm16`, `lambda_bayesian: 0.15`, `lambda_dm16: 0.85`.
  * Tuyệt đối không bao giờ bỏ point-level supervision khi độ phân giải đặc trưng thô ($\text{stride} \ge 4$).

---

### 4. Thủ phạm 4: Ungated FED-Carrier trên cảnh mật độ trung bình ($100 < N \le 500$)

* **Bản chất kỹ thuật & Cơ chế thất bại**:
  * Toán tử FED-Carrier (`feat_energy_weight > 0.0`) sử dụng năng lượng không gian cao tần $E_{\text{feat}}(u) = \|P_4(u) - \text{AvgPool}(P_4(u))\|_2$ làm sóng mang cảm ứng thay thế khi $y_0(u) = 0$.
  * Cơ chế này hoạt động xuất sắc trên cụm dày ($N > 500$, nơi đầu người san sát $3 - 6$px, $E_{\text{feat}}$ bắt chính xác các đỉnh vân đầu người, giúp hạ Dense MAE từ 145.80 xuống 124.13).
  * Tuy nhiên, ở cảnh trung bình ($100 < N \le 500$, 129 ảnh trong tập test), mật độ hộp vĩ mô $b_R$ vẫn vượt qua ngưỡng $\tau_{\text{reg}} = 0.00025$, khiến Presence Gate $M_{\text{reg}} \approx 1.0$.
  * Khi có sai lệch nhỏ giữa $b_R$ và tích phân sóng mang thô ($b_R > \mu_0$), FED-Carrier bơm mass vào bất kỳ vị trí nào có $E_{\text{feat}}$ cao. Trong cảnh đô thị trung bình, các cạnh chấn song lan can, gờ gạch lát nền, tán cây, biển báo có $E_{\text{feat}}$ rất mạnh, dẫn đến việc gieo rắc $+1$ đến $+4$ người ảo mỗi hộp.
* **Hậu quả định lượng thực tế**:
  * Trên nhóm trung bình ($100 < N \le 500$, 129 ảnh): V65 bị thoái hóa **57.34 so với 51.46 ở V64 (+5.88 MAE)**.
* **Quy tắc thiết kế thí nghiệm bắt buộc (Experimental Invariant)**:
  * **ÁP DỤNG DENSITY-GATED FED-CARRIER**:
    FED-Carrier chỉ được kích hoạt khi số đếm hộp vĩ mô thỏa mãn điều kiện cụm dày:
    $$b_s(R) > \tau_{\text{dense\_fed}} = 2.0$$
    Khi $b_s(R) \le 2.0$ (cảnh thưa và trung bình), `feat_energy_weight` phải tự động gán bằng $0.0$, bảo vệ tuyệt đối các cạnh kiến trúc không bị gieo mass ảo.

---

## II. QUY TRÌNH HÀNH VI KHI THIẾT KẾ THÍ NGHIỆM (EXPERIMENTAL CONDUCT & HYGIENE)

Để ngăn chặn triệt để tình trạng lặp lại vết xe đổ stall/thoái hóa, mọi thành viên nghiên cứu và AI Agent phải tuân thủ 5 nguyên tắc vàng sau đây:

### 1. Nguyên tắc Can Thiệp Tối Thiểu (Principle of Minimal Intervention)
* **Quy tắc**: Không bao giờ thay đổi đồng thời cả cấu trúc kiến trúc (Architecture), dải biến đổi dữ liệu (Augmentation), và hàm mục tiêu (Loss formulation) trong cùng một đợt thí nghiệm.
* **Hành vi chuẩn**: Khi giới thiệu một module toán học mới (ví dụ: solver operator mới), phải giữ nguyên 100% các siêu tham số huấn luyện của mỏ neo tốt nhất (V64 Gold Baseline).

### 2. Nguyên tắc Đối Chứng Đơn Biến (Strict Single-Variable Control)
* **Quy tắc**: Mọi đợt thử nghiệm bắt buộc phải có một mô hình Đối chứng (Control Baseline) chạy song song với cùng một môi trường và recipe huấn luyện, chỉ khác duy nhất 1 biến số cần kiểm tra.
* **Hành vi chuẩn**: Đánh giá hiệu quả của module dựa trên phép trừ trực tiếp $(M_{\text{new}} - M_{\text{baseline}})$, không so sánh chéo giữa hai run có recipe huấn luyện khác nhau.

### 3. Nguyên tắc Tiền Kiểm Tra Cấu Hình (Preflight Config Audit)
Trước khi gửi lệnh huấn luyện lên server GPU, bắt buộc phải chạy script kiểm tra:
1. Số lượng tham số có thể huấn luyện:
   * Track A: $\le 104,441$ parameters.
   * Track B: $\le 200,000$ parameters.
2. Kiểm tra cờ cấm:
   * `use_diag: False` (Cấm bật `True`).
   * `wsd_stable_ratio: BANNED` (Bắt buộc dùng High-Floor Cosine Annealing, $1500$ epochs, `min_lr_ratio: 0.20`).
   * `use_top_down_semantic_gate: BANNED`.
   * `foreground_gate: BANNED`.
3. Kiểm tra dải dữ liệu:
   * `scale_range: [0.6, 1.4]`.
   * `crop_size: [512, 512]`.

### 4. Nguyên tắc Đánh Giá Đa Tầng (Multi-Tier Evaluation Protocol)
* Tuyệt đối không chỉ nhìn vào MAE toàn cục đơn lẻ để kết luận thành bại.
* Bắt buộc phân tích trên 4 lát cắt mật độ chuẩn:
  1. **Sparse ($N \le 100$)**: Đánh giá khả năng chống nhiễu nền và bảo toàn độ đo 0.
  2. **Moderate ($100 < N \le 500$)**: Đánh giá độ chính xác của thang đo tầm trung (chiếm đa số tập test, 129/182 ảnh).
  3. **Dense ($N > 500$)**: Đánh giá khả năng giải phóng bẫy zero-support.
  4. **Outliers ($N > 1000$)**: Đánh giá độ hồi phục số đếm tại các ảnh cực đoan (`IMG_90`, `IMG_92`, `IMG_8`).
* Một cải tiến chỉ được xem là thành công khi nó cải thiện lát cắt mục tiêu mà **không gây thoái hóa trên các lát cắt còn lại**.

### 5. Nguyên tắc Cam Kết và Đồng Bộ Ngay Lập Tức (Continuous Commit & Push)
* Mọi thay đổi mã nguồn, file cấu hình YAML, kết quả đánh giá thực nghiệm phải được commit và push lên git remote ngay lập tức:
  ```powershell
  git add <files>
  git commit -m "docs/feat/fix: <clear conventional message>"
  git push origin RMR
  ```
* Không để tích tụ thay đổi chưa commit qua các lượt tương tác.
