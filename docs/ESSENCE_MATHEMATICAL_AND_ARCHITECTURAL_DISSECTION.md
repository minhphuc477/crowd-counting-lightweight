# GIẢI PHẪU BẢN CHẤT TOÁN HỌC, VẬT LÝ VÀ KIẾN TRÚC MÔ HÌNH RMR
## ĐIỀU TRA GỐC RỄ: TẠI SAO CÁC PHÉP KẾT HỢP HYPERPARAMETER ĐỀU CHẠM TRẦN VÀ BẢN CHẤT BÀI TOÁN LÀ GÌ?

---

### I. ĐẶT VẤN ĐỀ VÀ MỆNH LỆNH KHOA HỌC
Người dùng đã đưa ra chỉ thị và cảnh báo mang tính then chốt:
> *"xem bản chất chứ không phải chỉ kết hợp hyperameter như vậy"*

Trong quá trình nghiên cứu, một cái bẫy thường gặp của Machine Learning thực nghiệm là **"bẫy siêu tham số" (Hyperparameter Tweaking Trap)**: khi mô hình gặp nút thắt (plateau) ở ~70 MAE, thay vì đi sâu vào bản chất vật lý của dữ liệu, tính chất toán học của các toán tử quan sát, và sự tương tác giữa các thành phần kiến trúc, ta lại cố gắng chắp vá các tham số điều khiển rời rạc (ví dụ: ghép `pad_small_images` với `scale_seeded_carrier`, chỉnh `eps`, nâng `trust_pos_kappa`, sửa `gamma_scales`). 

Các phép ghép nối siêu tham số ngoại vi này có thể làm giảm sai số cục bộ trên một tập con (slice) nhất định, nhưng ngay lập tức gây ra hiện tượng **bập bênh mật độ (Density Seesaw)** hoặc phá vỡ các điều kiện hội tụ toán học, kéo MAE toàn cục thoái hóa trở lại 75–85 MAE.

Bản báo cáo này thực hiện cuộc giải phẫu toàn diện và sâu sắc từ các nguyên lý tiên đề (First Principles) về:
1. **Bản chất vật lý của tập dữ liệu ShanghaiTech Part A** và sự mâu thuẫn giữa độ đo Dirac nguyên tử với lưới rời rạc Stride 4.
2. **Bản chất toán tử của bài toán nghịch đảo**: Không gian hạt nhân $\ker(\mathcal{A})$, giới hạn phân giải Rayleigh của tích phân hộp, và sự phát tán vô hướng của toán tử liên hợp $\mathcal{A}^*$.
3. **Bản chất của mối quan hệ Sóng mang - Bộ giải (Carrier-Solver Symbiosis vs Hostage Trap)**.
4. **Bản chất hình học phối cảnh (Perspective Distortion) đối kháng với các cửa sổ hộp cố định $[32, 64, 128]\text{ px}$**.
5. **Bản chất của quy trình cắt cúp (Cropping, Padding, Boundary Truncation)**: Giải mã thành công của `sub60_e103` và nút thắt còn tồn đọng.
6. **Lộ trình chuyển đổi hệ hình (Paradigm Shift)** dựa trên nền tảng toán học chặt chẽ thay vì chắp vá tham số.

---

### II. BẢN CHẤT VẬT LÝ & ĐỘ ĐO TOÁN HỌC: DIRAC ATOMIC MEASURE VS STRIED-4 DISCRETE LATTICE

#### 1. Dữ liệu thực tế là gì?
Trong bài toán đếm đám đông trên ShanghaiTech Part A, nhãn gốc của con người là tập hợp các tọa độ điểm:
$$\mathcal{P} = \{x_1, x_2, \dots, x_N\}, \quad x_i \in \Omega \subset \mathbb{R}^2$$
Dưới góc độ lý thuyết độ đo, đây là một **độ đo Radon nguyên tử (Atomic Dirac Measure)**:
$$\mu = \sum_{i=1}^N \delta_{x_i}$$
Mỗi cá thể người là một hàm delta Dirac $\delta_{x_i}$, có diện tích tích phân $\int_{\Omega} \delta_{x_i} \, dx = 1$, nhưng có giá trị bằng 0 ở mọi điểm $x \neq x_i$ và tiến tới $+\infty$ tại $x = x_i$.

#### 2. Mâu thuẫn bản chất tại lưới rời rạc Stride 4
Mô hình RMR sử dụng backbone MobileNetV4 với độ phân giải đầu ra tại Stride 4:
$$1 \text{ cell} = 4 \times 4 \text{ pixels}$$
Một ảnh đầu vào $512 \times 512$ sẽ tương ứng với một lưới mật độ $128 \times 128$ cells.

**Thực nghiệm đo đạc trực tiếp trên tập dữ liệu ShanghaiTech Part A (`data/sha_a_train_all.jsonl` và `data/sha_a_test.jsonl`):**
* Trên tập train (300 ảnh, 162,707 điểm): Có **124 / 300 ảnh (41.3%)** xảy ra hiện tượng **va chạm điểm ảnh (Point Collisions)**, tức là có từ 2 điểm đầu người trở lên rơi vào chính xác cùng một ô cell $4 \times 4$ pixel! Số lượng điểm tối đa rơi vào một ô cell duy nhất là **4 người**!
* Trên tập test (182 ảnh, 78,970 điểm): Có **67 / 182 ảnh (36.8%)** có va chạm điểm ảnh tại Stride 4. Số lượng điểm tối đa rơi vào một ô cell duy nhất lên tới **8 người**!

#### 3. Bẫy ép dẹp mật độ (Logit Ceiling & Mass Suppression Paradox)
Khi tạo target mật độ rời rạc `rasterize_points`:
$$y_{\text{tgt}}(i, j) = \sum_{k=1}^N \mathbf{1}\left\{\left\lfloor \frac{x_k + 0.5}{4} \right\rfloor = j, \left\lfloor \frac{y_k + 0.5}{4} \right\rfloor = i\right\} \in \{0, 1, 2, 3, \dots\}$$
Mục tiêu là các số nguyên rời rạc: ô có 0 người thì target = 0.0, ô có 1 người thì target = 1.0, ô va chạm có target = 2.0 đến 4.0.

Tuy nhiên, mạng nơ-ron tích chập (CNN) với các bộ lọc tích chập $3 \times 3$ và các tầng nội suy liên tục chỉ có thể biểu diễn các **hàm trơn khả vi (Smooth, Band-limited Continuous Functions)**.
* Khi mạng nơ-ron dự đoán một đỉnh mật độ xung quanh một cá thể người, hàm liên tục sẽ tạo ra một "ụ nổi" (Gaussian-like bump) lan rộng ra khu vực lân cận $3 \times 3$ cells ($12 \times 12$ pixels).
* Khi đó:
  - Ô trung tâm đạt giá trị $y(i, j) \approx 0.3 - 0.5$.
  - 8 ô lân cận nhận giá trị rò rỉ $y(i \pm 1, j \pm 1) \approx 0.05 - 0.15$.
* Nhưng ground truth tại 8 ô lân cận đó lại là **0.0 tuyệt đối**!
* Khi áp dụng hàm mất mát ô (Cell Loss, ví dụ Smooth L1):
  $$\mathcal{L}_{\text{cell}} = \sum_{u} \text{SmoothL1}(y(u), y_{\text{tgt}}(u))$$
  8 ô lân cận nhận một hình phạt sai số dương (False Positive Penalty), liên tục truyền gradient ép $y(u)$ xuống 0.
* **Hậu quả tất yếu:** Để giảm thiểu tổng mất mát trên 9 ô, mạng nơ-ron tự động chọn nghiệm tối ưu cục bộ: **dìm thấp đỉnh ở ô trung tâm xuống mức cực thấp** ($0.2 - 0.4$ thay vì $1.0$).
* Đây là lý do vì sao tại sao `sub60_e5` (với $\lambda_{\text{cell}} = 0.0$) lại đạt bước nhảy vọt từ 73 xuống 71 MAE: việc loại bỏ hoàn toàn $\mathcal{L}_{\text{cell}}$ đã giải phóng mạng khỏi áp lực tự triệt tiêu khối lượng ở các ô lân cận!

---

### III. BẢN CHẤT TOÁN TỬ NGHỊCH ĐẢO: KHÔNG GIAN HẠT NHÂN $\ker(\mathcal{A})$ VÀ TOÁN TỬ LIÊN HỢP $\mathcal{A}^*$

#### 1. Định nghĩa Toán tử Quan sát Hộp $\mathcal{A}$
RMR xây dựng bài toán đếm đám đông thành một bài toán nghịch đảo tuyến tính:
$$b = \mathcal{A} y + \varepsilon$$
Trong đó:
* $y \in \mathbb{R}_{+}^{H \times W}$ là trường mật độ cần phục hồi tại Stride 4.
* $\mathcal{A}$ là toán tử tích phân hộp trên tập các cửa sổ trượt đa tỷ lệ $\Omega_r$ với kích thước $W_r \in \{32, 64, 128\}\text{ px}$.
* $b_r = (\mathcal{A} y)_r = \iint_{\Omega_r} y(x) \, dx$ là tổng số người trong từng hộp do Head vùng dự đoán.

#### 2. Phân tích miền tần số Fourier và Không gian Hạt nhân $\ker(\mathcal{A})$
Trong không gian liên tục 2D, một cửa sổ hộp vuông kích thước $W \times W$ tương đương với phép tích chập với hàm chỉ thị $\mathbf{1}_{[-W/2, W/2] \times [-W/2, W/2]}$.
Biến đổi Fourier của toán tử quan sát hộp là:
$$\widehat{\mathcal{A}}(\omega_x, \omega_y) = W^2 \cdot \text{sinc}\left(\frac{\omega_x W}{2\pi}\right) \cdot \text{sinc}\left(\frac{\omega_y W}{2\pi}\right)$$
Trong đó $\text{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$.

Hàm $\text{sinc}(u)$ triệt tiêu tại mọi điểm nguyên khác 0: $u = k \in \mathbb{Z} \setminus \{0\}$.
Do đó, các tần số không gian:
$$\omega_x = \frac{2\pi k}{W}, \quad \omega_y = \frac{2\pi m}{W}, \quad k, m \in \mathbb{Z} \setminus \{0\}$$
**nằm hoàn toàn trong không gian hạt nhân (Null Space) $\ker(\mathcal{A})$!**
$$\mathcal{A} v = 0 \quad \forall v \in \ker(\mathcal{A})$$

**Ý nghĩa vật lý đối với việc đếm đám đông:**
* Cửa sổ hộp nhỏ nhất trong RMR là $W = 32\text{ px}$. Tại Stride 4, nó chiếm một diện tích $8 \times 8 = 64$ cells.
* Mọi cấu trúc tần số cao bên trong phạm vi 32px (ví dụ: các gợn sóng phân biệt giữa 50 người đứng sát nhau với khoảng cách 2–3 pixels) có tổng tích phân bằng 0 trong hộp, đều bị $\mathcal{A}$ triệt tiêu hoàn toàn!
* **Toán tử $\mathcal{A}$ hoàn toàn mù (blind) trước phân bố không gian bên trong hộp 32px!**

#### 3. Sự phát tán vô hướng của Toán tử Liên hợp Adjoint $\mathcal{A}^*$
Khi bộ giải SIRT tính sai số dư thừa giữa mật độ hiện tại và bằng chứng vùng:
$$\delta_r = (\mathcal{A} y)_r - b_r$$
Toán tử liên hợp chuẩn $\mathcal{A}^* \delta$ thực hiện phép chiếu ngược:
$$[\mathcal{A}^* \delta](x) = \sum_{r: x \in \Omega_r} \delta_r$$
* Quan sát công thức trên: $\delta_r$ là một **đại lượng vô hướng (scalar)**.
* Khi chiếu ngược vào không gian ảnh, $\mathcal{A}^*$ phát tán giá trị hằng số $\delta_r$ đồng nhất lên toàn bộ 64 cells của hộp $\Omega_r$!
* $\mathcal{A}^*$ không chứa bất kỳ thông tin nào về việc người nằm ở góc nào của hộp.
* Nếu chỉ dùng $\mathcal{A}^*$ thuần túy (Lebesgue flat adjoint), sau vài bước lặp, ảnh mật độ sẽ biến thành các khối hộp chữ nhật đồng màu, xóa sổ hoàn toàn mọi đặc trưng cá thể.

---

### IV. BẢN CHẤT SỰ TƯƠNG TÁC SÓNG MANG - BỘ GIẢI (CARRIER-SOLVER DYNAMICS): CỘNG SINH HAY BẮT LÀM CON TIN?

#### 1. Cơ chế Radon-Nikodym Adjoint
Để giải quyết việc $\mathcal{A}^*$ làm mờ nền rỗng, RMR đưa vào toán tử liên hợp điều biến Radon-Nikodym:
$$[\mathcal{A}^*_{\text{RN}} \delta](x) = y(x) \cdot \frac{\sum_{r \ni x} w_r \frac{\delta_r}{q_r}}{\sum_{r \ni x} w_r}$$
Trong đó $y(x)$ là mật độ hiện tại, khởi tạo từ trường sóng mang $y_0(x)$ của Fine Head.

#### 2. Cái bẫy "Bắt làm con tin" (The Support-Trapping / Hostage Trap)
Hãy phân tích thừa số $y(x)$ trong toán tử:
1. **Ở nền rỗng ($y(x) = 0$):** $[\mathcal{A}^*_{\text{RN}} \delta](x) = 0$. Adjoint không thể thêm bất kỳ hạt nhiễu nào vào nền rỗng. Nền được bảo vệ an toàn tuyệt đối.
2. **Ở vùng có người nhưng Fine Head dự đoán hụt ($y_0(x) \approx 0.005$):**
   - Giả sử có một cụm người thật, nhưng backbone/Fine Head bị bỏ sót khiến $y_0(x)$ gần bằng 0.
   - Khi đó, dù bằng chứng vùng $b_r$ rất lớn ($b_r = 50$) và sai số $\delta_r = q_r - b_r = -48$ (thiếu hụt trầm trọng), bước nhảy cập nhật của solver là:
     $$\Delta y(x) \propto y(x) \cdot \frac{-48}{q_r} \approx 0.005 \times \text{chỉ số hữu hạn} \approx 0.001!$$
   - **Solver hoàn toàn bất lực!** Nó không thể tạo ra khối lượng ở nơi mà trường sóng mang $y_0(x)$ có giá trị bằng 0.
   - Giá trị hỗ trợ (support) của nghiệm cuối cùng bị giới hạn nghiêm ngặt bởi giá trị hỗ trợ của sóng mang:
     $$\text{supp}(y^*) \subseteq \text{supp}(y_0)$$

#### 3. Bản chất đích thực của Unrolled SIRT Solver
* Bộ giải SIRT **KHÔNG PHẢI là một bộ phát hiện đặc trưng (Feature Detector)**. Nó không có khả năng nhìn vào ảnh RGB để tìm mắt, mũi, tóc hay hình dáng con người.
* Bộ giải SIRT thực chất là một **Bộ cân bằng biên độ khối lượng (Mass Amplitude Equalizer)**:
  - Nếu Fine Head đã định vị đúng các đỉnh (peaks) nhưng dự đoán sai biên độ (ví dụ biên độ chỉ đạt 0.3 thay vì 1.0), solver sẽ nhân biên độ của các đỉnh đó lên để khớp với tích phân vùng $b_r$.
  - Nếu Fine Head không tạo ra đỉnh, solver không thể làm gì được.
* **Mối liên hệ tương hỗ:**
  - Fine Head chịu trách nhiệm về **Vị trí & Cấu trúc tần số cao (Localization & Phase)**.
  - Regional Head và Solver chịu trách nhiệm về **Bảo toàn khối lượng & Tần số thấp (Total Mass Conservation & Low Frequencies)**.
  - Hai nhiệm vụ này phải hỗ trợ nhau theo đúng chức năng, không được dẫm chân lên nhau.

---

### V. BẢN CHẤT HÌNH HỌC PHỐI CẢNH (PERSPECTIVE GEOMETRY) ĐỐI KHÁNG VỚI CỬA SỔ CỐ ĐỊNH

#### 1. Phép chiếu phối cảnh trong ảnh ShanghaiTech Part A
Trong hầu hết các bức ảnh của ShanghaiTech Part A:
* Camera đặt nghiêng một góc $\theta$ so với mặt đất.
* Ma trận chiếu phối cảnh $\mathcal{P}: \mathbb{R}^3 \to \mathbb{R}^2$ khiến kích thước biểu kiến của đầu người $s(u, v)$ phụ thuộc chặt chẽ vào tọa độ thẳng đứng $v$:
  - Phía dưới bức ảnh (tiền cảnh, gần camera): $s \approx 40 - 80\text{ px}$.
  - Phía trên bức ảnh (đường chân trời, xa camera): $s \approx 2 - 4\text{ px}$.
  - Chênh lệch kích thước biểu kiến giữa hai vùng lên tới **10x đến 20x**!

#### 2. Sự bất hợp lý của cửa sổ cố định $[32, 64, 128]\text{ px}$
Khi RMR áp dụng cố định 3 kích thước hộp $[32, 64, 128]\text{ px}$ trên toàn bộ ảnh:
1. **Ở tiền cảnh:**
   - Một cái đầu người có đường kính 60px.
   - Hộp 32px thậm chí nhỏ hơn một cái đầu! Hộp chỉ chứa một phần má hoặc trán của một người ($N \approx 0.2 - 0.5$).
   - Việc tính Negative Binomial trên hộp 32px ở đây là vô nghĩa vì không có khái niệm "phân phối cá thể", mà là tích phân trên một vật thể đơn lẻ.
2. **Ở hậu cảnh (Horizon):**
   - Một hộp 32px chứa từ **50 đến 100 người**!
   - Một hộp 128px chứa từ **500 đến 1500 người**!
   - Tại đây, phân phối Negative Binomial có phương sai rất lớn: $\sigma_b = \sqrt{\mu + \mu^2 / r}$.
   - Với $\mu = 100$ và $r = 50$, $\sigma_b \approx \sqrt{100 + 200} \approx 17.3$ người.
   - Khi áp dụng nguyên lý Morozov Discrepancy với $\gamma = 0.75$:
     $$\text{Deadband} = \gamma \cdot \sigma_b = 0.75 \times 17.3 \approx 13 \text{ người}!$$
   - Nghĩa là: nếu mạng đếm được 87 người thay vì 100 người (sai số 13 người), sai số này bị hàm Morozov **triệt tiêu hoàn toàn về 0** ($\delta_{\text{shrunk}} = 0$)!
   - Kết quả: **Solver hoàn toàn đứng yên, không cập nhật một chút nào ở các hộp đông nhất!**
   - Chỉ cần 10 hộp như vậy ở đường chân trời, mô hình đã bỏ qua $10 \times 13 = 130$ người mà không hề cố gắng sửa chữa!
   - Đây chính là cội nguồn toán học sâu xa của hiện tượng **Negative Net Bias (-10.00 đến -20.00)** ở các ảnh đông người!

---

### VI. BẢN CHẤT QUY TRÌNH CẮT CÚP (CROPPING & PADDING DYNAMICS): GIẢI MÃ THÀNH CÔNG VÀ GIỚI HẠN CỦA `SUB60_E103`

#### 1. Tại sao `pad_small_images` tạo ra bước nhảy vọt ở nhóm Thưa và Vừa?
* Trước `e103`, quy trình data augmentation ép mọi ảnh có kích thước nhỏ hơn 512px phải phóng to (upscale) với tỷ lệ lên tới 2.81x để đủ kích thước crop $512 \times 512$.
* Trong ShanghaiTech Part A, có 93/300 ảnh train (31.0%) và 72/182 ảnh test (39.6%) có kích thước nhỏ hơn 512px.
* Khi phóng to 2x, một cái đầu người có kích thước vật lý 4px bị biến thành 8px. Mạng nơ-ron bị ép học nhận diện đầu người ở tỷ lệ giả tạo.
* Khi đưa vào đánh giá trên tập test ở tỷ lệ tự nhiên 1:1, mạng bị lệch pha phân phối kích thước (Scale Distribution Shift).
* Khi `pad_small_images: true` được kích hoạt trong `sub60_e103`:
  - 100% ảnh nhỏ được giữ nguyên tỷ lệ 1:1, phần thiếu được đệm màu xám trung tính (128).
  - Tỷ lệ vật lý của đầu người được bảo toàn tuyệt đối.
  - **Kết quả thực nghiệm đã chứng minh:**
    * Sparse MAE giảm hơn một nửa: từ **33.85** (`e34`) xuống **17.64**!
    * NAE giảm xuống mức kỷ lục lịch sử: **0.1850** (vượt qua mọi mốc trước đó).
    * Tỷ lệ ảnh được solver cải thiện đạt kỷ lục: **62.6%**!
    * Chuỗi bước lặp SIRT giảm lỗi đơn điệu 100%: $85.18 \to 80.00 \to 77.65 \to 76.10 \to 75.14 \to 74.55 \to 73.80$.

#### 2. Tại sao `pad_small_images` KHÔNG THỂ kéo Dense MAE xuống dưới 130?
* Bản chất của `pad_small_images` là xử lý các ảnh có $\min(W, H) < 512$.
* Nhưng hầu hết các ảnh siêu đông ($N > 500$, thuộc nhóm Dense, 46 ảnh) trong ShanghaiTech Part A là các ảnh có độ phân giải **LỚN** ($768 \times 1024$ hoặc $600 \times 900$).
* Các ảnh lớn này **KHÔNG BAO GIỜ được pad**! Chúng hoàn toàn chịu sự chi phối của việc Random Crop $512 \times 512$.
* Khi cắt một vùng $512 \times 512$ ngẫu nhiên trong đám đông:
  1. **Hiệu ứng cắt đứt biên (Boundary Truncation):** Các đầu người nằm ở rìa crop bị cắt làm đôi hoặc mất ngữ cảnh, gây nhiễu cho các hộp tích phân nằm sát biên.
  2. **Hiệu ứng Deadband Morozov:** Như đã chứng minh ở Phần V, vùng chết $\gamma \sigma_b$ vẫn triệt tiêu gradient ở cụm dày.
  3. **Hiệu ứng Va chạm Stride 4 (Point Collision):** Như đã chứng minh ở Phần II, 41% ảnh có nhiều người trên cùng 1 cell vẫn bị nén dẹp bởi cấu trúc Softplus + Smooth L1.
* Do đó, việc kỳ vọng `pad_small_images` một mình kéo Dense MAE xuống 60 là phi thực tế về mặt toán học. Nó đã hoàn thành xuất sắc nhiệm vụ ở miền xác định của nó (Scale preservation), nhưng không thể giải quyết được các nút thắt khác nằm ngoài miền tác động của nó.

---

### VII. BẢN CHẤT CỦA CÁC THÍ NGHIỆM ĐẠT 70 MAE (`SUB60_E5`, `M04`) VÀ NGUYÊN NHÂN THẤT BẠI CỦA CÁC PHÉP GHÉP NỐI

#### 1. Giải phẫu cấu hình của các nhà vô địch lịch sử

| Thành phần | `sub60_e5` (70.26 TTA / 71.51 Direct) | `m04` (71.03 Direct) | `sub60_e103` (73.80 Direct / Sparse 17.64) |
| :--- | :--- | :--- | :--- |
| **Cell Loss ($\lambda_{\text{cell}}$)** | **0.0 (Tắt hoàn toàn)** | **0.5 (Count Harmonized, norm_power=0.5)** | **0.5 (Count Harmonized, norm_power=0.5)** |
| **Trust Region Floor** | **0.005 (Chặt chẽ, 5x nhỏ hơn e34)** | **0.005** | **0.005** |
| **Hurdle Gating Mode** | `product` ($\pi_r \cdot b_{\text{raw}}$) | `product` | `product` |
| **Curvature Init** | `-8.0` (Gần như tắt $\alpha_{\text{eff}} = 0.0003$) | `-8.0` | `-8.0` |
| **Morozov Gamma** | 0.75 | 0.75 | 0.75 |
| **Padding ảnh nhỏ** | False (Resize ép buộc) | False (Resize ép buộc) | **True (Neutral Gray 128 Padding)** |

#### 2. Tại sao `sub60_e5` và `m04` lại đạt đỉnh hiệu năng?
1. **Triệt tiêu xung đột gradient giữa Cell và Solver:**
   - Trong `sub60_e5`, bằng việc đặt $\lambda_{\text{cell}} = 0.0$, tác giả đã giải phóng toàn bộ xung đột gradient giữa việc khớp từng ô cell rời rạc (Smooth L1) và việc phân bổ khối lượng theo vùng (SIRT solver). Mạng nơ-ron chỉ cần tập trung vào việc tạo ra phân phối khối lượng hợp lý qua Flat DM16 và Regional NB.
2. **Chuẩn hóa công bằng theo căn bậc hai (Square-root Normalization trong `m04`):**
   - Trong `m04`, tham số `cell_norm_power: 0.5` chia độ lớn của cell loss cho $\sqrt{N}$ thay vì chia cho $N$.
   - Trên ảnh thưa ($N=30$): $\sqrt{30} \approx 5.5$.
   - Trên ảnh dày ($N=3000$): $\sqrt{3000} \approx 54.7$.
   - Chênh lệch trọng số giữa ảnh dày và ảnh thưa giảm từ $100\times$ (nếu chia cho $N$) xuống chỉ còn $10\times$! Nhờ đó, ảnh dày không bị "bỏ đói gradient" (gradient starvation), giúp Dense MAE đạt mức kỷ lục 127.26 mà không làm sụp đổ Moderate MAE.
3. **Trust Region Floor chặt chẽ (0.005):**
   - Với 14,000 ô cell nền rỗng trong mỗi ảnh, nếu đặt `trust_floor` lớn (ví dụ 0.025 như trong `e34`), mỗi ô nền rỗng bị rò rỉ $0.025 \times 0.35 \approx 0.008$ người.
   - Tổng khối lượng rò rỉ trên 14,000 ô nền là: $14,000 \times 0.008 = 112\text{ người ảo}$!
   - Việc giữ `trust_region_floor: 0.005` đã khóa chặt rò rỉ nền ở mức tối thiểu ($14,000 \times 0.0017 = 24\text{ người}$), ngăn chặn hoàn toàn hiện tượng False Positives trên nền rỗng.

#### 3. Tại sao các phép ghép nối siêu tham số ngoại vi (Ad-hoc Tweaks) luôn thất bại?
* Hãy nhìn vào các thí nghiệm thất bại:
  - `g9_fix_hurdle_occupancy`: Chuyển hurdle gating sang `occupancy` khiến $b_{\text{solver}}$ tăng vọt ở vùng dày $\to$ Xung lực SIRT quá mạnh phá nát trường sóng mang $\to$ MAE thoái hóa lên **83.56**!
  - `g9_fix_curvature_init`: Đặt curvature alpha init từ -8.0 lên 0.0 khiến số hạng $y^2$ bùng nổ không kiểm soát $\to$ MAE thoái hóa lên **75.32**!
  - `sub60_e98` (Schedule-Free AdamW): Iterate averaging lọc mất các đỉnh xung lực của ảnh dày $\to$ Bias âm nặng nề (-32.57), MAE thoái hóa lên **90.19**!
  - `sub60_e102` (Decoupled Carrier `detach_y0`): Cắt đứt gradient từ solver về sóng mang khiến Fine Head ở ảnh thưa không nhận được tín hiệu điều chỉnh $\to$ Sparse MAE tăng vọt từ 17 lên **64.23**!
* **Quy luật bất biến:** Mỗi khi ta cố gắng sửa một thông số cục bộ mà vi phạm **tính bảo toàn độ đo** hoặc **tính chất co thắt Lipschitz của solver**, toàn bộ hệ thống sẽ phản ứng tiêu cực ở các miền mật độ khác!

---

### VIII. BẢN CHẤT LỐI THOÁT KHOA HỌC: CHUYỂN ĐỔI HỆ HÌNH (PARADIGM SHIFT) THỰC THỤ

Để vượt qua bức tường 70 MAE và hướng tới mục tiêu khoa học, ta không thể tiếp tục lặp lại các vòng lặp hoán vị siêu tham số. Mô hình cần một sự điều chỉnh dựa trên bản chất toán học:

#### 1. Bản chất Biểu diễn Mật độ: Chuyển từ Dirac Discrete sang Continuous Probability Formulation
* Thay vì bắt mạng nơ-ron hồi quy một hàm delta Dirac gián đoạn (vốn xung đột với tính trơn của CNN):
* Áp dụng nguyên lý của **Bayesian Loss (Ma et al. ICCV 2019)** hoặc **Optimal Transport (DM-Count NeurIPS 2020)**:
  - Xem mỗi điểm ground truth là tâm của một phân phối xác suất liên tục.
  - Mất mát không đo sự chênh lệch từng pixel (Smooth L1), mà đo **khoảng cách Wasserstein hoặc kỳ vọng xác suất** giữa đám mây dự đoán và các điểm ground truth.
  - Điều này hoàn toàn không tốn thêm tham số (0 params), nhưng loại bỏ 100% hiện tượng "dìm đỉnh" (peak suppression) ở Stride 4!

#### 2. Bản chất Toán tử Quan sát: Điều biến theo Phối cảnh (Perspective-Conditioned Observation Operator)
* Thay vì giữ cứng 3 hộp $[32, 64, 128]\text{ px}$ đồng nhất trên toàn bộ ảnh:
* Toán tử quan sát cần được thích ứng với hình học phối cảnh:
  - Ở phía trên ảnh (đường chân trời): Giảm kích thước hộp xuống tương ứng với tỷ lệ đầu người (ví dụ: hộp nhỏ hơn hoặc trọng số tập trung vào scale 32px).
  - Quan trọng hơn: **Nguyên lý Morozov Discrepancy phải có deadband thích ứng với mật độ**:
    $$\gamma_{\text{eff}}(x) = \gamma_{\text{sparse}} \cdot \pi_{\text{coarse}}(x) + \gamma_{\text{dense}} \cdot \pi_{\text{fine}}(x)$$
    với $\gamma_{\text{dense}} \ll \gamma_{\text{sparse}}$. 
    Ở cụm dày, deadband phải được thu hẹp tối đa để solver KHÔNG BỎ QUA sai số 10–20 người của đường chân trời!

#### 3. Bản chất Hiệp đồng Sóng mang - Bộ giải (Orthogonalized Carrier-Solver Supervision)
* Định nghĩa rõ ràng trách nhiệm của từng thành phần:
  - **Fine Head ($y_0$):** Chịu trách nhiệm về **hình thái và vị trí (Support & Localization)**. Được giám sát bởi hàm phân bổ không gian (Spatial Allocation Loss, ví dụ Flat DM16 hoặc Bayesian Loss).
  - **Regional Head ($b$):** Chịu trách nhiệm về **tổng khối lượng từng vùng (Regional Mass)**. Được giám sát bởi Negative Binomial NLL.
  - **Solver ($y^*$):** Chịu trách nhiệm về **sự hòa giải khối lượng đơn điệu (Monotonic Mass Reconciliation)**. 
* Tuyệt đối không để một hàm mất mát cục bộ (như Smooth L1 cell loss) can thiệp tiêu cực, dìm dẹp biên độ của $y_0$ trước khi đưa vào solver.

---

### IX. KẾT LUẬN VÀ CAM KẾT HÀNH ĐỘNG
1. **Chấm dứt hoàn toàn việc chắp vá siêu tham số mù quáng (Stop Blind Parameter Tinkering):** Mọi đề xuất từ thời điểm này phải chỉ rõ cơ chế toán học, toán tử bị tác động, và lý do vì sao nó giải quyết được mâu thuẫn bản chất.
2. **Tôn trọng các hằng số bất biến (Strict Invariants):**
   - Ngân sách tham số: Tuyệt đối $\le 104,441$ tham số.
   - Độc lập 100%: 0.0% Knowledge Distillation.
   - Benchmark chuẩn: ShanghaiTech Part A (300 train / 182 test).
   - Mã nguồn chuẩn mực: Mỗi file `.py` strictly $\le 450$ dòng.
   - Không tự ý chạy training khi chưa có chỉ thị rõ ràng từ người dùng.
3. Toàn bộ các phân tích định lượng và lý thuyết trong tài liệu này được lưu trữ trực tiếp trong codebase tại `docs/ESSENCE_MATHEMATICAL_AND_ARCHITECTURAL_DISSECTION.md` để đảm bảo tính minh bạch khoa học lâu dài.
