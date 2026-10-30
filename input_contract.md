# Input Contract: Empirical Ground Truth, Hard Constraints, and Falsification Protocols

**Trạng thái**: Hoàn thiện bởi **Dissector Agent** (Reverse Engineer)  
**Mục tiêu**: Thiết lập biên bản giao kèo sự thật kỹ thuật làm đầu vào bất biến cho **Strategist Agent**. Ngăn chặn mọi hình thức hallucination, narrative bias, và cherry-picking trong toàn bộ quy trình nghiên cứu và công bố khoa học.  
**Mã nguồn đối chiếu**: Nhánh `RMR`, commit [`d6df22f`](file:///f:/lightweightcrcn/model_facts.md) trên kho lưu trữ `lightweightcrcn`.

---

## Phần A: Model Facts (Technical Inventory & Code-Level Ground Truth)

Toàn bộ chi tiết kỹ thuật từng dòng code của kiến trúc, kiểm toán cơ chế, tính khả tái lập, đối chiếu tuyên bố và sơ đồ phụ thuộc đã được nghiệm thu tại tệp [`model_facts.md`](file:///f:/lightweightcrcn/model_facts.md).

### 1. Tóm tắt bảng tham số kiến trúc thực tế (`rmr_v3/model/architecture.py:L40-L106`)
- `encoder` (`TimmPyramidBackbone` / MobileNetV4-Conv-Small-050): **87,568** learnable params, 3,352 buffers (`rmr_core/backbones.py:L7-L96`).
- `fusion` (`ASPPLiteFPNNeck`): **10,784** learnable params (`rmr_core/necks/aspp.py:L10-L98`).
- `fine_head` (`FineMeasureHead`): **1,473** learnable params (`rmr_core/heads.py:L25-L200`).
- `region_head` (`ProbabilisticRegionalEvidenceHead`): **4,082** learnable params (`rmr_v3/regional_head.py:L31-L270`).
- `scale_router` (`ScaleRoutingHead`): **483** learnable params (`rmr_core/scale_routing.py:L23-L85`).
- `solver` (`closed_form_mcbpro_transport`): **0** learnable params (`rmr_core/operators/mcbpro.py:L45-L221`).
- **Tổng tham số học được thực tế**: **104,390 tham số** (51 tham số dưới trần 104,441; nhưng $104,390 > 100,000$, không thuộc phân khúc sub-100k).

### 2. Hai phát hiện thực nghiệm bổ sung từ Dissector (Empirical Breakthroughs)
- **Phát hiện 1: Nguồn gốc của Bias âm (-17 đến -25) và Bottleneck của Solver**:
  Đo đạc thực nghiệm trên toàn bộ 182 ảnh kiểm thử của ShanghaiTech Part A (`scratch/test_bias_and_determinism.py`):
  - *Fine Head đơn lẻ ($y_0$, không dùng solver)*: **MAE 65.23**, RMSE 108.40, Bias -17.44.
  - *Mô hình đầy đủ ($y$, MC-BPRO + Regional Head dự đoán)*: **MAE 65.57**, RMSE 111.34, Bias -19.78.
  - *Oracle Solver ($y_{oracle}$, MC-BPRO + Ground-Truth Box Counts)*: **MAE 4.77**, RMSE 5.78, Bias +4.70.
  - *Ý nghĩa khoa học*: Toán tử MC-BPRO hoàn toàn chính xác về mặt toán học (nếu có bằng chứng vùng chuẩn, mô hình đạt MAE 4.77). Tuy nhiên, trong mô hình triển khai thực tế, **Regional Evidence Head dự đoán bị undercount ở các vùng đông**, kéo bản đồ mật độ mịn $y_0$ đi xuống, khiến solver thực tế làm MAE tăng nhẹ từ 65.23 lên 65.57 và kéo bias từ -17.44 xuống -19.78.
- **Phát hiện 2: Nguồn gốc của tính Non-Deterministic giữa các run**:
  - Phép toán forward-backward trên cùng batch tensor cố định trên GPU là đơn định tuyệt đối ($\Delta = 0.00000000$).
  - Sự dao động kết quả giữa các lần chạy bắt nguồn từ cấu hình `deterministic: false` trong `configs/rmr_canonical/rmr_canonical_104k_seed42.yaml:L153`:
    1. DataLoader không có generator (`gen = None`, `rmr_v3/trainer.py:L171`) kết hợp với 4 tiến trình con bất đồng bộ (`workers: 4`, `persistent_workers: True`) làm thứ tự nạp batch bị xáo trộn giữa các epoch.
    2. Các phép tăng cường dữ liệu ngẫu nhiên (RandomErasing, RandomScale) áp dụng không đồng bộ.
    3. `cudnn.benchmark = True` và `TF32 = True` làm tích lũy sai số trôi nổi sau 1500 epochs tối ưu AdamW.

---

## Phần B: Ràng buộc cứng (Hard Invariant Constraints)

Mọi đề xuất, bài báo, và phân tích từ Strategist và Writer **BẮT BUỘC** phải tuân thủ 5 ràng buộc cứng sau:

1. **Trần ngân sách tham số (Parameter Ceiling)**:
   - Nghiêm cấm mọi sửa đổi kiến trúc làm tổng số tham số học được vượt quá **104,441 tham số** (`max_trainable_params: 104441`).
   - Mọi phân loại phân khúc phải ghi chính xác là **sub-105k** hoặc **sub-150k**. Tuyệt đối không dùng cụm từ "sub-100k" cho mô hình 104,390 tham số.

2. **Quy chuẩn tập dữ liệu và phân chia (Dataset & Split Contract)**:
   - Sử dụng chuẩn ShanghaiTech Part A: đúng **300 ảnh huấn luyện** (`data/sha_a_train_all.jsonl`) và đúng **182 ảnh kiểm thử đầy đủ** (`data/sha_a_test.jsonl`).
   - Không cắt crop kiểm thử, không rescale ảnh kiểm thử (chỉ dùng zero-padding chia hết cho 16 ở đầu vào).
   - Zero Transfer / Pretraining ngoài ImageNet-1k tiêu chuẩn trên backbone.

3. **Không chưng cất tri thức (Zero Distillation Contract)**:
   - Huấn luyện độc lập 100% từ đầu (standalone training). Không dùng mô hình giáo viên (Teacher), không dùng loss tương đồng đặc trưng (feature mimicking) từ Vision Transformers hoặc mô hình nặng.

4. **Chuẩn mực báo cáo đa-seed (Multi-Seed Reliability Contract)**:
   - Bất kỳ kết luận hoặc bảng số liệu chính nào trong bài báo phải được báo cáo trên tối thiểu **$n = 3$ random seeds độc lập** dưới dạng:
     $$\text{Metric} = \text{Mean} \pm \text{Std} \quad [\text{CI}_{95\%}^{\text{lower}}, \text{CI}_{95\%}^{\text{upper}}]$$
   - Tuyệt đối cấm báo cáo một seed đơn lẻ (ví dụ Seed 42 = 65.62) như là đại diện hiệu năng của mô hình.

5. **Tính truy vết và Khả tái lập (Traceability Contract)**:
   - Mọi checkpoint đưa vào bài báo phải lưu kèm tệp `resolved_config.yaml`, commit hash git, danh sách prediction từng ảnh `predictions.csv`, và tệp tóm tắt `summary.json`.

---

## Phần C: Bảng kiểm kê giả thuyết có thể phản chứng (Claim Inventory & Falsification Protocols)

Các tuyên bố narrative cũ được viết lại thành các giả thuyết khoa học hình thức với ngưỡng định lượng và điều kiện bác bỏ rõ ràng:

### Giả thuyết $H_1$: Tác động thực sự của Giám sát Đối ngẫu (Dual Supervision) vs Pure Bayesian
- **Nội dung giả thuyết**: "Giám sát đối ngẫu (Dual Supervision kết hợp Radon-OT SWD và Bayesian Point Supervision) không cải thiện MAE tổng hoặc RMSE so với Pure Bayesian trên ShanghaiTech Part A, nhưng cải thiện đáng kể độ lệch dự đoán (Bias) và giảm sai số ở vùng thưa (Sparse MAE) ít nhất 5.0 điểm."
- **Ngưỡng kiểm chứng định lượng**:
  1. $|\text{Mean MAE}_{\text{Dual}} - \text{Mean MAE}_{\text{Bayesian}}| \le 1.0$ MAE trên $n=3$ seeds.
  2. $\text{Mean Sparse MAE}_{\text{Dual}} \le \text{Mean Sparse MAE}_{\text{Bayesian}} - 5.0$ với khoảng tin cậy 95% không chồng lấp.
  3. $|\text{Mean Bias}_{\text{Dual}}| < |\text{Mean Bias}_{\text{Bayesian}}|$ (ít bị undercount hơn).
- **Điều kiện bác bỏ (Falsification Criteria)**:
  - Bác bỏ nếu trên $n=3$ seeds, Pure Bayesian có Sparse MAE ngang bằng hoặc tốt hơn Dual Supervision ($\Delta \le 2.0$).
  - Bác bỏ nếu Dual Supervision vượt trội Pure Bayesian trên MAE tổng $> 2.0$ MAE (nếu điều này xảy ra thì narrative "Radon-OT cải thiện MAE tổng" mới được phục hồi).

### Giả thuyết $H_2$: Bản chất của Toán tử Hồi phục Độ đo MC-BPRO
- **Nội dung giả thuyết**: "MC-BPRO là một toán tử bảo toàn độ đo dạng đóng (closed-form, 0 tham số, $O(1)$ graph complexity). Tiềm năng hồi phục của toán tử phụ thuộc tuyến tính vào độ chính xác của bằng chứng vùng: khi nhận bằng chứng vùng hoàn hảo (Oracle GT box count), sai số hồi phục giảm xuống dưới 5.0 MAE; nhưng khi kết hợp với Regional Evidence Head học được hiện tại, solver bị giới hạn bởi độ lệch của đầu vùng và không vượt qua Fine Head thô $y_0$."
- **Ngưỡng kiểm chứng định lượng**:
  1. $\text{MAE}_{\text{Oracle Solver}} \le 5.0$ (kết quả thực nghiệm thực tế: 4.77).
  2. $|\text{MAE}_{\text{Full Solver}} - \text{MAE}_{y_0}| \le 0.5$ MAE (kết quả thực nghiệm: 65.57 vs 65.23).
  3. Số tham số học được của solver $= 0$.
- **Điều kiện bác bỏ**:
  - Bác bỏ nếu Oracle Solver không đưa được MAE về $< 10.0$ (chứng tỏ toán tử bị lỗi toán học nội tại).
  - Bác bỏ nếu chứng minh được solver cải thiện được MAE mà không cần phụ thuộc vào độ chính xác của đầu vùng.

### Giả thuyết $H_3$: Khả thi của Direct Neural Operator ở Lưới mịn Stride 2
- **Nội dung giả thuyết**: "Ở độ phân giải lưới mịn Stride 2 ($256 \times 256$), việc loại bỏ hoàn toàn unrolled solver và chỉ sử dụng mạng nơ-ron truyền thẳng (Direct Neural Operator, $y = y_0$) đạt hiệu năng cạnh tranh $\le 68.0$ MAE trên $n=3$ seeds, đồng thời giảm 50% độ trôi dạt Bias so với mô hình có solver ở Stride 4."
- **Ngưỡng kiểm chứng định lượng**:
  1. $\text{Mean MAE}_{\text{Stride2}} \le 68.0$ trên $n=3$ seeds.
  2. $|\text{Mean Bias}_{\text{Stride2}}| \le 12.0$ (so với $-19.5$ đến $-24.8$ của Stride 4).
- **Điều kiện bác bỏ**:
  - Bác bỏ nếu $\text{Mean MAE}_{\text{Stride2}} > 70.0$ hoặc độ biến thiên giữa các seed $> 8.0$ MAE.

### Giả thuyết $H_4$: Định vị Cạnh tranh Thực tế trong Phân khúc Siêu nhẹ (Lightweight SOTA)
- **Nội dung giả thuyết**: "Trong phân khúc mô hình siêu nhẹ $\le 105k$ tham số không dùng distillation, RMRv3 cạnh tranh với TinyCount (78.2 MAE, 104k) và vượt qua các baseline lịch sử (SANet 67.0, TEDnet 64.2) ở mức độ tối ưu tham số (104k vs 8.0M và 1.6M params), nhưng chưa thể vượt qua các mô hình chuyên biệt dẫn đầu như EBC-ZIP-S (57.2 MAE, 84k) hay FFNet (48.3 MAE)."
- **Ngưỡng kiểm chứng định lượng**:
  1. Bảng so sánh SOTA phải trích dẫn đầy đủ: EBC-ZIP-S (57.2), FFNet (48.3), HMSTUNet (49.1), TEDnet (64.2), SANet (67.0), TinyCount (78.2).
  2. Không đưa ra bất kỳ tuyên bố "World Record" nào.
- **Điều kiện bác bỏ**:
  - Bác bỏ nếu bất kỳ tuyên bố nào so sánh RMRv3 mà bỏ qua EBC-ZIP-S hoặc khẳng định RMRv3 đạt SOTA tuyệt đối phân khúc sub-100k.

### Giả thuyết $H_5$: Kiểm soát Tính Khả tái lập (Determinism Control)
- **Nội dung giả thuyết**: "Việc thiết lập `deterministic: true` kết hợp cố định `DataLoader` generator với `torch.backends.cudnn.benchmark = False` sẽ kiểm soát phương sai chạy lặp lại trên cùng một seed xuống mức sai số chấp nhận được $\le 0.5$ MAE."
- **Ngưỡng kiểm chứng định lượng**:
  - Chênh lệch giữa 2 lần chạy lại cùng seed 42 với cờ deterministic: $|\text{MAE}_1 - \text{MAE}_2| \le 0.5$.
- **Điều kiện bác bỏ**:
  - Bác bỏ nếu dù đã cố định toàn bộ các cờ đơn định, sai số giữa 2 lần chạy cùng seed vẫn $> 2.0$ MAE.

---

## Phần D: Lưới thực nghiệm bắt buộc trước khi kích hoạt Strategist (Mandatory Experiment Grid)

Trước khi Strategist hoàn thiện câu chuyện nghiên cứu cuối cùng, 4 thực nghiệm sau là điều kiện tiên quyết:

| Mã TN | Mục tiêu thực nghiệm | Cấu hình & Lệnh chạy | Kết quả hiện có ($n$) | Kết quả cần bổ sung ($n=3$) |
|---|---|---|---|---|
| **D1** | Bổ sung 2 seed cho Ablation 3 (Pure Bayesian) để đối chiếu trực tiếp với Full Model | `configs/rmr_canonical/ablations/rmr_abl_no_radon_ot.yaml`<br>Seed 7 và Seed 1337 | $n=1$ (Seed 42: 65.74 MAE, 108.27 RMSE) | Cần chạy **Seed 7** và **Seed 1337** |
| **D2** | Bổ sung 1 seed cho Full RMR Canonical (Track A Stride 4) | `configs/rmr_canonical/rmr_canonical_104k_seed42.yaml`<br>Seed 1337 | $n=2$ (Seed 42: 65.62, Seed 7: 72.20; Mean: 68.91) | Cần chạy **Seed 1337** để hoàn tất $n=3$ |
| **D3** | Bổ sung 2 seed cho Track B (Stride 2 Direct Neural Operator) | `configs/rmr_canonical/rmr_canonical_stride2_seed42.yaml`<br>Seed 7 và Seed 1337 | $n=1$ (Seed 42: 67.20 MAE, 110.70 RMSE) | Cần chạy **Seed 7** và **Seed 1337** |
| **D4** | Kiểm tra kiểm soát tính đơn định (Determinism Verification) | Chạy 50 epochs trên Seed 42 với `deterministic: true` vs `deterministic: false` | Đã xác nhận single-batch determinism = 0 | Chạy kiểm tra trên tập con train/val |

---

## Phần E: Kho lưu trữ kết quả tiêu cực (Negative Results Archive)

Một bài báo khoa học chuẩn mực phải công khai trung thực các kết quả không như kỳ vọng. Đây là tài sản bảo vệ bài báo khỏi việc bị reviewer phản bác trong quá trình bình duyệt:

1. **Kết quả tiêu cực 1: Biến thiên nhạy cảm theo Seed (Seed Fragility)**
   - Seed 42 đạt 65.62 MAE, nhưng Seed 7 tăng vọt lên **72.20 MAE** ($\Delta = +6.58$ MAE, RMSE tăng từ 109.16 lên 125.45).
   - Trung bình 2 seed là **68.91 MAE**. Con số này kém mô hình kinh điển SANet (67.0) và TEDnet (64.2). Việc che giấu Seed 7 sẽ khiến bài báo bị quy kết là cherry-picking.

2. **Kết quả tiêu cực 2: Ablation 3 không hề sụp đổ mà ngang ngửa Full Model**
   - Không có Radon-OT (`lambda_radon_ot: 0.0`), Pure Bayesian đạt **65.74 MAE** (chỉ lệch 0.12 so với Full 65.62).
   - Đặc biệt, Ablation 3 có **RMSE tốt hơn Full** (108.27 vs 109.16) và **Dense MAE tốt hơn Full** (123.74 vs 127.67).
   - *Kết luận tiêu cực*: Radon-OT không có đóng góp đo lường được vào việc giảm MAE tổng hay RMSE trên tập dữ liệu này; nó chỉ đóng vai trò thu hẹp độ lệch bias ở vùng thưa.

3. **Kết quả tiêu cực 3: TTA (Test-Time Augmentation) không mang lại cải thiện**
   - Trên cả 3 cấu hình thử nghiệm, Horizontal Flip TTA đều làm MAE tăng nhẹ thay vì giảm:
     - Seed 42: 65.62 $\rightarrow$ 65.69 (+0.07).
     - Seed 7: 72.20 $\rightarrow$ 72.72 (+0.52).
     - Track B Stride 2: 67.20 $\rightarrow$ 67.99 (+0.79).
   - *Kết luận tiêu cực*: Mô hình không hưởng lợi từ phép lật TTA thông thường.

4. **Kết quả tiêu cực 4: Regional Evidence Head là nguyên nhân gây Undercount Bias**
   - Fine Head thô ($y_0$) tự đạt MAE 65.23 và Bias -17.44.
   - Khi đưa qua MC-BPRO solver với Regional Head dự đoán, MAE tăng lên 65.57 và Bias tệ đi thành -19.78.
   - Trong khi đó, nếu dùng GT box count thì MAE đạt **4.77**.
   - *Kết luận tiêu cực*: Đầu dự đoán vùng xác suất hiện tại bị bão hòa/undercount ở các đám đông lớn, kéo lùi độ chính xác của solver. Solver không thể phát huy sức mạnh nếu đầu vùng không chuẩn xác.

5. **Kết quả tiêu cực 5: Mô hình vượt ngưỡng 100k tham số và chưa vượt SOTA dẫn đầu**
   - Mô hình có **104,390 tham số** ($> 100k$).
   - Trong cùng phân khúc dưới 150k tham số, EBC-ZIP-S đạt **57.2 MAE** (ít hơn 20k tham số và tốt hơn 11.71 MAE).
   - *Kết luận tiêu cực*: Bài báo không thể định vị theo kiểu "phá kỷ lục tuyệt đối thế giới" mà phải định vị theo hướng "nghiên cứu thực nghiệm giải phẫu toán tử độ đo và tính hiệu quả tính toán".

---

## Xác nhận Nghiệm thu của Dissector
Bản giao kèo `input_contract.md` này đã phản ánh 100% sự thật kỹ thuật từ mã nguồn và dữ liệu thực nghiệm. Tất cả các phát hiện tiêu cực đã được ghi nhận đầy đủ. Agent **Strategist** chỉ được phép xây dựng bài toán nghiên cứu (Research Questions) và câu chuyện khoa học dựa trên các ràng buộc cứng và kho kết quả tiêu cực nêu trên.
