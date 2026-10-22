# PROJECT RULES & OPERATIONAL INVARIANTS

## 1. Strict Script Immutability Rule (BẮT BUỘC)
- **TUYỆT ĐỐI KHÔNG ĐƯỢC TỰ Ý TẠO HAY SỬA BẤT KỲ FILE `.sh` HOẶC `.ps1` NÀO** nếu không có yêu cầu rõ ràng và sự cho phép cụ thể từ người dùng.
- Khi người dùng yêu cầu lệnh chạy thực nghiệm, **PHẢI CUNG CẤP TỪNG DÒNG LỆNH RIÊNG LẺ TRỰC TIẾP TRONG TEXT PHẢN HỒI (markdown code blocks)** để người dùng tự sao chép và thực thi trong terminal của họ.
- Không tự ý sinh ra các batch runner files (`.sh`, `.ps1`, `.bat`) hay chỉnh sửa các file script chạy sẵn có trong thư mục `scripts/`.

---

## 2. Parameter Ceiling & Knowledge Distillation (BẤT BIẾN NGHIÊN CỨU)
- **Tham số mô hình**:
  - **Track A (Anchor siêu nhẹ)**: Tuyệt đối $\le 104,441$ tham số có thể huấn luyện (`trainable parameters`).
  - **Track B (Mở rộng dung lượng)**: Tuyệt đối $\le 200,000$ tham số có thể huấn luyện (theo giới hạn người dùng cho phép).
  - Bất kỳ cấu hình nào vượt quá trần tham số của từng track tương ứng đều bị loại bỏ ngay từ bước preflight check.
- **Không dùng Knowledge Distillation**: Tuyệt đối $0.0\%$ tri thức chưng cất, không sử dụng bất kỳ mạng giáo viên (teacher model) nào. Mô hình phải hoạt động độc lập 100% (Standalone).

---

## 3. Quy chuẩn Mã Nguồn & Kích thước File (CODE HEALTH INVARIANT)
- Tất cả các file mã nguồn Python trong `rmr_core/` và `rmr_v3/` **bắt buộc phải có độ dài $\le 450$ dòng** (được kiểm chứng tự động bởi test suite).
- Không thêm các thư viện phụ thuộc rác. Không monkey patch. Sửa lỗi đúng gốc rễ toán học.

---

## 4. Giao thức Chống Thoái lui (ANTI-REGRESSION PROTOCOL)
- Mọi thí nghiệm mới phải được đối chứng trực tiếp với mỏ neo tốt nhất lịch sử repo (`sub60_e5`, **71.51 MAE**).
- Bảo toàn nguyên vẹn 3 điều kiện cân bằng tinh tế trừ khi có đối chứng đơn biến chứng minh độc lập:
  1. `curvature_alpha_init: -8.0` ($\alpha_{\text{eff}} \approx 0.000335$, ngăn nổ mật độ bậc hai ở cụm dày).
  2. `hurdle_gating_mode: product` (hệ số co thắt Lipschitz bảo vệ sóng mang trong SIRT).
  3. `lambda_cell: 0.0` (ngăn hiện tượng đói gradient $\mathcal{O}(1/N)$ ở cụm dày).
- Báo cáo số liệu trung thực từ file `summary.json` đã đánh giá thật trên tập test chuẩn ShanghaiTech Part A (182 ảnh), không đưa ra số liệu suy đoán hoặc từ ngữ sáo rỗng.

---

## 5. Quy chuẩn Commit & Push Luôn Luôn Khi Edit (CONTINUOUS COMMIT & PUSH INVARIANT)
- **BẮT BUỘC LUÔN LUÔN COMMIT VÀ PUSH NGAY LẬP TỨC SAU MỖI LẦN EDIT / CHỈNH SỬA**:
  - Bất kỳ khi nào tạo mới file, chỉnh sửa mã nguồn Python, cập nhật cấu hình YAML, hoặc viết tài liệu/báo cáo: **PHẢI THỰC HIỆN `git add`, `git commit` VÀ `git push` NGAY LẬP TỨC**.
  - Commit message phải rõ ràng, tuân thủ nghiêm ngặt chuẩn Conventional Commits (ví dụ: `feat:`, `fix:`, `chore:`, `docs:`).
  - Tuyệt đối không để dồn ứ thay đổi chưa commit (`uncommitted changes`) hoặc commit chưa push (`unpushed commits`) qua các lượt tương tác với người dùng.
  - Luôn kiểm tra `git status` và `git push` thành công trước khi kết thúc câu trả lời.

---

## 6. Tuyệt Đối Nghiêm Cấm Làm Giả Số Liệu & Cấm Dùng Oracle Giả (ABSOLUTE EMPIRICAL INTEGRITY & NO FAKE ORACLES INVARIANT)
- **TUYỆT ĐỐI NGHIÊM CẤM LÀM GIẢ SỐ LIỆU (ZERO DATA FABRICATION)**:
  - Mọi chỉ số (MAE, RMSE, NAE, Bias, Help Rate, v.v.) báo cáo cho người dùng **BẮT BUỘC** phải được trích xuất trực tiếp, nguyên bản 100% từ kết quả inference thực tế của mô hình thật trên tập dữ liệu kiểm thử chuẩn ShanghaiTech Part A (182 ảnh test đầy đủ).
  - Nghiêm cấm mọi hành vi bịa đặt, suy đoán, làm tròn có chủ đích, gọt giũa số liệu (cherry-picking / smoothing), hoặc sửa đổi kết quả trong `summary.json`, `train_log.csv` hay bất kỳ file báo cáo nào.

- **TUYỆT ĐỐI NGHIÊM CẤM DÙNG ORACLE GIẢ ĐỂ BÁO CÁO HAY ĐÁNH LẬN KẾT QUẢ (STRICT BAN ON FAKE ORACLES)**:
  - **Không bao giờ dùng Oracle làm thước đo thành tựu**: Chỉ số Oracle (chọn mô hình dự đoán tốt nhất cho từng ảnh dựa vào nhãn Ground Truth) là phi thực tế và mang tính "gian lận nhãn kiểm thử" (Test Label Leakage). Tuyệt đối KHÔNG được dùng chỉ số Oracle để làm đẹp báo cáo, tạo ảo tưởng rằng mô hình đã đạt Sub-50 hay Sub-60, hoặc nói giảm nói tránh thất bại của mô hình đơn lẻ.
  - **Tiêu chuẩn duy nhất được công nhận**: Mọi đánh giá hiệu năng bắt buộc phải là của **MỘT MÔ HÌNH ĐỘC LẬP DUY NHẤT (Single Standalone Model Checkpoint)** chạy inference đầu cuối thực tế, hoàn toàn mù với nhãn Ground Truth.
  - **Nếu có dùng Ensemble**: Chỉ được phép đánh giá các mô hình Ensemble thực tế có thể triển khai sản xuất độc lập (ví dụ: trung bình cộng tĩnh cố định giữa các mô hình đã huấn luyện xong), tuyệt đối không được dùng Ground Truth để gán nhánh (Gated by GT) hoặc tạo Oracle kết hợp.

- **TRUNG THỰC KHOA HỌC TUYỆT ĐỐI (EMPIRICAL BRUTAL HONESTY)**:
  - Mô hình đạt kết quả bao nhiêu phải báo cáo chính xác bấy nhiêu.
  - Khi mô hình bị thoái hóa hoặc thất bại (kết quả xấu hơn mỏ neo cũ), phải thừa nhận thẳng thắn, báo cáo chính xác số điểm tụt lùi, giải phẫu đúng nguyên nhân kỹ thuật, không dùng thuật ngữ hoa mỹ để che đậy hoặc dùng Oracle để "vớt vát" tâm lý.
---

## 7. Lệnh Cấm Vĩnh Viễn WSD & Lịch Trình Cắt Ngắn 650 Epoch (PERMANENT BAN ON WSD & SHORT DECAY)
- **TUYỆT ĐỐI CẤM SỬ DỤNG WSD (Warmup-Stable-Decay)**: Cấm vĩnh viễn việc sử dụng `scheduler_type: wsd`, `warmup_stable_decay`, hoặc tham số `wsd_stable_ratio` trong toàn bộ cấu hình YAML và mã nguồn Python.
- **CƠ CHẾ THẤT BẠI ĐÃ CHỨNG MINH**: Trên 300 ảnh train nhỏ, giữ LR cao quá lâu khiến tham số dao động tán loạn, sau đó decay gấp gáp chỉ trong 100 epoch làm đóng băng mô hình ở lòng chảo cực tiểu tồi, kéo kết quả thoái hóa lên 98-102 MAE.
- **QUY CHUẨN BẮT BUỘC**:
  1. Bắt buộc dùng **Cosine Annealing chuẩn (1,000 epochs, warmup 10, min_lr 0.05)**.
  2. Tuyệt đối không dùng Early Stopping cắt ngắn quá trình huấn luyện.

---

## 8. Quy Chuẩn Vệ Sinh Mã Nguồn & Công Cụ (CODEBASE HYGIENE & TOOL STANDARDIZATION)
- **TUYỆT ĐỐI KHÔNG TẠO SCRIPT DÙNG 1 LẦN BỪA BÃI**: Cấm để lại các script tạm bợ (`analyze_*.py`, `step*.py`, `inspect_*.py`) trong `tools/`, `scripts/` hay `scratch/`.
- **THƯ MỤC `tools/` CHỈ CHỨA CLI TOOLS TÁI SỬ DỤNG**: Chỉ giữ lại các công cụ tổng quát có tham số CLI rõ ràng (`--config`, `--checkpoint`).
- **DỌN DẸP NGAY LẬP TỨC**: Mọi file tạm sinh ra trong quá trình gỡ lỗi phải được xóa sạch trước khi commit.


---

## 9. Bốn Thủ Phạm Kỹ Thuật Bị Cấm Tuyệt Đối (STRICT BAN ON 4 REGRESSION CULPRITS)
*Chi tiết cơ sở vật lý và toán học xem tại `LUU_Y_THIET_KE_THI_NGHIEM_ANTI_REGRESSION.md`.*
1. **CẤM BẬT `use_diag: True`**: Mặc định bắt buộc `use_diag: False`. Global Average Pooling làm ô nhiễm định tuyến thang đo và phạt +4.96 MAE.
2. **CẤM SCALE JITTER QUÁ ĐÀ**: Bắt buộc `scale_range: [0.6, 1.4]`. Tuyệt đối không dùng cận dưới $< 0.5$ (gây aliasing phá hủy micro-head trên Stride 4).
3. **CẤM BỎ BAYESIAN POINT LOSS TRÊN TRACK A**: Track A (Stride 4) bắt buộc dùng `dual_bayesian_dm16` (`lambda_bayesian: 0.15`) để bảo toàn định vị điểm trong khối $64 \times 64$ px.
4. **BẮT BUỘC DENSITY-GATING CHO FED-CARRIER**: FED-Carrier chỉ được kích hoạt khi $b_s > 2.0$, cấm bơm mass tự do vào cảnh thưa và trung bình ($b_s \le 2.0$).
