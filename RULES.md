# PROJECT RULES & OPERATIONAL INVARIANTS

## 1. Strict Script Immutability Rule (BẮT BUỘC)
- **TUYỆT ĐỐI KHÔNG ĐƯỢC TỰ Ý TẠO HAY SỬA BẤT KỲ FILE `.sh` HOẶC `.ps1` NÀO** nếu không có yêu cầu rõ ràng và sự cho phép cụ thể từ người dùng.
- Khi người dùng yêu cầu lệnh chạy thực nghiệm, **PHẢI CUNG CẤP TỪNG DÒNG LỆNH RIÊNG LẺ TRỰC TIẾP TRONG TEXT PHẢN HỒI (markdown code blocks)** để người dùng tự sao chép và thực thi trong terminal của họ.
- Không tự ý sinh ra các batch runner files (`.sh`, `.ps1`, `.bat`) hay chỉnh sửa các file script chạy sẵn có trong thư mục `scripts/`.

---

## 2. Parameter Ceiling & Knowledge Distillation (BẤT BIẾN NGHIÊN CỨU)
- **Tham số mô hình**: Tuyệt đối $\le 105,000$ tham số có thể huấn luyện (`trainable parameters`). Bất kỳ cấu hình nào vượt quá $105,000$ tham số đều bị loại bỏ ngay từ bước preflight check.
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
