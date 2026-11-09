# AGENTS.md — Global Operating Rules for All AI Agents in this Repository

## 1. LỆNH CẤM TUYỆT ĐỐI VÀ KHÔNG NGOẠI LỆ TỪ "PARETO" (ABSOLUTE & UNCONDITIONAL BAN ON "PARETO")
- **BẢN CHẤT TOÁN HỌC & GIẢI TÍCH**:
  - Bài toán trong repository này là bài toán nghịch đảo khôi phục độ đo Radon liên tục có ràng buộc (**Constrained Radon Measure Recovery**): $b = \mathcal{A}_\pi y + \eta$ dưới ngân sách tham số cực tiểu ($\le 104\text{k}$ params, 0% Knowledge Distillation).
  - Quá trình tối ưu bằng AdamW cực tiểu hóa một hàm mất mát vô hướng duy nhất $\mathcal{L}_{\text{total}}$, **hoàn toàn không phải tối ưu hóa đa mục tiêu tiến hóa Pareto** (như NSGA-II hay MOEA/D).
  - Việc gọi biểu đồ phân tán 2D (Scatter Plot: MAE vs Params/FLOPs) là "Pareto" là thói quen lạm dụng buzzword sáo rỗng của AI, thể hiện sự thiếu hiểu biết về toán học tối ưu, gây thiên kiến (bias) nặng nề, và **BỊ CẤM TRIỆT ĐỂ KHÔNG CÓ NGOẠI LỆ**.
- **QUY TẮC BẮT BUỘC CHO MỌI AGENT VÀ SUBAGENT**:
  - **TUYỆT ĐỐI CẤM SỬ DỤNG TỪ "PARETO"** dưới mọi hình thức trong:
    * Tiêu đề, Tóm tắt, Problem Statement, Research Objectives (RO), Research Questions (RQ).
    * Phân tích, Thảo luận, Thực nghiệm, Bảng biểu, Chú thích hình vẽ.
    * Giao tiếp với người dùng và chỉ đạo giữa các subagents.
  - **Thuật ngữ thay thế chuẩn xác**:
    * Tiếng Việt: *"Biểu đồ đánh đổi thực nghiệm độ phức tạp - độ chính xác"* hoặc *"Biểu đồ phân tán tham số - sai số"*.
    * Tiếng Anh: *"Empirical Complexity-Accuracy Trade-off Plot"* hoặc *"Parameter vs. Error Scatter Plot"*.

---

## 2. TÍNH BẤT BIẾN CỦA KỊCH BẢN (STRICT SCRIPT IMMUTABILITY)
- TUYỆT ĐỐI KHÔNG tự động tạo ra, chỉnh sửa hoặc chạy các file kịch bản thực thi tự động (`run_all.sh`, `train_all.ps1`, `batch_runner.py`).
- Mọi lệnh huấn luyện hoặc đánh giá phải được cung cấp dưới dạng các dòng lệnh độc lập trong các khối code block để người dùng tự copy-paste vào terminal.

---

## 3. KHÁCH QUAN KHOA HỌC & CHỐNG TÂN BỐC (ANTI-SYCOPHANCY & ZERO HYPE)
- Nghiêm cấm sử dụng các tính từ phóng đại, tiếp thị ("đột phá mang tính nền tảng", "kỷ lục toàn cầu", "nghịch lý thế kỷ").
- Luôn giữ thái độ kiểm toán viên độc lập, nghiêm khắc, chỉ nói sự thật toán học và số liệu thực nghiệm.
- Bắt buộc phải chỉ rõ các thất bại kỹ thuật thực tế:
  * Điểm cân bằng sương mù nền (Phantom Background Fog chiếm ~64.4% mass trên nền trống).
  * Hiện tượng đóng băng bias của Softplus tại $b \approx -5.442$ do đạo hàm triệt tiêu.
  * Hiện tượng undercounting ở vùng mật độ cực cao (Dense MAE tăng lên 127.67 khi $d < 4\text{px}$).

---

## 4. CHÍNH SÁCH CAM KẾT VÀ ĐỒNG BỘ GIT LIÊN TỤC (CONTINUOUS COMMIT & PUSH)
- Mọi sửa đổi trên mã nguồn, cấu hình, hoặc tài liệu phải được commit và push ngay lập tức lên nhánh `RMR`.
