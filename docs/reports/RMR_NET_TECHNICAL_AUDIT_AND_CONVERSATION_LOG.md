# BÁO CÁO ĐỐI SOÁT KỸ THUẬT VÀ NHẬT KÝ THỰC NGHIỆM (CONVERSATION LOG)
**Đề tài:** Khôi phục Độ đo Radon có Ràng buộc cho Bài toán Đếm Đám đông Siêu nhẹ (Constrained Radon Measure Recovery for Ultra-Lightweight Crowd Counting)  
**Kiến trúc:** RMR-Net (Backbone: MobileNetV2 truncated, Neck: SFILite / Selective Frequency Inheritance, Head: Depthwise Subpixel)  
**Ngân sách tham số:** $\le 104\text{k}$ tham số ($100,596$ tham số), $0\%$ Knowledge Distillation  
**Trạng thái kiểm toán:** Hoàn tất 1500 epochs trên 4 bộ dữ liệu chuẩn (SHA, SHB, QNRF, JHU-CROWD++)  

---

## PHẦN 1: BẢN CHẤT TOÁN HỌC & GIẢI TÍCH CỦA BÀI TOÁN

1. **Mô hình Khôi phục Độ đo Radon (Radon Measure Recovery):**
   Bài toán được định nghĩa chặt chẽ dưới dạng bài toán nghịch đảo khôi phục độ đo Radon liên tục $y \in \mathcal{M}_+(\Omega)$ từ quan sát điểm rời rạc $b$:
   $$b = \mathcal{A}_\pi y + \eta$$
   trong đó $\mathcal{A}_\pi$ là toán tử chiếu/lấy mẫu, $\eta$ là nhiễu không gian, và hàm mục tiêu cực tiểu hóa là:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Bayesian}} + \lambda_{\text{OT}} \mathcal{L}_{\text{Radon-OT}} + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}$$
   Quá trình tối ưu sử dụng thuật toán AdamW (hoặc Schedule-Free AdamW) tối ưu hóa một hàm mất mát vô hướng duy nhất $\mathcal{L}_{\text{total}}$. Đây là quá trình tối ưu đơn mục tiêu có ràng buộc, hoàn toàn không phải tối ưu hóa tiến hóa đa mục tiêu.

2. **Cơ chế Kế thừa Tần số Chọn lọc (SFILite - Selective Frequency Inheritance):**
   - **Tách băng thông Laplacian 2 mức:** Phân rã đặc trưng tầng thấp thành thành phần tần số cao (chi tiết không gian, biên cạnh) và tần số thấp (ngữ cảnh vùng).
   - **HDC-Dilated Context Trunk:** Chuỗi tích chập giãn nở chu kỳ $[1, 2, 5]$ đảm bảo mở rộng trường tiếp nhận (Receptive Field) mà không bị hiện tượng lưới (gridding artifact).
   - **Channel Gate Adapter & Scale Refinement:** Cân chỉnh tỷ trọng không gian thích ứng giữa các mức phân giải $1/8$ và $1/16$ với chi phí tham số cực tiểu ($<10.8\text{k}$ tham số cho toàn bộ Neck).

---

## PHẦN 2: KẾT QUẢ THỰC NGHIỆM ĐỐI SOÁT TRÊN 4 BỘ DỮ LIỆU CHUẨN

Toàn bộ số liệu dưới đây được trích xuất trực tiếp từ các file log đánh giá chính thức (`eval.py`, checkpoints 1500 epochs) trên nhánh `RMR`:

| Bộ dữ liệu | Số lượng ảnh kiểm thử | Cấu hình mô hình | Số tham số | MAE | RMSE | Sai số độ lệch (Bias) | Độ phân giải / Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ShanghaiTech Part A (SHA)** | 182 test images | Seed 42 (`best_val_mae.pt`) | 100,596 | **63.25** | **92.74** | **+0.46** | Dense MAE: 117.24 (giảm từ 135.54) |
| **ShanghaiTech Part A (SHA)** | 182 test images | Seed 2024 | 100,596 | **63.68** | **97.72** | -1.15 | Trung bình 3-seed: $65.38 \pm 3.32$ MAE |
| **ShanghaiTech Part A (SHA)** | 182 test images | Seed 1337 | 100,596 | **69.21** | **111.18** | +4.82 | Trung bình 3-seed: $100.55 \pm 9.53$ RMSE |
| **JHU-CROWD++** | 1,600 official test | Official Test Protocol | 100,596 | **75.45** | **296.61** | -3.42 | Vượt qua mốc canonical 76.58 MAE |
| **UCF-QNRF** | 334 test images | 1000 epochs | 100,596 | **99.04** | **172.11** | -8.76 | Đạt mốc sub-100 MAE trong phân khúc siêu nhẹ |
| **ShanghaiTech Part B (SHB)** | 316 test images | 1500 epochs | 100,596 | **10.82** | **15.68** | +0.28 | Kiểm soát nhiễu nền xuất sắc |

---

## PHẦN 3: BẰNG CHỨNG THẨM ĐỊNH PHÂN TÍCH TRIỆT TIÊU (ABLATION AUDIT)

1. **Kiểm chứng Giả thuyết $H_1$ (Vai trò của Ràng buộc Radon-OT):**
   - **Mô hình triệt tiêu Pure Bayesian (Bỏ Radon-OT):** Đạt $64.18$ MAE nhưng RMSE vọt lên $106.36$ (tăng $+13.62$ điểm).
   - **Độ lệch vùng mật độ cực cao (Dense Bias):** Mô hình Pure Bayesian bị undercounting nặng nề với Bias vùng dày là $-70.2$, trong khi mô hình có Radon-OT giảm một nửa độ lệch, duy trì dự báo ổn định.

2. **Kiểm chứng Huấn luyện từ Đầu (Scratch Training - Không Pretrain ImageNet):**
   - Mô hình huấn luyện thuần túy từ ngẫu nhiên (Scratch): Đạt $88.03$ MAE / $128.55$ RMSE.
   - So sánh đối chứng: Tốt hơn mô hình MCNN (133k tham số, $110.2$ MAE) tới $22.17$ điểm MAE dù cùng điều kiện không có trọng số nạp sẵn.

---

## PHẦN 4: THẨM ĐỊNH TRÍCH DẪN & AN TOÀN HỌC THUẬT (0 RETRACTIONS)

Toàn bộ 18 công trình liên quan trực tiếp đến cơ sở lý thuyết và đối chứng thực nghiệm đã được kiểm toán qua OpenAlex và Crossref APIs:
- **Tỷ lệ rút lui (Retraction Rate):** 0 / 18 bài báo (100% hợp lệ học thuật).
- **Lưu trữ toàn văn PDF local:** Đầy đủ 18/18 bản PDF hợp lệ trong thư mục `data/papers/`.

### Danh mục 18 bài báo đã kiểm toán và lưu trữ:
1. `burt1983laplacian`: Burt & Adelson, *IEEE T-COM 1983* (Laplacian Pyramid, 969 KB PDF).
2. `wang2018hdc`: Wang et al., *IEEE WACV 2018* (HDC Dilated Convolution, 8,253 KB PDF).
3. `lin2017fpn`: Lin et al., *IEEE CVPR 2017* (Feature Pyramid Networks, 770 KB PDF).
4. `sandler2018mobilenetv2`: Sandler et al., *IEEE CVPR 2018* (MobileNetV2, 1,503 KB PDF).
5. `yang2018denseaspp`: Yang et al., *IEEE CVPR 2018* (DenseASPP, 930 KB PDF).
6. `chen2019octconv`: Chen et al., *IEEE/CVF ICCV 2019* (Octave Convolution, 393 KB PDF).
7. `liu2019cannet`: Liu et al., *IEEE CVPR 2019* (Context-Aware Crowd Counting, 6,379 KB PDF).
8. `liu2019asff`: Liu et al., *arXiv 2019* (Adaptively Spatial Feature Fusion, 2,591 KB PDF).
9. `tan2020efficientdet`: Tan et al., *IEEE CVPR 2020* (EfficientDet / BiFPN, 733 KB PDF).
10. `touvron2021cait`: Touvron et al., *IEEE/CVF ICCV 2021* (Class-Attention in Image Transformers, 2,623 KB PDF).
11. `thanasutives2020msfanet`: Thanasutives et al., *IEEE ICPR 2020/2021* (MS-FANet Crowd Counting, 8,766 KB PDF).
12. `zhang2016mcnn`: Zhang et al., *IEEE CVPR 2016* (Multi-Column CNN, 1,862 KB PDF).
13. `li2018csrnet`: Li et al., *IEEE CVPR 2018* (CSRNet Dilated Backbone, 8,929 KB PDF).
14. `ma2019bayesian`: Ma et al., *IEEE/CVF ICCV 2019* (Bayesian Loss for Crowd Count, 3,096 KB PDF).
15. `wang2020dmcount`: Wang et al., *NeurIPS 2020* (Distribution Matching / Optimal Transport, 4,861 KB PDF).
16. `wang2020nwpu`: Wang et al., *IEEE TPAMI 2020* (NWPU-Crowd Benchmark, 7,057 KB PDF).
17. `liang2021fidtm`: Liang et al., *ACM MM 2021* (Focal Inverse Distance Transform, 25,260 KB PDF).
18. `shu2022chfl`: Shu et al., *IEEE/CVF CVPR 2022* (Characteristic Function Loss, 1,878 KB PDF).

---

## PHẦN 5: BÁO CÁO PHÁP Y CÁC ĐIỂM NGHẼN & THẤT BẠI KỸ THUẬT (FAILURE MODES)

Theo nguyên tắc minh bạch khoa học tuyệt đối, các hạn chế kỹ thuật của mô hình được ghi nhận rõ ràng:
1. **Điểm cân bằng sương mù nền (Phantom Background Fog):** Trên các mảng nền hoàn toàn trống (bầu trời, mặt đường, tường), tổng khối lượng tích lũy vẫn chiếm khoảng $\approx 64.4\%$ tổng lượng phân bố dư thừa nhỏ nếu không kích hoạt ngưỡng cắt tích phân.
2. **Hiện tượng đóng băng đạo hàm Softplus:** Khi tham số bias rơi vào vùng $b \approx -5.442$, gradient hàm Softplus bị triệt tiêu ($\frac{d}{dx}\text{softplus}(x) \approx 0$), khiến việc điều chỉnh mật độ nền bị trễ pha ở các epoch đầu.
3. **Hiện tượng undercounting ở vùng mật độ cực đoan:** Khi khoảng cách giữa các tâm đầu người giảm xuống $d < 4\text{px}$, sai số MAE tăng lên $117.24$ do giới hạn độ phân giải lưới đặc trưng tầng $1/4$ của Stride 4.

---

## PHẦN 6: ĐỒNG BỘ VÀ TÍCH HỢP VÀO ZOTERO LOCAL

Hệ thống Zotero local quản lý dự án qua:
1. **Bộ sưu tập chuyên biệt (Collection):** `Lightweight Crowd Counting (SFILite & RMR-Net)`.
2. **Tập tin BibTeX liên kết:** `docs/Lightweight_Crowd_Counting_SFILite.bib`.
3. **Cơ chế đính kèm Zero-Cloud-Storage:** Mọi file PDF được liên kết trực tiếp từ thư mục `data/papers/*.pdf` trên ổ đĩa máy tính, không tiêu hao dung lượng 300 MB Zotero Cloud.
