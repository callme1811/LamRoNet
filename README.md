# AI ECG Enhancer - Công Cụ Làm Rõ Nét Bản Quét Điện Tim Bằng AI ⚡🩺

Ứng dụng web được xây dựng trên nền tảng **Streamlit** kết hợp sức mạnh của mô hình học sâu **Real-ESRGAN** (phiên bản tối ưu hóa di động **NCNN Vulkan**) để làm sắc nét, phóng to và khử nhiễu các bản quét hoặc ảnh chụp giấy điện tim (ECG) chất lượng thấp.

Ứng dụng được thiết kế hoàn toàn bằng **Trí tuệ nhân tạo (Pure AI Pipeline)**, sử dụng thư viện **Pillow** tiêu chuẩn và gọi trực tiếp bộ xử lý Real-ESRGAN qua GPU, giúp đạt hiệu năng cực cao mà không cần cài đặt các thư viện nặng nề như PyTorch hay OpenCV.

---

## ✨ Các Tính Năng Nổi Bật

1. **Siêu phân giải AI 4x (AI Super-Resolution):**
   * Sử dụng mô hình học sâu hàng đầu để tái cấu trúc các đường sóng điện tim bị mờ, đứt đoạn, giúp các bác sĩ dễ dàng đọc các chuyển đạo (leads).
   * Khôi phục độ rõ nét của chữ số, ký hiệu và lưới giấy kỹ thuật trên bản ghi.
2. **Lựa chọn mô hình linh hoạt (Model Selection):**
   * **`realesrgan-x4plus-anime` (Khuyên dùng):** Tối ưu hóa đặc biệt cho các nét vẽ và đường sóng mảnh, giúp khử nhiễu hạt giấy cực tốt và làm mịn đường cong sóng điện tim.
   * **`realesrgan-x4plus`:** Giữ nguyên vẹn và tái tạo chi tiết tối đa cho văn bản và các vân gai phức tạp.
   * **`realesr-animevideov3`:** Phiên bản siêu nhanh và nhẹ cho các máy cấu hình thấp.
3. **Kính soi chi tiết cục bộ (Pixel-Perfect Crop Inspector):**
   * Bản quét ECG thường rất rộng, hiển thị toàn bộ sẽ làm ảnh bị thu nhỏ trên trình duyệt.
   * Kính soi chi tiết tích hợp cho phép chọn bất kỳ vùng chuyển đạo nào (X, Y và kích thước) để zoom và so sánh trực tiếp chất lượng Trước & Sau siêu phân giải ở mức độ điểm ảnh (pixel).
4. **Tự động tối ưu hóa tài nguyên:**
   * Hỗ trợ cơ chế phân mảnh ảnh (**Tiling**) tùy chỉnh, giúp ứng dụng hoạt động mượt mà trên các card đồ họa phổ thông (VRAM 4GB) mà không bị lỗi tràn bộ nhớ (Out of Memory).
5. **Giao diện Y tế Cao cấp (Clinical Dark Dashboard):**
   * Giao diện chủ đề tối y tế sang trọng với các hiệu ứng kính mờ (glassmorphism), biểu đồ nhịp tim động (heartbeat pulse) và bảng đo chỉ số thời gian thực.
   * Nút tải xuống ảnh sắc nét độ phân giải cao chỉ với một click.

---

## 🛠️ Yêu Cầu Hệ Thống

* **Hệ điều hành:** Windows 10 / 11 (64-bit).
* **Python:** Phiên bản 3.9 trở lên (đã test hoạt động tốt nhất trên Python 3.10).
* **Card đồ họa (GPU):** Hỗ trợ Vulkan (NVIDIA, AMD hoặc Intel). Hệ thống sẽ tự động sử dụng GPU để đạt tốc độ xử lý nhanh nhất (từ 1 - 3 giây).

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### Bước 1: Tải mã nguồn về máy
```bash
git clone https://github.com/callme1811/LamRoNet.git
cd LamRoNet
```

### Bước 2: Cài đặt các thư viện Python cần thiết
Mở Terminal / PowerShell tại thư mục dự án và chạy lệnh sau:
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Sau khi chạy lệnh, trình duyệt web sẽ tự động mở trang ứng dụng tại địa chỉ: `http://localhost:8501`.

*Lưu ý: Trong lần đầu tiên bấm "BẮT ĐẦU TĂNG NÉT BẰNG AI", ứng dụng sẽ tự động tải bộ chạy Real-ESRGAN NCNN Vulkan chính thức (~43MB) từ GitHub về thư mục `bin/` của dự án. Bạn không cần phải cấu hình hay tải thủ công.*

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
LamRoNet/
│
├── bin/                 # Thư mục chứa bộ chạy Real-ESRGAN (Được tự động tải về khi chạy app)
├── temp/                # Thư mục chứa ảnh tạm thời khi xử lý
├── app.py               # Giao diện chính của Streamlit Dashboard
├── utils.py             # Bộ điều khiển tải mô hình & chạy subprocess Real-ESRGAN
├── styles.css           # Tệp thiết kế giao diện lâm sàng tối cao cấp
├── requirements.txt     # Danh sách thư viện Python cần thiết
└── README.md            # Tài liệu hướng dẫn sử dụng (tệp này)
```

---

## 🛡️ Bản Quyền & Tài Liệu Tham Khảo

* Dự án sử dụng mô hình học sâu **Real-ESRGAN** được phát triển bởi [xinntao](https://github.com/xinntao/Real-ESRGAN).
* Bản quyền bộ chạy di động NCNN thuộc về nhóm phát triển Tencent và được phân phối theo giấy phép BSD-3-Clause.
