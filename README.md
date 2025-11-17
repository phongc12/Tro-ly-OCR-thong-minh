# 🧠 Trợ lý OCR thông minh

Ứng dụng Streamlit cho phép bạn tải ảnh hoá đơn, tài liệu hoặc ảnh chụp bất kỳ để trích xuất văn bản bằng EasyOCR. App hỗ trợ lọc trước ảnh, tải văn bản về, cũng như hiển thị kết quả với khung bao quanh từng dòng chữ.

## Yêu cầu hệ thống

- Python 3.9+
- Kết nối internet để cài đặt thư viện (EasyOCR cần Torch)

## Cách chạy

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Ứng dụng sẽ mở trên `http://localhost:8501`.

## Tính năng chính

- Nhận dạng đa ngôn ngữ (Tiếng Việt, Tiếng Anh — có thể mở rộng).
- Các bộ lọc ảnh: grayscale, đảo màu, tăng tương phản.
- Hiển thị song song ảnh gốc, ảnh đã xử lý, ảnh có khung OCR.
- Bảng chi tiết văn bản + độ tin cậy, tải file `.txt`.

## Tuỳ biến

- Mở rộng danh sách ngôn ngữ trong hàm `build_sidebar`.
- Thêm bước xử lý ảnh (ví dụ làm mờ, sharpen) trong `preprocess_image`.
- Tích hợp thêm hậu xử lý như tóm tắt hoặc dịch với các API tuỳ nhu cầu.

