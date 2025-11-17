import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_reader(languages: Tuple[str, ...]):
    """Lazily create the EasyOCR reader to avoid re-loading on each change."""
    try:
        import easyocr  # Imported lazily so Streamlit can cache the reader correctly.
    except ModuleNotFoundError:
        st.error(
            "Không tìm thấy thư viện `easyocr`. Vui lòng chạy `pip install -r requirements.txt` "
            "hoặc cài thủ công bằng `pip install easyocr` rồi tải lại trang."
        )
        st.stop()

    return easyocr.Reader(list(languages), gpu=False)


@dataclass
class OcrResult:
    bbox: List[Tuple[int, int]]
    text: str
    confidence: float


def preprocess_image(image: Image.Image, grayscale: bool, invert: bool, contrast: float) -> Image.Image:
    processed = image
    if grayscale:
        processed = ImageOps.grayscale(processed)
    if invert:
        processed = ImageOps.invert(processed)
    if contrast != 1.0:
        processed = ImageEnhance.Contrast(processed).enhance(contrast)
    return processed


def perform_ocr(image: Image.Image, languages: Tuple[str, ...]) -> List[OcrResult]:
    reader = load_reader(languages)
    np_img = np.array(image)
    raw_results = reader.readtext(np_img)
    return [
        OcrResult(
            bbox=[tuple(map(int, point)) for point in bbox],
            text=text.strip(),
            confidence=float(confidence),
        )
        for bbox, text, confidence in raw_results
    ]


def draw_boxes(image: Image.Image, results: List[OcrResult]) -> Image.Image:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    for result in results:
        draw.polygon(result.bbox, outline="lime")
        if result.text:
            x, y = result.bbox[0]
            draw.text((x, y - 12), result.text, fill="yellow")
    return preview


def build_sidebar() -> Tuple[Tuple[str, ...], bool, bool, float]:
    st.sidebar.header("⚙️ Cấu hình OCR")
    languages = st.sidebar.multiselect(
        "Ngôn ngữ nhận dạng",
        options=["vi", "en"],
        format_func=lambda code: {"vi": "Tiếng Việt", "en": "English"}[code],
        default=["vi", "en"],
        max_selections=3,
    )
    grayscale = st.sidebar.toggle("Chuyển ảnh sang grayscale", value=True)
    invert = st.sidebar.toggle("Đảo màu (đối với nền tối, chữ sáng)")
    contrast = st.sidebar.slider("Tăng độ tương phản", min_value=0.5, max_value=3.0, step=0.1, value=1.4)
    return tuple(languages) or ("vi",), grayscale, invert, contrast


def text_summary(results: List[OcrResult]) -> str:
    text = "\n".join(r.text for r in results if r.text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def download_text_button(text: str, filename: str = "ocr_output.txt"):
    st.download_button("⬇️ Tải văn bản", data=text, file_name=filename, mime="text/plain")


def main():
    st.set_page_config(page_title="Trợ lý OCR thông minh", page_icon="🧠", layout="wide")
    st.title("🧠 Trợ lý OCR thông minh")
    st.write(
        "Tải lên hình ảnh chứa văn bản (hoá đơn, tài liệu, ảnh chụp, v.v.) "
        "để hệ thống tự động nhận dạng, phân tích và trích xuất nội dung."
    )

    languages, grayscale, invert, contrast = build_sidebar()

    uploaded_file = st.file_uploader("Chọn ảnh PNG, JPG hoặc JPEG", type=["png", "jpg", "jpeg"])
    if not uploaded_file:
        st.info("👆 Hãy tải ảnh lên để bắt đầu.")
        return

    original_image = Image.open(uploaded_file)
    processed_image = preprocess_image(original_image, grayscale, invert, contrast)

    with st.spinner("Đang xử lý OCR ..."):
        results = perform_ocr(processed_image, languages)

    if not results:
        st.warning("Không tìm thấy văn bản trong ảnh. Thử điều chỉnh bộ lọc hoặc chọn ảnh khác.")
        return

    text = text_summary(results)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh gốc")
        st.image(original_image, use_column_width=True)
        st.subheader("Ảnh sau xử lý")
        st.image(processed_image, use_column_width=True)

    with col2:
        st.subheader("Ảnh có khung OCR")
        st.image(draw_boxes(original_image, results), use_column_width=True)
        st.subheader("Văn bản trích xuất")
        st.text_area("Kết quả OCR", value=text, height=300)
        download_text_button(text)

    st.subheader("Chi tiết từng dòng")
    rows = [
        {
            "Văn bản": result.text,
            "Độ tin cậy": f"{result.confidence * 100:.1f}%",
            "tọa độ": result.bbox,
        }
        for result in results
    ]
    st.dataframe(rows)

    st.caption(
        "Ứng dụng sử dụng EasyOCR (PyTorch) và Streamlit. "
        "Nếu cần hỗ trợ GPU hoặc ngôn ngữ khác, mở rộng danh sách ngôn ngữ trong thanh bên."
    )


if __name__ == "__main__":
    main()

