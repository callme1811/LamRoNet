﻿import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from src.homography import four_point_transform
from src.segment_waveform import enhance_ecg_image

try:
    from src.realesrgan_enhance import enhance_with_realesrgan
    HAS_REALESRGAN = True
except Exception:
    HAS_REALESRGAN = False


TEMP_DIR = Path(__file__).resolve().parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PATH = Path(__file__).resolve().parent / "weights" / "RealESRGAN_x4plus.pth"


st.set_page_config(
    page_title="LamRoNet - ECG Rectifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


def auto_rotate_image(pil_img):
    w, h = pil_img.size

    # Không xoay ECG strip rất dài
    if w > h and w / h < 1.8:
        return pil_img.rotate(90, expand=True)

    return pil_img


def draw_corners_preview(image, corners):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    w, h = image.size
    line_w = max(3, int(min(w, h) * 0.005))
    circle_r = max(5, int(min(w, h) * 0.012))

    draw.polygon(corners, outline="#EF4444", width=line_w)

    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(corners):
        x, y = pt
        draw.ellipse(
            [x - circle_r, y - circle_r, x + circle_r, y + circle_r],
            fill="#22D3EE",
            outline="#0EA5E9",
            width=2,
        )
        draw.text((x + circle_r + 2, y - circle_r - 2), labels[i], fill="#FFFFFF")

    return draw_img


def save_rgb(path, image_rgb):
    Image.fromarray(image_rgb.astype(np.uint8)).save(path)


st.title("⚡ LamRoNet - ECG Paper Flattening & Enhancement")

st.markdown(
    """
Upload ảnh điện tim → chỉnh 4 góc → làm phẳng bằng homography → làm rõ nét bằng OpenCV hoặc Real-ESRGAN x4.
"""
)

st.sidebar.header("⚙️ Cấu hình")

rotate_mode = st.sidebar.selectbox(
    "Xoay ảnh",
    [
        "Tự động",
        "Không xoay",
        "Xoay 90 độ phải",
        "Xoay 90 độ trái",
        "Xoay 180 độ",
    ],
    index=0,
)

use_realesrgan = st.sidebar.checkbox(
    "Dùng Real-ESRGAN x4",
    value=False,
    help="Tăng độ phân giải 4 lần. Chạy chậm hơn nhưng chữ nhỏ và đường sóng có thể rõ hơn.",
)

enhance_strength = st.sidebar.slider(
    "Độ làm nét OpenCV",
    min_value=1.0,
    max_value=2.5,
    value=1.4,
    step=0.1,
)

denoise_strength = st.sidebar.slider(
    "Khử nhiễu OpenCV",
    min_value=0,
    max_value=15,
    value=3,
    step=1,
)

uploaded_file = st.file_uploader(
    "Upload ảnh tờ điện tim ECG",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Hãy upload ảnh ECG để bắt đầu.")
    st.stop()


orig_img = Image.open(uploaded_file).convert("RGB")

if rotate_mode == "Tự động":
    orig_img = auto_rotate_image(orig_img)
elif rotate_mode == "Xoay 90 độ phải":
    orig_img = orig_img.rotate(-90, expand=True)
elif rotate_mode == "Xoay 90 độ trái":
    orig_img = orig_img.rotate(90, expand=True)
elif rotate_mode == "Xoay 180 độ":
    orig_img = orig_img.rotate(180, expand=True)

W_orig, H_orig = orig_img.size
st.markdown(f"**Kích thước ảnh sau xoay:** {W_orig} x {H_orig}px")

if use_realesrgan:
    if not HAS_REALESRGAN:
        st.warning("Chưa import được Real-ESRGAN. App sẽ dùng OpenCV enhancement.")
    elif not WEIGHT_PATH.exists():
        st.warning(f"Không thấy weights: {WEIGHT_PATH}. App sẽ dùng OpenCV enhancement.")

col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.subheader("🎯 Chỉnh 4 góc vùng ECG")

    st.caption("Kéo slider để khung đỏ ôm đúng vùng giấy ECG cần làm phẳng.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Góc Trên-Trái (TL)**")
        tl_x = st.slider("TL X %", 0, 100, st.session_state.get("tl_x", 0))
        tl_y = st.slider("TL Y %", 0, 100, st.session_state.get("tl_y", 0))

        st.markdown("**Góc Dưới-Trái (BL)**")
        bl_x = st.slider("BL X %", 0, 100, st.session_state.get("bl_x", 0))
        bl_y = st.slider("BL Y %", 0, 100, st.session_state.get("bl_y", 100))

    with col_b:
        st.markdown("**Góc Trên-Phải (TR)**")
        tr_x = st.slider("TR X %", 0, 100, st.session_state.get("tr_x", 100))
        tr_y = st.slider("TR Y %", 0, 100, st.session_state.get("tr_y", 0))

        st.markdown("**Góc Dưới-Phải (BR)**")
        br_x = st.slider("BR X %", 0, 100, st.session_state.get("br_x", 100))
        br_y = st.slider("BR Y %", 0, 100, st.session_state.get("br_y", 100))

    for key, val in {
        "tl_x": tl_x,
        "tl_y": tl_y,
        "tr_x": tr_x,
        "tr_y": tr_y,
        "br_x": br_x,
        "br_y": br_y,
        "bl_x": bl_x,
        "bl_y": bl_y,
    }.items():
        st.session_state[key] = val

    pts_pixels = [
        (int((tl_x / 100) * W_orig), int((tl_y / 100) * H_orig)),
        (int((tr_x / 100) * W_orig), int((tr_y / 100) * H_orig)),
        (int((br_x / 100) * W_orig), int((br_y / 100) * H_orig)),
        (int((bl_x / 100) * W_orig), int((bl_y / 100) * H_orig)),
    ]

    run_btn = st.button("🚀 Làm phẳng & làm rõ nét", type="primary")

with col_right:
    st.subheader("🔎 Xem trước vùng chọn")

    preview_img = draw_corners_preview(orig_img, pts_pixels)
    show_image(preview_img, caption="Khung đỏ là vùng sẽ được kéo phẳng.")

    if run_btn:
        try:
            start = time.time()

            image_np = np.array(orig_img)
            pts_np = np.array(pts_pixels, dtype=np.float32)

            with st.spinner("Đang làm phẳng ảnh ECG bằng homography..."):
                flat_image = four_point_transform(image_np, pts_np)

            with st.spinner("Đang khử nhiễu và làm nét bằng OpenCV..."):
                enhanced_image = enhance_ecg_image(
                    flat_image,
                    denoise_strength=denoise_strength,
                    sharp_strength=enhance_strength,
                )

            used_realesrgan = False

            if use_realesrgan and HAS_REALESRGAN and WEIGHT_PATH.exists():
                try:
                    with st.spinner("Đang tăng độ phân giải bằng Real-ESRGAN x4..."):
                        enhanced_image = enhance_with_realesrgan(enhanced_image)
                    used_realesrgan = True
                except Exception as esr_error:
                    st.warning(f"Real-ESRGAN lỗi, đã fallback về OpenCV: {esr_error}")

            flat_path = TEMP_DIR / "ecg_flat.png"
            enhanced_path = TEMP_DIR / "ecg_enhanced.png"

            save_rgb(flat_path, flat_image)
            save_rgb(enhanced_path, enhanced_image)

            duration = time.time() - start

            if used_realesrgan:
                st.success(f"Đã xử lý xong bằng OpenCV + Real-ESRGAN x4 trong {duration:.2f} giây.")
            else:
                st.success(f"Đã xử lý xong bằng OpenCV trong {duration:.2f} giây.")

            st.subheader("📊 Kết quả")

            col_1, col_2 = st.columns(2)

            with col_1:
                st.markdown("**Ảnh đã làm phẳng**")
                show_image(str(flat_path))

            with col_2:
                st.markdown("**Ảnh đã làm rõ nét**")
                show_image(str(enhanced_path))

            with open(enhanced_path, "rb") as f:
                st.download_button(
                    label="📥 Tải ảnh ECG đã làm rõ nét",
                    data=f.read(),
                    file_name="ecg_enhanced.png",
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"Lỗi xử lý ảnh: {e}")