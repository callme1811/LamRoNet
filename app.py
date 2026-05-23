import os
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from utils import (
    BIN_DIR,
    TEMP_DIR,
    download_realesrgan_binary,
    get_executable_path,
    run_realesrgan,
)

st.set_page_config(
    page_title="ECG Enhancer - Real-ESRGAN",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "1.0.5"


def load_css(file_name: str):
    css_path = Path(__file__).resolve().parent / file_name
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("styles.css")

st.markdown(
    """
    <div class="heartbeat-container">
        <span class="heart-icon">❤️</span>
        <div style="flex-grow: 0;">
            <span class="badge-teal">AnimeVideoV3 Super-Resolution</span>
            <h1 style='margin: 0; padding-top: 4px; font-size: 2.2rem; background: linear-gradient(90deg, #FFFFFF 0%, #06B6D4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                ECG ENHANCER - TĂNG ĐỘ NÉT ĐIỆN TIM
            </h1>
        </div>
        <div class="pulse-line"></div>
    </div>
    <p style='color: #94A3B8; margin-top: -8px; margin-bottom: 24px; font-size: 1.05rem;'>
        Tăng độ nét ảnh điện tim bằng model <b>realesr-animevideov3</b>. Nếu Streamlit Cloud không hỗ trợ Vulkan, app tự chuyển sang chế độ tăng nét bằng Pillow.
    </p>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: #06B6D4 !important; font-size: 1.4rem;'>CẤU HÌNH XỬ LÝ</h2>
        <span style='color: #64748B; font-size: 0.85rem;'>Tối ưu cho Streamlit Cloud</span>
    </div>
    """,
    unsafe_allow_html=True,
)

model_choice = st.sidebar.selectbox(
    "Mô hình xử lý",
    options=[
        "realesr-animevideov3",
        "realesrgan-x4plus-anime",
        "realesrgan-x4plus",
    ],
    index=0,
)

if model_choice == "realesr-animevideov3":
    scale_options = [2]
    scale_help = "AnimeVideoV3 chạy ổn nhất ở 2x trên Cloud."
else:
    scale_options = [4]
    scale_help = "Model x4plus yêu cầu 4x."

scale_choice = st.sidebar.selectbox(
    "Tỷ lệ phóng đại",
    options=scale_options,
    index=0,
    help=scale_help,
)

tile_size = st.sidebar.slider(
    "Tile Size",
    min_value=50,
    max_value=400,
    value=100,
    step=50,
    help="Cloud nên để 50–100 để giảm lỗi bộ nhớ.",
)

auto_resize_cloud = st.sidebar.checkbox(
    "Tối ưu kích thước trên Cloud",
    value=(os.name != "nt"),
)

if auto_resize_cloud:
    cloud_resolution = st.sidebar.selectbox(
        "Độ rộng xử lý tối đa",
        options=[
            "360px",
            "480px",
            "640px",
        ],
        index=1,
    )
    max_cloud_width = int(cloud_resolution.replace("px", ""))
else:
    max_cloud_width = 1200

st.sidebar.markdown("<div class='divider' style='margin: 16px 0;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("### ⚡ Trạng Thái Hệ Thống")

if os.name == "nt":
    os_display = "Windows"
    mode_display = "Real-ESRGAN Vulkan"
else:
    os_display = "Linux / Streamlit Cloud" if sys.platform != "darwin" else "macOS"
    mode_display = "Real-ESRGAN nếu Vulkan chạy được, nếu không fallback Pillow"

st.sidebar.info(
    f"💻 Hệ điều hành: {os_display}\n"
    f"⚙️ Chế độ: {mode_display}\n"
    f"📦 Phiên bản: {APP_VERSION}\n"
    f"📁 BIN: {BIN_DIR}"
)

if get_executable_path():
    st.sidebar.success("✅ Real-ESRGAN binary: Sẵn sàng")
else:
    st.sidebar.warning("⚠️ Binary sẽ được tải tự động khi xử lý")

col_upload, col_view = st.columns([1, 2.2], gap="large")

with col_upload:
    st.markdown(
        """
        <div class="clinical-card">
            <h3 style='margin: 0 0 12px 0; font-size: 1.15rem; color: #38BDF8 !important;'>📁 TẢI LÊN ẢNH ĐIỆN TIM</h3>
            <p style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 16px;'>Tải lên ảnh PNG, JPG hoặc JPEG.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Chọn ảnh",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        orig_image = Image.open(uploaded_file).convert("RGB")

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        if auto_resize_cloud and orig_image.width > max_cloud_width:
            ratio = max_cloud_width / float(orig_image.width)
            new_height = int(orig_image.height * ratio)
            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
            processed_image = orig_image.resize((max_cloud_width, new_height), resample_filter)
        else:
            processed_image = orig_image

        input_img_path = TEMP_DIR / "input_temp.png"
        processed_image.save(input_img_path, "PNG")

        st.markdown(
            f"""
            <div class="clinical-card" style="margin-top: 16px; padding: 16px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; color: #E2E8F0 !important;">Thông số ảnh:</h4>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #94A3B8;"><b>Gốc:</b> {orig_image.width} x {orig_image.height}px</p>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #94A3B8;"><b>Xử lý:</b> {processed_image.width} x {processed_image.height}px</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        run_btn = st.button("🚀 BẮT ĐẦU TĂNG NÉT")
    else:
        st.info("👈 Tải ảnh lên để bắt đầu.")
        run_btn = False

with col_view:
    if uploaded_file is not None:
        output_img_path = TEMP_DIR / "enhanced_temp.png"

        if run_btn:
            progress_status = st.empty()
            progress_bar = st.progress(0.0)

            def update_progress(msg, progress_val):
                progress_status.markdown(
                    f"<p style='color: #22D3EE; font-weight: 500;'>⏳ {msg}</p>",
                    unsafe_allow_html=True,
                )
                progress_bar.progress(progress_val)

            try:
                t_start = time.time()

                update_progress("Đang kiểm tra/tải Real-ESRGAN...", 0.15)
                download_realesrgan_binary(status_callback=update_progress)

                update_progress("Đang xử lý bằng animevideov3...", 0.65)
                result_mode = run_realesrgan(
                    input_path=str(input_img_path),
                    output_path=str(output_img_path),
                    model_name=model_choice,
                    tile_size=tile_size,
                    scale=scale_choice,
                )

                duration = time.time() - t_start

                progress_status.empty()
                progress_bar.empty()

                st.session_state["processed_file"] = {
                    "input": str(input_img_path),
                    "output": str(output_img_path),
                    "duration": duration,
                    "model": model_choice,
                    "scale": scale_choice,
                    "mode": result_mode,
                }

                st.success(f"✨ Xử lý xong trong {duration:.2f} giây. Chế độ: {result_mode}")

            except Exception as e:
                progress_status.empty()
                progress_bar.empty()
                st.error(f"❌ Có lỗi xảy ra trong quá trình xử lý: {str(e)}")

        if (
            "processed_file" in st.session_state
            and os.path.exists(st.session_state["processed_file"]["output"])
        ):
            p_data = st.session_state["processed_file"]
            orig_loaded = Image.open(p_data["input"])
            enhanced_img = Image.open(p_data["output"])

            st.markdown("### 🔍 So Sánh Kết Quả")

            tab_overview, tab_zoom = st.tabs(["📊 Xem Toàn Cảnh", "🔎 Soi Chi Tiết"])

            with tab_overview:
                col_orig, col_enh = st.columns(2)

                with col_orig:
                    st.markdown(
                        "<p style='text-align: center; color: #EF4444; font-weight: 600;'>ẢNH TRƯỚC</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(orig_loaded, use_container_width=True)

                with col_enh:
                    st.markdown(
                        "<p style='text-align: center; color: #10B981; font-weight: 600;'>ẢNH SAU</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(enhanced_img, use_container_width=True)

            with tab_zoom:
                w, h = orig_loaded.size
                w_enh, h_enh = enhanced_img.size

                crop_x = st.slider("Vị trí X (%)", 0, 100, 50, 1)
                crop_y = st.slider("Vị trí Y (%)", 0, 100, 50, 1)
                crop_w_pct = st.slider("Kích thước vùng soi (%)", 5, 40, 15, 1)

                crop_w = max(1, int(w * crop_w_pct / 100))
                crop_h = max(1, int(h * crop_w_pct / 100))

                cx = int(w * crop_x / 100)
                cy = int(h * crop_y / 100)

                x1 = max(0, cx - crop_w // 2)
                y1 = max(0, cy - crop_h // 2)
                x2 = min(w, x1 + crop_w)
                y2 = min(h, y1 + crop_h)

                sx = w_enh / w
                sy = h_enh / h

                ex1 = int(x1 * sx)
                ey1 = int(y1 * sy)
                ex2 = int(x2 * sx)
                ey2 = int(y2 * sy)

                cropped_orig = orig_loaded.crop((x1, y1, x2, y2))
                cropped_enh = enhanced_img.crop((ex1, ey1, ex2, ey2))

                col_crop_o, col_crop_e = st.columns(2)

                with col_crop_o:
                    st.markdown(
                        "<p style='text-align: center; color: #EF4444;'>VÙNG SOI TRƯỚC</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(cropped_orig, use_container_width=True)

                with col_crop_e:
                    st.markdown(
                        "<p style='text-align: center; color: #10B981;'>VÙNG SOI SAU</p>",
                        unsafe_allow_html=True,
                    )
                    st.image(cropped_enh, use_container_width=True)

            st.markdown("### 💾 TẢI XUỐNG")

            with open(p_data["output"], "rb") as f:
                img_bytes = f.read()

            st.download_button(
                label="📥 TẢI XUỐNG ẢNH ECG SẮC NÉT",
                data=img_bytes,
                file_name=f"ECG_Enhanced_{Path(uploaded_file.name).stem}.png",
                mime="image/png",
            )

        else:
            st.info("👆 Bấm nút xử lý để xem kết quả.")