import streamlit as st
import time
import os
import sys
from pathlib import Path
from PIL import Image
from utils import download_realesrgan_binary, run_realesrgan, TEMP_DIR, get_executable_path, BIN_DIR

st.set_page_config(
    page_title="ECG Enhancer - Real-ESRGAN",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_VERSION = "1.0.4"


def load_css(file_name):
    css_path = Path(__file__).resolve().parent / file_name
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("styles.css")

st.markdown("""
    <div class="heartbeat-container">
        <span class="heart-icon">❤️</span>
        <div style="flex-grow: 0;">
            <span class="badge-teal">Super-Resolution Pipeline</span>
            <h1 style='margin: 0; padding-top: 4px; font-size: 2.2rem; background: linear-gradient(90deg, #FFFFFF 0%, #06B6D4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                ECG ENHANCER - TĂNG ĐỘ NÉT ĐIỆN TIM
            </h1>
        </div>
        <div class="pulse-line"></div>
    </div>
    <p style='color: #94A3B8; margin-top: -8px; margin-bottom: 24px; font-size: 1.05rem;'>
        Khôi phục, khử nhiễu và làm sắc nét các bản quét điện tim (ECG) mờ nhạt bằng mô hình Real-ESRGAN.
    </p>
    <div class="divider"></div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: #06B6D4 !important; font-size: 1.4rem;'>CẤU HÌNH REAL-ESRGAN</h2>
        <span style='color: #64748B; font-size: 0.85rem;'>Tối ưu hóa xử lý ảnh</span>
    </div>
""", unsafe_allow_html=True)

model_choice = st.sidebar.selectbox(
    "Mô hình xử lý (Model Selection)",
    options=[
        "realesrgan-x4plus",
        "realesrgan-x4plus-anime",
        "realesr-animevideov3"
    ],
    index=0,
    help="• x4plus: chi tiết tốt cho ảnh thực tế.\n"
         "• x4plus-anime: tốt cho nét vẽ mảnh.\n"
         "• animevideov3: nhẹ hơn, phù hợp CPU."
)

if model_choice in ["realesrgan-x4plus", "realesrgan-x4plus-anime"]:
    scale_options = [4]
    scale_help = "Mô hình này yêu cầu tỷ lệ phóng đại 4x."
else:
    scale_options = [2, 3, 4]
    scale_help = "Chọn 2x hoặc 3x để chạy nhanh hơn trên CPU."

scale_choice = st.sidebar.selectbox(
    "Tỷ lệ phóng đại (Scale)",
    options=scale_options,
    index=0,
    help=scale_help
)

tile_size = st.sidebar.slider(
    "Kích thước phân mảnh (Tile Size)",
    min_value=100,
    max_value=800,
    value=200,
    step=50,
    help="Trên Streamlit Cloud nên để 100–200 để tránh lỗi bộ nhớ."
)

auto_resize_cloud = st.sidebar.checkbox(
    "Tối ưu kích thước trên Cloud",
    value=(os.name != "nt"),
    help="Tự động thu nhỏ ảnh trước khi xử lý để chạy nhanh hơn trên CPU cloud."
)

if auto_resize_cloud:
    cloud_resolution = st.sidebar.selectbox(
        "Độ phân giải xử lý",
        options=[
            "360px (Siêu nhanh)",
            "480px (Nhanh)",
            "640px (Chậm hơn)"
        ],
        index=1
    )
    max_cloud_width = int(cloud_resolution.split("px")[0])
else:
    max_cloud_width = 1000

st.sidebar.markdown("<div class='divider' style='margin: 16px 0;'></div>", unsafe_allow_html=True)

st.sidebar.markdown("### ⚡ Trạng Thái Hệ Thống")

if os.name == "nt":
    os_display = "Windows"
    acceleration_display = "GPU/Vulkan hoặc CPU"
else:
    os_display = "Linux / Streamlit Cloud" if sys.platform != "darwin" else "macOS"
    acceleration_display = "CPU Mode trên Cloud"

st.sidebar.info(
    f"💻 Hệ điều hành: {os_display}\n"
    f"⚙️ Chế độ xử lý: {acceleration_display}\n"
    f"📦 Phiên bản: {APP_VERSION}\n"
    "📊 Tối ưu hóa: Có"
)

executable_found = get_executable_path() is not None
if executable_found:
    st.sidebar.success("✅ Real-ESRGAN: SẴN SÀNG")
else:
    st.sidebar.warning("⚠️ Real-ESRGAN sẽ được tải tự động khi xử lý")

col_upload, col_view = st.columns([1, 2.2], gap="large")

with col_upload:
    st.markdown("""
        <div class="clinical-card">
            <h3 style='margin: 0 0 12px 0; font-size: 1.15rem; color: #38BDF8 !important;'>📁 TẢI LÊN ẢNH ĐIỆN TIM</h3>
            <p style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 16px;'>Tải lên bản quét giấy ECG mờ, chất lượng thấp hoặc có nhiều nhiễu hạt.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Chọn tệp ảnh (JPEG, PNG, JPG)...",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        orig_image = Image.open(uploaded_file).convert("RGB")

        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext not in [".png", ".jpg", ".jpeg"]:
            file_ext = ".png"

        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        if auto_resize_cloud and orig_image.size[0] > max_cloud_width:
            w_percent = max_cloud_width / float(orig_image.size[0])
            h_size = int(float(orig_image.size[1]) * w_percent)
            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
            processed_orig_image = orig_image.resize((max_cloud_width, h_size), resample_filter)
        else:
            processed_orig_image = orig_image

        input_img_path = TEMP_DIR / "input_temp.png"
        processed_orig_image.save(input_img_path)

        st.markdown(f"""
            <div class="clinical-card" style="margin-top: 16px; padding: 16px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; color: #E2E8F0 !important;">Thông số ảnh gốc:</h4>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #94A3B8;"><b>Kích thước gốc:</b> {orig_image.size[0]} x {orig_image.size[1]} px</p>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #94A3B8;"><b>Kích thước xử lý:</b> {processed_orig_image.size[0]} x {processed_orig_image.size[1]} px</p>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #94A3B8;"><b>Định dạng:</b> {uploaded_file.type}</p>
            </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("🚀 BẮT ĐẦU TĂNG NÉT ĐIỆN TIM")
    else:
        st.info("👈 Hãy kéo thả hoặc chọn tệp ảnh điện tim ở khung bên trái để bắt đầu.")
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
                    unsafe_allow_html=True
                )
                progress_bar.progress(progress_val)

            try:
                t_start = time.time()

                update_progress("Đang kiểm tra và tải Real-ESRGAN nếu cần...", 0.1)
                download_realesrgan_binary(status_callback=update_progress)

                update_progress("Đang xử lý ảnh bằng CPU Mode...", 0.6)
                run_realesrgan(
                    input_path=str(input_img_path),
                    output_path=str(output_img_path),
                    model_name=model_choice,
                    tile_size=tile_size,
                    scale=scale_choice
                )

                duration = time.time() - t_start

                update_progress("Hoàn tất! Đang kết xuất kết quả...", 1.0)
                time.sleep(0.5)

                progress_status.empty()
                progress_bar.empty()

                st.session_state["processed_file"] = {
                    "input": str(input_img_path),
                    "output": str(output_img_path),
                    "duration": duration,
                    "model": model_choice,
                    "scale": scale_choice
                }

                st.success(f"✨ Đã làm sắc nét thành công trong {duration:.2f} giây.")

            except Exception as e:
                progress_status.empty()
                progress_bar.empty()
                st.error(f"❌ Có lỗi xảy ra trong quá trình xử lý: {str(e)}")

        if "processed_file" in st.session_state and os.path.exists(st.session_state["processed_file"]["output"]):
            p_data = st.session_state["processed_file"]
            enhanced_img = Image.open(p_data["output"])
            orig_loaded = Image.open(p_data["input"])

            st.markdown("### 🔍 So Sánh Kết Quả")

            tab_side_by_side, tab_zoom = st.tabs(["📊 Xem Toàn Cảnh", "🔎 Soi Chi Tiết"])

            with tab_side_by_side:
                col_orig, col_enh = st.columns(2)

                with col_orig:
                    st.markdown("<p style='text-align: center; color: #EF4444; font-weight: 600;'>ẢNH TRƯỚC</p>", unsafe_allow_html=True)
                    st.image(orig_loaded, use_column_width=True)

                with col_enh:
                    st.markdown("<p style='text-align: center; color: #10B981; font-weight: 600;'>ẢNH SAU</p>", unsafe_allow_html=True)
                    st.image(enhanced_img, use_column_width=True)

            with tab_zoom:
                w, h = orig_loaded.size
                w_enh, h_enh = enhanced_img.size

                crop_x = st.slider("Vị trí X (%)", 0, 100, 50, 1)
                crop_y = st.slider("Vị trí Y (%)", 0, 100, 50, 1)
                crop_w_pct = st.slider("Kích thước vùng soi (%)", 5, 40, 15, 1)

                crop_size_orig_w = max(1, int(w * (crop_w_pct / 100)))
                crop_size_orig_h = max(1, int(h * (crop_w_pct / 100)))

                center_x_orig = int(w * (crop_x / 100))
                center_y_orig = int(h * (crop_y / 100))

                x1_orig = max(0, center_x_orig - crop_size_orig_w // 2)
                y1_orig = max(0, center_y_orig - crop_size_orig_h // 2)
                x2_orig = min(w, x1_orig + crop_size_orig_w)
                y2_orig = min(h, y1_orig + crop_size_orig_h)

                scale_w = w_enh / w
                scale_h = h_enh / h

                x1_enh = int(x1_orig * scale_w)
                y1_enh = int(y1_orig * scale_h)
                x2_enh = int(x2_orig * scale_w)
                y2_enh = int(y2_orig * scale_h)

                cropped_orig = orig_loaded.crop((x1_orig, y1_orig, x2_orig, y2_orig))
                cropped_enhanced = enhanced_img.crop((x1_enh, y1_enh, x2_enh, y2_enh))

                col_crop_o, col_crop_e = st.columns(2)

                with col_crop_o:
                    st.markdown("<p style='text-align: center; color: #EF4444;'>VÙNG SOI TRƯỚC</p>", unsafe_allow_html=True)
                    st.image(cropped_orig, use_column_width=True)

                with col_crop_e:
                    st.markdown("<p style='text-align: center; color: #10B981;'>VÙNG SOI SAU</p>", unsafe_allow_html=True)
                    st.image(cropped_enhanced, use_column_width=True)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 💾 TẢI XUỐNG")

            with open(p_data["output"], "rb") as f:
                img_bytes = f.read()

            st.download_button(
                label="📥 TẢI XUỐNG ẢNH ECG SẮC NÉT",
                data=img_bytes,
                file_name=f"ECG_Enhanced_{Path(uploaded_file.name).stem}.png",
                mime="image/png"
            )

        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("👆 Tải ảnh điện tim lên và bấm nút xử lý để xem kết quả.")