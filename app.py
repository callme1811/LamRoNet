import streamlit as st
import time
import os
import shutil
import platform
from pathlib import Path
from PIL import Image
from utils import (
    download_realesrgan_binary,
    run_realesrgan,
    preprocess_image,
    postprocess_image,
    TEMP_DIR,
    get_executable_path
)

# Set Streamlit Page Configurations
st.set_page_config(
    page_title="AI ECG Enhancer - LamRoNet",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load external CSS styles
def load_css(file_name):
    css_path = Path(__file__).resolve().parent / file_name
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load our premium clinical stylesheet
load_css("styles.css")

# --- UI HEADER ---
st.markdown("""
    <div class="heartbeat-container">
        <span class="heart-icon">❤️</span>
        <div style="flex-grow: 0;">
            <span class="badge-teal">Pure AI & OpenCV Medical Super-Resolution Pipeline</span>
            <h1 style='margin: 0; padding-top: 4px; font-size: 2.2rem; background: linear-gradient(90deg, #FFFFFF 0%, #06B6D4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                AI ECG ENHANCER - TĂNG ĐỘ NÉT ĐIỆN TIM VÀ VĂN BẢN
            </h1>
        </div>
        <div class="pulse-line"></div>
    </div>
    <p style='color: #94A3B8; margin-top: -8px; margin-bottom: 24px; font-size: 1.05rem;'>
        Khôi phục, khử nhiễu và làm sắc nét các bản quét điện tim (ECG) mờ nhạt bằng mô hình học sâu <b>Real-ESRGAN</b> kết hợp bộ lọc xử lý ảnh y tế nâng cao.
    </p>
    <div class="divider"></div>
""", unsafe_allow_html=True)

# --- SIDEBAR SETTINGS ---
st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: #06B6D4 !important; font-size: 1.4rem;'>CẤU HÌNH PIPELINE LÂM SÀNG</h2>
    </div>
""", unsafe_allow_html=True)

# Pipeline Configuration Mode
st.sidebar.markdown("### 🛠️ Bộ Lọc & Siêu Phân Giải")

scale_label = st.sidebar.selectbox(
    "Tỷ lệ siêu phân giải (Upscale Scale)",
    options=[
        "1x (Chỉ dùng bộ lọc OpenCV - Siêu tốc <0.5s)",
        "2x (Siêu phân giải nét)",
        "3x (Siêu phân giải chi tiết)",
        "4x (Siêu phân giải cực đại)"
    ],
    index=1,
    help="• 1x: Bỏ qua mô hình AI, chỉ áp dụng bộ lọc xử lý ảnh y tế nâng cao bằng OpenCV. Cực nhanh!\n"
         "• 2x, 3x, 4x: Chạy mô hình AI tương ứng kết hợp bộ lọc làm nét mịn."
)
scale_choice = int(scale_label.split("x")[0])

model_choice = st.sidebar.selectbox(
    "Mô hình AI (Model Selection)",
    options=[
        "realesrgan-x4plus",
        "realesrnet-x4plus",
        "realesrgan-x4plus-anime",
        "realesr-animevideov3"
    ],
    index=0,
    disabled=(scale_choice == 1),
    help="• 'x4plus': Tối ưu nhất cho chữ viết, lưới số và đường nét điện tim thực tế (Khuyên dùng!).\n"
         "• 'realesrnet-x4plus': Phiên bản khử nhiễu mịn đường sóng (MSE loss).\n"
         "• 'x4plus-anime': Tối ưu cho tranh vẽ 2D, dễ làm nhòe mất chữ và chi tiết nhỏ y tế.\n"
         "• 'animevideov3': Rất nhanh và nhẹ."
)

tile_size = st.sidebar.slider(
    "Kích thước phân mảnh (Tile Size)",
    min_value=0,
    max_value=800,
    value=0,
    step=50,
    disabled=(scale_choice == 1),
    help="• 0 (Auto): Hệ thống tự động ghép mảnh tối ưu (Khuyên dùng để tránh sọc lệch lưới/bóng ma).\n"
         "• 100-800: Chia nhỏ mảnh khi card đồ họa bị tràn bộ nhớ (Out of Memory)."
)

# Advanced Medical Image Filters Expander
with st.sidebar.expander("🩺 Bộ lọc nâng cao (Medical Filters)", expanded=True):
    color_mode = st.selectbox(
        "Chế độ hiển thị lâm sàng",
        options=[
            "Original Enhanced",
            "Clinical Monochromatic",
            "Waveform Isolation"
        ],
        index=0,
        help="• 'Original Enhanced': Giữ màu nguyên bản, tăng bão hòa và độ tương phản lưới giấy.\n"
             "• 'Clinical Monochromatic': Chuyển ảnh xám tương phản cao, làm nổi bật đường sóng trên nền giấy trắng.\n"
             "• 'Waveform Isolation': Làm mờ nhẹ nền giấy lưới hồng để tập trung tối đa vào đường sóng ECG màu đen."
    )
    
    clahe_clip = st.slider(
        "Độ tương phản cục bộ (CLAHE)",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.2,
        help="Tăng tương phản thích ứng giúp kéo các đường sóng mờ nhạt nổi bật rõ nét mà không làm cháy hình."
    )
    
    pre_sharpen_weight = st.slider(
        "Làm nét thô trước AI (Pre-Sharpen)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Làm sắc nét các cạnh trước khi gửi vào mô hình AI để tránh hiện tượng vỡ hình và răng cưa."
    )
    
    denoise_strength = st.slider(
        "Khử nhiễu nền giấy (Denoise)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Lọc các hạt nhiễu, cát trên giấy quét bằng bộ lọc Bilateral bảo toàn biên."
    )
    
    post_sharpen_weight = st.slider(
        "Làm nét tinh sau AI (Post-Sharpen)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Tối ưu hóa độ sắc cạnh cuối cùng của ảnh sau khi phóng đại."
    )

st.sidebar.markdown("<div class='divider' style='margin: 16px 0;'></div>", unsafe_allow_html=True)

# System Diagnostics Display in Sidebar
st.sidebar.markdown("### ⚡ Trạng Thế Hệ Thống")
os_name = platform.system()
st.sidebar.info(f"💻 Hệ điều hành: {os_name}\n"
                "⚙️ Gia tốc: Vulkan GPU (Tự động fallback CPU)\n"
                "📊 Tự động tối ưu hóa: Có (Tiling)")

executable_found = get_executable_path() is not None
if executable_found:
    st.sidebar.success("✅ Real-ESRGAN Binary: SẴN SÀNG")
else:
    st.sidebar.warning("⚠️ Cần tải mô hình Real-ESRGAN (Tự động tải khi bấm xử lý)")

# --- MAIN WORKSPACE ---
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
        # Load and save original image
        orig_image = Image.open(uploaded_file)
        
        # Extract file extension and normalize name
        file_ext = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
        
        # Ensure temporary directory exists explicitly
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save original to temp with clean, safe name
        input_img_path = TEMP_DIR / f"input_temp{file_ext}"
        orig_image.save(input_img_path)
        
        # Details metadata card
        st.markdown(f"""
            <div class="clinical-card" style="margin-top: 16px; padding: 16px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; color: #E2E8F0 !important;">Thông số ảnh gốc:</h4>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #94A3B8;"><b>Kích thước:</b> {orig_image.size[0]} x {orig_image.size[1]} px</p>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #94A3B8;"><b>Định dạng:</b> {orig_image.format}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Upscale button
        run_btn = st.button("🚀 BẮT ĐẦU LÀM RÕ NÉT BẰNG AI")
    else:
        st.info("👈 Hãy kéo thả hoặc chọn tệp ảnh điện tim ở khung bên trái để bắt đầu.")
        run_btn = False

with col_view:
    if uploaded_file is not None:
        # Extract file extension and normalize name
        file_ext = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
        
        # Create container paths for output with clean, safe name
        output_img_path = TEMP_DIR / f"enhanced_temp{file_ext}"
        preprocessed_img_path = TEMP_DIR / f"preprocessed_temp{file_ext}"
        
        if run_btn:
            progress_status = st.empty()
            progress_bar = st.progress(0.0)
            
            def update_progress(msg, progress_val):
                progress_status.markdown(f"<p style='color: #22D3EE; font-weight: 500;'>⏳ {msg}</p>", unsafe_allow_html=True)
                progress_bar.progress(progress_val)
            
            try:
                t_start = time.time()
                
                # Step 1: Pre-processing (OpenCV)
                update_progress("Đang tối ưu cấu trúc ảnh lâm sàng (Khử nhiễu & CLAHE)...", 0.15)
                preprocess_image(
                    input_path=str(input_img_path),
                    output_path=str(preprocessed_img_path),
                    denoise_strength=denoise_strength,
                    clahe_clip=clahe_clip,
                    clahe_grid=8,
                    sharpen_weight=pre_sharpen_weight
                )
                
                # Step 2: AI Super Resolution or Bypass
                if scale_choice == 1:
                    update_progress("Sao chép ảnh đã xử lý lưới nét (Bỏ qua mô hình AI)...", 0.6)
                    shutil.copy(str(preprocessed_img_path), str(output_img_path))
                else:
                    # Check and download binary
                    update_progress("Đang kiểm tra và tải gói thực thi mô hình AI đa nền tảng...", 0.3)
                    download_realesrgan_binary(status_callback=update_progress)
                    
                    update_progress(f"Mô hình AI đang tái cấu trúc và làm sắc nét (Scale {scale_choice}x, Tự động fallback CPU nếu thiếu GPU)...", 0.6)
                    run_realesrgan(
                        input_path=str(preprocessed_img_path),
                        output_path=str(output_img_path),
                        model_name=model_choice,
                        tile_size=tile_size,
                        scale=scale_choice
                    )
                
                # Step 3: Post-processing (OpenCV Filters & Clinical Color Modes)
                update_progress("Áp dụng bộ lọc làm nét mịn và bộ lọc màu lâm sàng...", 0.85)
                postprocess_image(
                    input_path=str(output_img_path),
                    output_path=str(output_img_path),
                    mode=color_mode,
                    sharpen_weight=post_sharpen_weight
                )
                
                t_end = time.time()
                duration = t_end - t_start
                
                update_progress("Hoàn tất! Đang kết xuất kết quả...", 1.0)
                time.sleep(0.5)
                
                # Clear progress widgets
                progress_status.empty()
                progress_bar.empty()
                
                st.session_state["processed_file"] = {
                    "input": str(input_img_path),
                    "output": str(output_img_path),
                    "duration": duration,
                    "model": model_choice if scale_choice > 1 else "Bypassed (Pure OpenCV)",
                    "scale": scale_choice
                }
                st.success(f"✨ Đã làm sắc nét thành công trong {duration:.2f} giây!")
                
            except Exception as e:
                progress_status.empty()
                progress_bar.empty()
                st.error(f"❌ Có lỗi xảy ra trong quá trình nâng cấp ảnh: {str(e)}")
        
        # Display results if already processed
        if "processed_file" in st.session_state and os.path.exists(st.session_state["processed_file"]["output"]):
            p_data = st.session_state["processed_file"]
            enhanced_img = Image.open(p_data["output"])
            orig_loaded = Image.open(p_data["input"])
            
            # Showcase comparison
            st.markdown("### 🔍 So Sánh Kết Quả (Trước & Sau khi tăng nét)")
            
            tab_side_by_side, tab_zoom = st.tabs(["📊 Xem Toàn Cảnh", "🔎 Soi Chi Tiết (Zoom/Crop Inspector)"])
            
            with tab_side_by_side:
                col_orig, col_enh = st.columns(2)
                with col_orig:
                    st.markdown("<p style='text-align: center; color: #EF4444; font-weight: 600; margin-bottom: 8px;'>ẢNH GỐC (MỜ / NHIỄU)</p>", unsafe_allow_html=True)
                    st.image(orig_loaded, use_column_width=True)
                with col_enh:
                    curr_scale = p_data["scale"]
                    st.markdown(f"<p style='text-align: center; color: #10B981; font-weight: 600; margin-bottom: 8px;'>ẢNH TĂNG NÉT LÂM SÀNG (SIÊU PHÂN GIẢI {curr_scale}X)</p>", unsafe_allow_html=True)
                    st.image(enhanced_img, use_column_width=True)
            
            with tab_zoom:
                st.markdown("<p style='color: #94A3B8; font-size: 0.85rem; margin-bottom: 12px;'>Kéo các thanh trượt bên dưới để chọn khu vực soi kỹ chuyển đạo điện tim (phóng to pixel-perfect):</p>", unsafe_allow_html=True)
                
                # Get image dimensions
                w, h = orig_loaded.size
                w_enh, h_enh = enhanced_img.size
                
                # Crop controls
                col_slider_x, col_slider_y, col_slider_size = st.columns(3)
                with col_slider_x:
                    crop_x = st.slider("Vị trí X (%)", min_value=0, max_value=100, value=50, step=1)
                with col_slider_y:
                    crop_y = st.slider("Vị trí Y (%)", min_value=0, max_value=100, value=50, step=1)
                with col_slider_size:
                    crop_w_pct = st.slider("Kích thước vùng soi (%)", min_value=5, max_value=40, value=15, step=1)
                
                # Calculate coordinates
                # Original coordinate box
                crop_size_orig_w = int(w * (crop_w_pct / 100))
                crop_size_orig_h = int(h * (crop_w_pct / 100))
                
                center_x_orig = int(w * (crop_x / 100))
                center_y_orig = int(h * (crop_y / 100))
                
                x1_orig = max(0, center_x_orig - crop_size_orig_w // 2)
                y1_orig = max(0, center_y_orig - crop_size_orig_h // 2)
                x2_orig = min(w, x1_orig + crop_size_orig_w)
                y2_orig = min(h, y1_orig + crop_size_orig_h)
                
                # Enhanced coordinate box (calculated dynamic scale)
                scale_w = w_enh / w
                scale_h = h_enh / h
                
                x1_enh = int(x1_orig * scale_w)
                y1_enh = int(y1_orig * scale_h)
                x2_enh = int(x2_orig * scale_w)
                y2_enh = int(y2_orig * scale_h)
                
                # Perform Crops
                cropped_orig = orig_loaded.crop((x1_orig, y1_orig, x2_orig, y2_orig))
                cropped_enhanced = enhanced_img.crop((x1_enh, y1_enh, x2_enh, y2_enh))
                
                # Display cropped side-by-side
                col_crop_o, col_crop_e = st.columns(2)
                with col_crop_o:
                    st.markdown("<p style='text-align: center; color: #EF4444; font-weight: 500; font-size: 0.9rem;'>VÙNG SOI TRƯỚC AI (RĂNG CƯA / MỜ)</p>", unsafe_allow_html=True)
                    st.image(cropped_orig, use_column_width=True, caption=f"Toạ độ gốc X: {x1_orig}-{x2_orig}, Y: {y1_orig}-{y2_orig}")
                with col_crop_e:
                    st.markdown("<p style='text-align: center; color: #10B981; font-weight: 500; font-size: 0.9rem;'>VÙNG SOI SAU AI (SẮC NÉT / KHÔNG VỠ HÌNH)</p>", unsafe_allow_html=True)
                    st.image(cropped_enhanced, use_column_width=True, caption=f"Toạ độ nâng cấp X: {x1_enh}-{x2_enh}, Y: {y1_enh}-{y2_enh}")
 
            # --- DOWNLOAD CENTER ---
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 💾 TRUNG TÂM TẢI XUỐNG")
            
            # telemetry cards
            st.markdown(f"""
                <div style="display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px; background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; text-align: center;">
                        <span style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">Độ phân giải gốc</span>
                        <h4 style="margin: 4px 0 0 0; color: #E2E8F0 !important;">{w} x {h}</h4>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; text-align: center;">
                        <span style="font-size: 0.75rem; color: #06B6D4; text-transform: uppercase;">Độ phân giải nâng cấp ({p_data["scale"]}x)</span>
                        <h4 style="margin: 4px 0 0 0; color: #22D3EE !important;">{w_enh} x {h_enh}</h4>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; text-align: center;">
                        <span style="font-size: 0.75rem; color: #64748B; text-transform: uppercase;">Thời gian xử lý</span>
                        <h4 style="margin: 4px 0 0 0; color: #E2E8F0 !important;">{p_data["duration"]:.2f} giây</h4>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Prepare download bytes
            with open(p_data["output"], "rb") as f:
                img_bytes = f.read()
                
            st.download_button(
                label="📥 TẢI XUỐNG ẢNH ECG SẮC NÉT (PNG ĐỘ PHÂN GIẢI CAO)",
                data=img_bytes,
                file_name=f"ECG_Enhanced_{uploaded_file.name}",
                mime="image/png"
            )
            
        else:
            # Placeholder if not processed yet
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("👆 Tải ảnh điện tim lên ở cột bên trái và cấu hình bộ lọc ở sidebar, sau đó bấm nút 'BẮT ĐẦU LÀM RÕ NÉT BẰNG AI' để xem kết quả siêu nét tại đây.")