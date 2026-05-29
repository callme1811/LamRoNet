import streamlit as st
import time
import os
import shutil
import platform
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Import source utilities
from src.homography import detect_paper_corners, four_point_transform
from src.segment_waveform import segment_ecg_waveform, apply_waveform_grid_overlay
from generator_wrapper import generate_synthetic_ecg

# Define temp directory
TEMP_DIR = Path(__file__).resolve().parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Set Streamlit Page Configurations
st.set_page_config(
    page_title="LamRoNet - AI ECG Dewarping & Synthesis",
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

# Load styles
load_css("styles.css")

# --- UI HEADER ---
st.markdown("""
    <div class="heartbeat-container">
        <span class="heart-icon">❤️</span>
        <div style="flex-grow: 0;">
            <span class="badge-teal">ECG Digitization & Dataset Generation Suite</span>
            <h1 style='margin: 0; padding-top: 4px; font-size: 2.2rem; background: linear-gradient(90deg, #FFFFFF 0%, #06B6D4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                LAMRONET - AI ECG RECTIFIER & DATASET GENERATOR
            </h1>
        </div>
        <div class="pulse-line"></div>
    </div>
    <div class="divider" style="margin-top: 8px; margin-bottom: 20px;"></div>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 16px;'>
        <h2 style='margin: 0; color: #06B6D4 !important; font-size: 1.3rem;'>HỆ THỐNG ECG AI</h2>
    </div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox(
    "CHỌN CHỨC NĂNG HỆ THỐNG",
    ["%s AI ECG Rectifier & Dewarp" % "🩺", "%s ECG Dataset Generator (Synthesis)" % "🧪"],
    index=0,
    help="• AI ECG Rectifier: Làm phẳng (dewarp) ảnh ECG chụp lệch và tách sóng.\n"
         "• ECG Dataset Generator: Sinh dữ liệu điện tim giả lập để huấn luyện AI."
)

st.sidebar.markdown("<div class='divider' style='margin: 16px 0;'></div>", unsafe_allow_html=True)

# Helper function to draw corners on a PIL image
def draw_corners_preview(image, corners):
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    # Scale lines based on image size
    w, h = image.size
    line_w = max(3, int(min(w, h) * 0.005))
    circle_r = max(5, int(min(w, h) * 0.012))
    
    # Draw polygon
    draw.polygon(corners, outline="#EF4444", width=line_w)
    # Draw vertices
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(corners):
        draw.ellipse([pt[0]-circle_r, pt[1]-circle_r, pt[0]+circle_r, pt[1]+circle_r], fill="#22D3EE", outline="#0EA5E9", width=2)
        # Draw labels
        draw.text((pt[0] + circle_r + 2, pt[1] - circle_r - 2), labels[i], fill="#FFFFFF")
    return draw_img

# --- HUB 1: AI ECG RECTIFIER & DEWARP ---
if "🩺" in app_mode:
    
    st.subheader("🩺 AI ECG Rectifier & Dewarping Hub")
    st.markdown("Khử biến dạng góc chụp (dewarp/flatten) bằng Homography phối hợp tách đường sóng điện tim (waveform segment) phục vụ số hoá và huấn luyện AI đọc điện tim.")
    
    # Configuration inputs in sidebar
    st.sidebar.markdown("### 🛠 Cấu hình Tách Sóng (Segmentation)")
    
    seg_model_type = st.sidebar.selectbox(
        "Mô hình Tách Sóng",
        ["OpenCV Adaptive Threshold (Khuyên dùng - Nhanh/Mịn)", "U-Net (PyTorch Custom Weights)", "SegFormer (HuggingFace Transformers)"],
        index=0
    )
    
    weights_path = ""
    if seg_model_type == "U-Net (PyTorch Custom Weights)":
        weights_path = st.sidebar.text_input("Đường dẫn tệp weights U-Net (.pt)", value="unet_ecg.pt")
    elif seg_model_type == "SegFormer (HuggingFace Transformers)":
        weights_path = st.sidebar.text_input("Model ID / Path", value="nvidia/segformer-b0-finetuned-ade-512-512")
        
    grid_render_style = st.sidebar.selectbox(
        "Chế độ hiển thị kết quả tách sóng",
        ["Clinical Pink Grid (Lưới hồng tinh khiết)", "Pure White Background (Chỉ giữ sóng)", "Original Flat Image (Ảnh phẳng gốc)"],
        index=0
    )
    
    c_val_param = st.sidebar.slider(
        "Độ nhạy tách sóng (OpenCV C)",
        min_value=3,
        max_value=25,
        value=10,
        step=1,
        help="Chỉ áp dụng với OpenCV Fallback. Giá trị càng nhỏ sẽ tách các đường sóng mảnh đậm nét hơn."
    )

    col_main_left, col_main_right = st.columns([1.2, 1.8], gap="large")
    
    with col_main_left:
        st.markdown("""
            <div class="clinical-card" style="padding: 18px; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #38BDF8 !important; font-size: 1rem;">📁 TẢI LÊN ẢNH ĐIỆN TIM THỰC TẾ</h4>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_ecg = st.file_uploader(
            "Tải lên tệp ảnh quét hoặc chụp ECG...",
            type=["png", "jpg", "jpeg"],
            key="ecg_rectify_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_ecg is not None:
            # Save and load original image
            orig_img = Image.open(uploaded_ecg).convert("RGB")
            W_orig, H_orig = orig_img.size
            
            orig_path = TEMP_DIR / f"rectify_input{os.path.splitext(uploaded_ecg.name)[1]}"
            orig_img.save(orig_path)
            
            st.markdown(f"<p style='color:#94A3B8; font-size:0.85rem; margin-top:4px;'>Độ phân giải: {W_orig} x {H_orig} px</p>", unsafe_allow_html=True)
            
            # --- CORNER SELECTOR COMPONENT ---
            st.markdown("""
                <div class="clinical-card" style="padding: 16px; margin-top: 12px; margin-bottom: 8px;">
                    <h4 style="margin: 0 0 8px 0; color: #E2E8F0 !important; font-size: 0.95rem;">🎯 Định vị Khung Điện Tim (Corners)</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: #94A3B8;">Hệ thống sẽ kéo dãn 4 góc này thành hình chữ nhật phẳng.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Checkbox for YOLO
            use_yolo = st.checkbox("Tự động nhận diện góc bằng YOLOv8-seg", value=False)
            yolo_path = st.text_input("Đường dẫn model YOLOv8-seg (.pt)", value="models/yolo_ecg_paper.pt", disabled=not use_yolo)
            
            auto_corners = None
            if use_yolo and st.button("🔍 Chạy YOLOv8 Corner Detection"):
                with st.spinner("Đang chạy mô hình YOLOv8-seg tìm góc..."):
                    auto_corners = detect_paper_corners(str(orig_path), model_path=yolo_path)
                    if auto_corners is not None:
                        st.success("Đã định vị được góc giấy từ YOLOv8-seg!")
                        st.session_state["tl_x"] = int((auto_corners[0][0] / W_orig) * 100)
                        st.session_state["tl_y"] = int((auto_corners[0][1] / H_orig) * 100)
                        st.session_state["tr_x"] = int((auto_corners[1][0] / W_orig) * 100)
                        st.session_state["tr_y"] = int((auto_corners[1][1] / H_orig) * 100)
                        st.session_state["br_x"] = int((auto_corners[2][0] / W_orig) * 100)
                        st.session_state["br_y"] = int((auto_corners[2][1] / H_orig) * 100)
                        st.session_state["bl_x"] = int((auto_corners[3][0] / W_orig) * 100)
                        st.session_state["bl_y"] = int((auto_corners[3][1] / H_orig) * 100)
                    else:
                        st.error("Không tìm thấy góc tự động. Vui lòng chỉnh bằng tay bên dưới.")
            
            # Sliders for manual adjustments
            st.markdown("**Hiêu chỉnh 4 góc theo % kích thước:**")
            
            col_sl_l, col_sl_r = st.columns(2)
            with col_sl_l:
                st.markdown("<span style='color:#38BDF8;font-size:0.85rem;'>Góc Trên-Trái (TL)</span>", unsafe_allow_html=True)
                s_tl_x = st.slider("TL X %", 0, 100, st.session_state.get("tl_x", 5))
                s_tl_y = st.slider("TL Y %", 0, 100, st.session_state.get("tl_y", 5))
                
                st.markdown("<span style='color:#F43F5E;font-size:0.85rem;'>Góc Dưới-Trái (BL)</span>", unsafe_allow_html=True)
                s_bl_x = st.slider("BL X %", 0, 100, st.session_state.get("bl_x", 5))
                s_bl_y = st.slider("BL Y %", 0, 100, st.session_state.get("bl_y", 95))
                
            with col_sl_r:
                st.markdown("<span style='color:#38BDF8;font-size:0.85rem;'>Góc Trên-Phải (TR)</span>", unsafe_allow_html=True)
                s_tr_x = st.slider("TR X %", 0, 100, st.session_state.get("tr_x", 95))
                s_tr_y = st.slider("TR Y %", 0, 100, st.session_state.get("tr_y", 5))
                
                st.markdown("<span style='color:#F43F5E;font-size:0.85rem;'>Góc Dưới-Phải (BR)</span>", unsafe_allow_html=True)
                s_br_x = st.slider("BR X %", 0, 100, st.session_state.get("br_x", 95))
                s_br_y = st.slider("BR Y %", 0, 100, st.session_state.get("br_y", 95))

            # Store in session state
            st.session_state["tl_x"] = s_tl_x
            st.session_state["tl_y"] = s_tl_y
            st.session_state["tr_x"] = s_tr_x
            st.session_state["tr_y"] = s_tr_y
            st.session_state["br_x"] = s_br_x
            st.session_state["br_y"] = s_br_y
            st.session_state["bl_x"] = s_bl_x
            st.session_state["bl_y"] = s_bl_y
            
            # Map percentages back to pixel coordinates
            pts_pixels = [
                (int((s_tl_x / 100) * W_orig), int((s_tl_y / 100) * H_orig)),
                (int((s_tr_x / 100) * W_orig), int((s_tr_y / 100) * H_orig)),
                (int((s_br_x / 100) * W_orig), int((s_br_y / 100) * H_orig)),
                (int((s_bl_x / 100) * W_orig), int((s_bl_y / 100) * H_orig))
            ]
            
            run_rectify = st.button("🚀 BẮT ĐẦU LÀM PHẲNG & TÁCH SÓNG")
        else:
            st.info("👈 Hãy tải lên ảnh ECG chụp từ thiết bị di động ở khung bên trái.")
            run_rectify = False
            
    with col_main_right:
        if uploaded_ecg is not None:
            st.markdown("### 🔎 Định vị Bề Măt Giấy ECG")
            preview_img = draw_corners_preview(orig_img, pts_pixels)
            st.image(preview_img, use_column_width=True, caption="Xem trước 4 góc hiệu chỉnh để uốn phẳng giấy ECG.")
            
            if run_rectify:
                status_box = st.empty()
                progress_bar = st.progress(0.0)
                
                try:
                    t_start = time.time()
                    
                    # Target paths
                    flat_path = TEMP_DIR / "ecg_flat.png"
                    mask_path = TEMP_DIR / "ecg_wave_mask.png"
                    final_path = TEMP_DIR / "ecg_rectified_enhanced.png"
                    
                    # Step 1: Flatting / Homography
                    status_box.info("Step 1: Áp dụng phép chiếu Homography làm phẳng tờ giấy điện tim...")
                    progress_bar.progress(0.3)
                    
                    image_np = np.array(orig_img)
                    pts_np = np.array(pts_pixels, dtype=np.float32)
                    flat_image_np = four_point_transform(image_np, pts_np)
                    
                    # Save flat image
                    Image.fromarray(flat_image_np).save(flat_path)
                    
                    # Step 2: Signal/Waveform Segmentation
                    status_box.info("Step 2: Chạy mô hình tách sóng (%s)..." % seg_model_type)
                    progress_bar.progress(0.6)
                    
                    flat_image_rgb = cv2.cvtColor(flat_image_np, cv2.COLOR_BGR2RGB)
                    
                    waveform_mask = segment_ecg_waveform(
                        flat_image_rgb,
                        model_path=weights_path if weights_path else None,
                        model_type="OpenCV" if "OpenCV" in seg_model_type else ("U-Net" if "U-Net" in seg_model_type else "SegFormer"),
                        C_val=c_val_param
                    )
                    
                    # Save mask
                    cv2.imwrite(str(mask_path), waveform_mask)
                    
                    # Step 3: Overlay rendering
                    status_box.info("Step 3: Khôi phục lưới giấy và kết xuất hình ảnh sắc nét...")
                    progress_bar.progress(0.9)
                    
                    if grid_render_style == "Original Flat Image (Ảnh phẳng gốc)":
                        shutil.copy(str(flat_path), str(final_path))
                    else:
                        rendered_overlay = apply_waveform_grid_overlay(
                            waveform_mask,
                            grid_style="Clinical Pink Grid" if "Pink Grid" in grid_render_style else "Pure White"
                        )
                        cv2.imwrite(str(final_path), cv2.cvtColor(rendered_overlay, cv2.COLOR_RGB2BGR))
                        
                    t_end = time.time()
                    duration = t_end - t_start
                    
                    progress_bar.progress(1.0)
                    time.sleep(0.3)
                    progress_bar.empty()
                    status_box.empty()
                    
                    st.success("✨ Đã làm phẳng và trích xuất sóng thành công trong %.2f giây!" % duration)
                    
                    # Display results
                    st.markdown("### 📊 Kết Quả Khôi Phục")
                    col_res_flat, col_res_seg = st.columns(2)
                    with col_res_flat:
                        st.markdown("<p style='text-align: center; color: #38BDF8; font-weight: 600;'>ẢNH UỐN PHẲNG (WARPED/FLAT)</p>", unsafe_allow_html=True)
                        st.image(str(flat_path), use_column_width=True)
                    with col_res_seg:
                        st.markdown("<p style='text-align: center; color: #10B981; font-weight: 600;'>SÓNG ĐÃ TÁCH KHỬ NHIỄU (SEGMENTED WAVE)</p>", unsafe_allow_html=True)
                        st.image(str(final_path), use_column_width=True)
                        
                    # Download button
                    with open(final_path, "rb") as f:
                        img_bytes = f.read()
                    
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 TẢI XUỐNG ẢNH ĐIỆN TIM LÀM SẠCH (PNG)",
                        data=img_bytes,
                        file_name="ECG_Rectified_%s" % uploaded_ecg.name,
                        mime="image/png"
                    )
                    
                except Exception as e:
                    progress_bar.empty()
                    status_box.empty()
                    st.error("Lỗi khi xử lý ảnh điện tim: %s" % e)

# --- HUB 2: ECG DATASET GENERATOR (SYNTHESIS) ---
else:
    st.subheader("🧪 ECG Synthetic Training Data Generator")
    st.markdown("Sử dụng **ECG-Image-Kit** để tạo hàng loạt ảnh điện tim có lưới nền cùng nhãn bounding box (tọa độ các chuyển đạo và chữ viết) để phục vụ huấn luyện mô hình học sâu.")
    
    col_gen_left, col_gen_right = st.columns([1.1, 1.9], gap="large")
    
    with col_gen_left:
        st.markdown("""
            <div class="clinical-card" style="padding: 16px; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #38BDF8 !important; font-size: 1rem;">⚙️ THÔNG SỐ GIẢ LẬP</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Data source selection
        data_source = st.radio("Nguồn dữ liệu ECG gốc (WFDB)", ["Bản ghi mẫu PTB-XL (00001_lr)", "Tải lên tệp WFDB riêng (.dat + .hea)"])
        
        custom_dat = None
        custom_hea = None
        if data_source == "Tải lên tệp WFDB riêng (.dat + .hea)":
            col_wfdb_1, col_wfdb_2 = st.columns(2)
            with col_wfdb_1:
                custom_dat = st.file_uploader("Chọn file .dat", type=["dat"])
            with col_wfdb_2:
                custom_hea = st.file_uploader("Chọn file .hea", type=["hea"])
                
        # Layout configurations
        num_cols = st.selectbox("Cấu trúc cột Layout (Lead columns)", [1, 2, 4, 12], index=2, help="4 cột tương đương cách bố trí 12-lead (3 chuyển đạo mỗi cột) phổ biến lâm sàng.")
        full_lead = st.selectbox("Chuyển đạo kéo dài bên dưới (Full mode lead)", ["II", "V1", "V5", "None"], index=0)
        grid_col_index = st.slider("Chỉ mục màu lưới giấy (Standard Grid Color Index)", 1, 7, 5, help="Giá trị 5 đại diện cho màu đỏ/hồng lưới điện tim tiêu chuẩn.")
        dpi_res = st.slider("Độ phân giải bản in (Resolution DPI)", 100, 300, 200, step=50)
        
        # Bounding box labels options
        st.markdown("**Xuất Nhãn Huấn Luyện (Training Annotations):**")
        lead_coord_csv = st.checkbox("Tạo toạ độ vùng chuyển đạo (Lead Box BBox)", value=True)
        label_coord_csv = st.checkbox("Tạo toạ độ nhãn chữ viết (Label Text BBox)", value=True)
        add_qr = st.checkbox("Tích hợp mã QR lên góc tờ giấy (QR Code Embed)", value=False)
        
        # Distortions configs
        st.markdown("**Các Yếu Tố Nhiễu Giấy & Ảnh Chụp (Paper Distortions):**")
        add_creases = st.checkbox("Thêm vết gấp và nếp nhăn giấy (Creases & Wrinkles)", value=True)
        add_handwritten = st.checkbox("Thêm văn bản viết tay loang lổ (Handwritten Annotations)", value=False)
        add_augmentations = st.checkbox("Thêm góc xoay camera & đổ bóng nhiễu (Camera Perspective & Noise)", value=True)
        
        camera_rot = 0
        photo_noise = 0
        if add_augmentations:
            col_aug_1, col_aug_2 = st.columns(2)
            with col_aug_1:
                camera_rot = st.slider("Góc xoay camera", -10, 10, 3)
            with col_aug_2:
                photo_noise = st.slider("Độ nhiễu hạt / bóng mờ", 0, 100, 40)
                
        seed_value = st.number_input("Random Seed", value=42, step=1)
        
        run_generation = st.button("🚀 BẮT ĐẦU TẠO ẢNH ECG GIẢ LẬP")
        
    with col_gen_right:
        st.markdown("### 📊 Kết Quả Sinh ECG Giả Lập")
        
        if run_generation:
            with st.spinner("Đang chạy ECG-Image-Kit pipeline để tổng hợp ảnh và tọa độ nhãn..."):
                try:
                    # Setup custom files if uploaded
                    in_dat_path = None
                    in_hea_path = None
                    gen_out_dir = str(TEMP_DIR / "generated_run")
                    
                    # Clean previous run
                    if os.path.exists(gen_out_dir):
                        shutil.rmtree(gen_out_dir)
                    os.makedirs(gen_out_dir, exist_ok=True)
                    
                    if data_source == "Tải lên tệp WFDB riêng (.dat + .hea)":
                        if custom_dat is not None and custom_hea is not None:
                            in_dat_path = os.path.join(gen_out_dir, custom_dat.name)
                            in_hea_path = os.path.join(gen_out_dir, custom_hea.name)
                            with open(in_dat_path, "wb") as f:
                                f.write(custom_dat.read())
                            with open(in_hea_path, "wb") as f:
                                f.write(custom_hea.read())
                        else:
                            st.warning("Vui lòng tải lên cả hai tệp .dat và .hea để chạy giả lập.")
                            st.stop()
                            
                    # Call generator wrapper
                    res = generate_synthetic_ecg(
                        input_file=in_dat_path,
                        header_file=in_hea_path,
                        output_dir=gen_out_dir,
                        seed=seed_value,
                        resolution=dpi_res,
                        pad_inches=0,
                        print_header=True,
                        num_columns=num_cols,
                        full_mode=full_lead,
                        mask_unplotted_samples=False,
                        add_qr_code=add_qr,
                        hw_text=add_handwritten,
                        wrinkles=add_creases,
                        augment=add_augmentations,
                        lead_bbox=lead_coord_csv,
                        lead_name_bbox=label_coord_csv,
                        standard_grid_color=grid_col_index,
                        rotate=camera_rot,
                        noise=photo_noise,
                        store_config=2
                    )
                    
                    if res["success"]:
                        st.success("Tạo dữ liệu điện tim giả lập thành công!")
                        
                        # Display generated image
                        generated_pngs = res["pngs"]
                        if len(generated_pngs) > 0:
                            st.image(generated_pngs[0], use_column_width=True, caption="Ảnh ECG giả lập với nếp gấp và nhiễu camera.")
                            
                            # Show CSV tables
                            st.markdown("#### 📁 Tọa Độ Nhãn Bounding Box (YOLO/U-Net training coordinates)")
                            
                            if res["coordinates_csv"] and os.path.exists(res["coordinates_csv"]):
                                import pandas as pd
                                df_coords = pd.read_csv(res["coordinates_csv"])
                                st.dataframe(df_coords.head(15), height=200)
                                
                            if res["gridsizes_csv"] and os.path.exists(res["gridsizes_csv"]):
                                import pandas as pd
                                df_grids = pd.read_csv(res["gridsizes_csv"])
                                st.markdown("#### 📏 Kích thước ô lưới điện tim (Grid spacing details)")
                                st.dataframe(df_grids, height=120)
                            
                            # Zip all files in the output directory and make it downloadable
                            zip_path = TEMP_DIR / "synthetic_dataset.zip"
                            if zip_path.exists():
                                zip_path.unlink()
                                
                            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_f:
                                for root, _, files in os.walk(gen_out_dir):
                                    for file in files:
                                        file_full_path = os.path.join(root, file)
                                        zip_f.write(file_full_path, os.path.relpath(file_full_path, gen_out_dir))
                                        
                            with open(zip_path, "rb") as fz:
                                zip_bytes = fz.read()
                                
                            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                            st.download_button(
                                label="📥 TẢI XUỐNG TRỌN BỘ DỮ LIỆU HUẤN LUYỆN (ZIP)",
                                data=zip_bytes,
                                file_name="ECG_Synthetic_Dataset.zip",
                                mime="application/zip"
                            )
                        else:
                            st.error("Không tìm thấy tệp ảnh .png kết quả trong thư mục đầu ra.")
                    else:
                        st.error("Generator chạy thất bại: %s" % res['stderr'])
                        
                except Exception as e:
                    st.error("Lỗi hệ thống generator: %s" % e)
        else:
            st.info("👆 Hãy thiết lập cấu hình giả lập ở khung bên trái và bấm 'BẮT ĐẦU TẠO ẢNH ECG GIẢ LẬP' để sinh dữ liệu huấn luyện.")
