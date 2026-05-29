import streamlit as st
import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO
from src.homography import flatten_ecg_paper
from src.segment_waveform import segment_ecg_waveform

st.set_page_config(page_title="ECG Image Preprocessing", layout="wide")

st.title("ECG Paper Flattening & Enhancement App")

uploaded_file = st.file_uploader(
    "Upload ảnh tờ điện tim ECG",
    type=["jpg", "jpeg", "png"]
)

@st.cache_resource
def load_yolo_model():
    return YOLO("models/yolo_ecg_paper.pt")

@st.cache_resource
def load_segmentation_model():
    # Có thể load U-Net hoặc SegFormer tại đây
    return None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Ảnh gốc")
    st.image(image_np, channels="RGB", use_column_width=True)

    yolo_model = load_yolo_model()
    seg_model = load_segmentation_model()

    with st.spinner("Đang phát hiện tờ ECG..."):
        results = yolo_model(image_np)

    with st.spinner("Đang làm phẳng ảnh ECG..."):
        flattened = flatten_ecg_paper(image_np, results)

    st.subheader("Ảnh ECG đã làm phẳng")
    st.image(flattened, channels="RGB", use_column_width=True)

    with st.spinner("Đang tách đường sóng ECG..."):
        waveform_mask = segment_ecg_waveform(flattened, seg_model)

    st.subheader("Mask đường sóng ECG")
    st.image(waveform_mask, use_column_width=True, clamp=True)

    st.success("Xử lý hoàn tất.")