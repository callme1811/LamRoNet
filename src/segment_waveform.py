import cv2
import numpy as np

def segment_ecg_waveform(image, model=None):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Threshold tách nét đậm
    _, binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Loại nhiễu nhỏ
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned