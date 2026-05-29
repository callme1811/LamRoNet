import cv2
import numpy as np


def ensure_rgb(image):
    if image is None:
        raise ValueError("Image is None")

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image.copy()


def white_balance_gray_world(image_rgb):
    image = image_rgb.astype(np.float32)

    avg_r = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_b = np.mean(image[:, :, 2])

    avg_gray = (avg_r + avg_g + avg_b) / 3.0

    image[:, :, 0] *= avg_gray / max(avg_r, 1)
    image[:, :, 1] *= avg_gray / max(avg_g, 1)
    image[:, :, 2] *= avg_gray / max(avg_b, 1)

    return np.clip(image, 0, 255).astype(np.uint8)


def enhance_ecg_image(
    image,
    denoise_strength=3,
    sharp_strength=1.4,
    contrast_strength=2.0,
    gamma=1.0,
):
    rgb = ensure_rgb(image)

    denoise_strength = int(denoise_strength)
    sharp_strength = float(sharp_strength)
    contrast_strength = float(contrast_strength)
    gamma = float(gamma)

    # 1. Cân bằng màu giấy để giảm ám vàng/xám
    balanced = white_balance_gray_world(rgb)

    # 2. Khử nhiễu nhẹ, tránh làm mất nét sóng ECG
    if denoise_strength > 0:
        denoised = cv2.fastNlMeansDenoisingColored(
            balanced,
            None,
            denoise_strength,
            denoise_strength,
            7,
            21,
        )
    else:
        denoised = balanced

    # 3. Tăng tương phản kênh sáng L trong LAB
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=contrast_strength,
        tileGridSize=(8, 8),
    )
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    contrast = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    # 4. Gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        contrast = cv2.LUT(contrast, table)

    # 5. Làm nét unsharp mask
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.1)

    alpha = sharp_strength
    beta = -(sharp_strength - 1.0)

    sharp = cv2.addWeighted(contrast, alpha, blur, beta, 0)

    # 6. Ép vùng giấy sáng hơn một chút nhưng không làm cháy nét sóng
    hsv = cv2.cvtColor(sharp, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    hsv2 = cv2.merge((h, s, v))
    final = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

    return final


def segment_ecg_waveform(
    image,
    model_path=None,
    model_type="OpenCV",
    C_val=10,
    model=None,
):
    return enhance_ecg_image(
        image,
        denoise_strength=3,
        sharp_strength=1.4,
    )


def apply_waveform_grid_overlay(
    waveform_mask,
    grid_style="Clinical Pink Grid",
):
    return waveform_mask