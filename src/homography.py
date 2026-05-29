import cv2
import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def detect_paper_corners(image_or_path, model_path=None):
    if isinstance(image_or_path, str):
        image_bgr = cv2.imread(image_or_path)
        if image_bgr is None:
            return None
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image = image_or_path.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    h, w = gray.shape[:2]

    # Nếu contour quá nhỏ thì bỏ qua
    if area < 0.05 * w * h:
        return None

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)

    return order_points(box)


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = int(max(height_right, height_left))

    if max_width <= 20 or max_height <= 20:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC
    )

    return warped


def flatten_ecg_paper(image):
    corners = detect_paper_corners(image)

    if corners is None:
        return image

    return four_point_transform(image, corners)
