import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped

def flatten_ecg_paper(image, yolo_results):
    result = yolo_results[0]

    if result.masks is None:
        raise ValueError("Không tìm thấy mask tờ ECG.")

    mask = result.masks.xy[0]
    pts = np.array(mask, dtype=np.float32)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    flattened = four_point_transform(image, box)

    return flattened