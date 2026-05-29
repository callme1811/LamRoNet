import cv2
import numpy as np
import os

def order_points(pts):
    """
    Sorts 4 coordinates in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    # Top-left has the smallest sum, bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has the smallest difference, bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """
    Applies homography transformation to warp perspective using 4 corner points.
    """
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
    """
    Flattens the ECG paper in the image using automatic YOLOv8-seg results.
    """
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

def detect_paper_corners(img_path, model_path=None):
    """
    Uses YOLOv8-seg model to automatically detect the four corners of the ECG paper.
    Returns 4 coordinate pairs (TL, TR, BR, BL) or None if detection fails.
    """
    if not model_path or not os.path.exists(model_path):
        return None
        
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model(img_path)
        result = results[0]
        
        if result.masks is None or len(result.masks) == 0:
            return None
            
        points = result.masks.xy[0]
        if len(points) < 4:
            return None
            
        contour = np.array(points, dtype=np.int32)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
        else:
            hull = cv2.convexHull(contour)
            pts = hull.reshape(-1, 2)
            
        return order_points(pts)
    except Exception as e:
        print(f"YOLOv8-seg corner detection error: {e}")
        return None