import cv2
import numpy as np

def resize_mask_to_frame(mask, frame):
    h, w = frame.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask

def is_inside_lane(bbox, mask):
    # Logic kiểm tra điểm chân xe
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    cy = y2 
    
    h, w = mask.shape
    # Kẹp giá trị để không bị lỗi index out of bound
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))
    
    # Giả sử mask trắng (>127) là làn đường
    return mask[cy, cx] > 127