import os
import pickle
import numpy as np
import cv2  # Cần thêm thư viện này để tính toán chuẩn xác

# --- ĐƯỜNG DẪN HỆ THỐNG ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tùy chỉnh session (0, 1...)
SESSION_DIR = os.path.join(PROJECT_ROOT, "Video_test", "BrnoCompSpeedSubset", "session_0")

VIDEO_PATH = os.path.join(SESSION_DIR, "video.avi") 
MASK_PATH = os.path.join(SESSION_DIR, "video_mask.png")
DATA_PKL_PATH = os.path.join(SESSION_DIR, "gt_data.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "Weights", "yolo11n.onnx") 

# --- CẤU HÌNH MẶC ĐỊNH ---
VIDEO_FPS = 25  # Nếu không rõ fps, để mặc định 25  

# 4 điểm hình thang trên video gốc (Source Points)
SOURCE_POINTS = [[390, 634], [1043, 637], [1370, 855], [484, 871]]
METERS_PER_PIXEL = 0.01036
TARGET_WIDTH = 1000
TARGET_HEIGHT = 1375

TARGET_CLASSES = [2, 3, 5, 7] # Chỉ theo dõi xe: Car, Motorbike, Bus, Truck