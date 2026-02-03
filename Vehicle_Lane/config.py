import os
import pickle
import numpy as np
import cv2  # Cần thêm thư viện này để tính toán chuẩn xác

# --- ĐƯỜNG DẪN HỆ THỐNG ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tùy chỉnh session (0, 1...)
SESSION_DIR = os.path.join(PROJECT_ROOT, "Video_test", "BrnoCompSpeedSubset", "session_0")

VIDEO_PATH = "D:\\HProjecT\\DHL\\Video_test\\test.mp4" 
MASK_PATH = "D:\\HProjecT\\DHL\\Video_test\\video_mask.png"
# DATA_PKL_PATH = os.path.join(SESSION_DIR, "gt_data.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "Weights", "yolo12n.onnx") 

# --- CẤU HÌNH MẶC ĐỊNH ---
VIDEO_FPS = 25  # FPS mặc định 25  

# 4 điểm hình thang trên video gốc (Source Points)
SOURCE_POINTS = [[800, 410], [1125, 410], [1920, 850], [0, 850]]
TARGET_WIDTH = 800  # Tính toán dựa trên tỉ lệ 140m / 32m
TARGET_HEIGHT = 3500 # Tính toán dựa trên tỉ lệ 140m / 32m
METERS_PER_PIXEL = 0.04

TARGET_CLASSES = [2, 3, 5, 7] # Chỉ theo dõi xe: Car, Motorbike, Bus, Truck