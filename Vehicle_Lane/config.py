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
MODEL_PATH = os.path.join(PROJECT_ROOT, "Weights", "yolo11n.pt") 

# --- CẤU HÌNH MẶC ĐỊNH ---
VIDEO_FPS = 30  

# 4 điểm hình thang trên video gốc (Source Points)
SOURCE_POINTS = [[413, 633], [1017, 638], [1368, 857], [493, 870]]
METERS_PER_PIXEL = 0.05035
TARGET_WIDTH = 200
TARGET_HEIGHT = 275

TARGET_CLASSES = [2, 3, 5, 7] 

# # --- LOGIC LOAD DATA VÀ CALIBRATE TỰ ĐỘNG ---
# print(f"Loading data from: {DATA_PKL_PATH}")
# try:
#     with open(DATA_PKL_PATH, 'rb') as f:
#         # Load data (fix lỗi python 2/3)
#         data = pickle.load(f, encoding='latin1')
        
#         # 1. Lấy FPS
#         if 'fps' in data:
#             VIDEO_FPS = data['fps']
#             print(f"✅ Auto-detected FPS: {VIDEO_FPS}")
            
#         # 2. Tìm vùng quan sát (ROI) từ dữ liệu gốc
#         if 'distanceMeasurement' in data and len(data['distanceMeasurement']) > 0:
#             all_points = []
#             for item in data['distanceMeasurement']:
#                 # Lấy p1, p2 (bỏ trục z)
#                 all_points.append(np.array(item['p1']).flatten()[:2])
#                 all_points.append(np.array(item['p2']).flatten()[:2])
            
#             pts = np.array(all_points)
#             if len(pts) > 0:
#                 x_min, y_min = pts.min(axis=0)
#                 x_max, y_max = pts.max(axis=0)
                
#                 # Định nghĩa 4 điểm nguồn (Source Points)
#                 SOURCE_POINTS = [
#                     [x_min, y_min], [x_max, y_min], 
#                     [x_max, y_max], [x_min, y_max]
#                 ]
#                 print(f"✅ Auto-defined ROI: {SOURCE_POINTS}")

#                 # Tự động điều chỉnh tỷ lệ khung hình BEV cho đỡ bị méo
#                 # (Ước lượng sơ bộ)
#                 src_w = x_max - x_min
#                 src_h = y_max - y_min
#                 TARGET_HEIGHT = int(TARGET_WIDTH * (src_h / src_w)) * 2 # Nhân 2 để kéo dài đường cho dễ nhìn
                
#                 # ====================================================
#                 # 🔴 BƯỚC QUAN TRỌNG NHẤT: RE-CALIBRATE MPP CHO BEV
#                 # ====================================================
#                 # Tạo ma trận biến đổi giả lập giống hệt Transformer
#                 src_pts = np.float32(SOURCE_POINTS)
#                 dst_pts = np.float32([
#                     [0, 0], [TARGET_WIDTH, 0],
#                     [TARGET_WIDTH, TARGET_HEIGHT], [0, TARGET_HEIGHT]
#                 ])
#                 M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
#                 # Tính lại MPP dựa trên khoảng cách TRONG BEV
#                 mpp_bev_values = []
#                 for item in data['distanceMeasurement']:
#                     p1_real = np.array(item['p1']).flatten()[:2]
#                     p2_real = np.array(item['p2']).flatten()[:2]
#                     dist_meters = item['distance']
                    
#                     # Transform điểm sang BEV
#                     pts_input = np.float32([p1_real, p2_real]).reshape(-1, 1, 2)
#                     pts_output = cv2.perspectiveTransform(pts_input, M).reshape(-1, 2)
                    
#                     # Đo khoảng cách pixel trong BEV
#                     dist_pixels_bev = np.linalg.norm(pts_output[0] - pts_output[1])
                    
#                     if dist_pixels_bev > 1: # Tránh chia cho 0
#                         # 1 Pixel BEV = Bao nhiêu mét?
#                         mpp_bev_values.append(dist_meters / dist_pixels_bev)
                
#                 if mpp_bev_values:
#                     METERS_PER_PIXEL = np.mean(mpp_bev_values)
#                     print(f"✅ FIXED Meters Per Pixel (BEV scale): {METERS_PER_PIXEL:.5f}")
#                     # Giá trị này thường khoảng 0.05 - 0.2 tùy video

# except Exception as e:
#     import traceback
#     print(f"⚠️ Warning: Lỗi config. Chi tiết:")
#     traceback.print_exc()