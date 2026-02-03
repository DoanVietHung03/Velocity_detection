import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import deque

import config
import utils
from transformer import ViewTransformer
from speed_estimator import SpeedEstimator

def main():
    # 1. Kiểm tra thiết bị phần cứng
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CẢNH BÁO: Đang chạy trên CPU! Sẽ rất chậm.")
    else:
        print(f"Đang chạy trên GPU: {torch.cuda.get_device_name(0)}")
        
    # 2. Khởi tạo các Modules
    model = YOLO(config.MODEL_PATH)
    
    transformer = ViewTransformer(
        source_points=config.SOURCE_POINTS,
        target_width=config.TARGET_WIDTH,
        target_height=config.TARGET_HEIGHT
    )
    
    # SpeedEstimator giờ đây đã tích hợp Kalman Filter bên trong
    speed_estimator = SpeedEstimator(fps=config.VIDEO_FPS, meters_per_pixel=config.METERS_PER_PIXEL)

    # 3. Quản lý lịch sử Bounding Box để khử Jitter
    # Lưu trữ 5 frame gần nhất của mỗi track_id để tính trung bình tọa độ
    bbox_history = {} 

    # 4. Load Video & Mask
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    mask_img = cv2.imread(config.MASK_PATH, 0)

    ret, first_frame = cap.read()
    if not ret: 
        print("Lỗi: Không thể đọc video.")
        return
        
    mask_img = utils.resize_mask_to_frame(mask_img, first_frame)
    
    # Tạo overlay màu cho mask (hiển thị vùng nhận diện)
    mask_overlay = np.zeros_like(first_frame)
    mask_overlay[mask_img > 127] = [0, 255, 0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    frame_count = 0

    print("--- BẮT ĐẦU XỬ LÝ TRAFFIC MONITOR (OPTIMIZED) ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # Hiển thị vùng mask nhẹ trên frame gốc
        frame_display = cv2.addWeighted(frame, 1, mask_overlay, 0.2, 0)

        # 5. YOLO Tracking với ByteTrack
        results = model.track(
            frame,
            classes=config.TARGET_CLASSES,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
            device=device
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            cls_ids = results[0].boxes.cls.int().cpu().numpy()
            
            # --- GIAI ĐOẠN XỬ LÝ TỌA ĐỘ (JITTER REDUCTION) ---
            current_stable_points = []
            for box, tid in zip(boxes, track_ids):
                # Lưu lịch sử 5 frame để làm mượt box
                if tid not in bbox_history:
                    bbox_history[tid] = deque(maxlen=5)
                bbox_history[tid].append(box)
                
                # Tính trung bình tọa độ Box (Moving Average)
                avg_box = np.mean(bbox_history[tid], axis=0)
                ax1, ay1, ax2, ay2 = avg_box
                
                # Lấy điểm chân xe từ Box đã làm mượt
                cx = (ax1 + ax2) / 2
                cy = ay2 # Điểm tiếp giáp mặt đất
                current_stable_points.append([cx, cy])
            
            # --- GIAI ĐOẠN BIẾN ĐỔI HÌNH HỌC (BEV) ---
            points_bev = transformer.transform_points(current_stable_points)
            
            # --- GIAI ĐOẠN TÍNH TOÁN VÀ HIỂN THỊ ---
            for i, (tid, cls_id) in enumerate(zip(track_ids, cls_ids)):
                # Tọa độ box gốc (để vẽ)
                x1, y1, x2, y2 = map(int, boxes[i])
                point_bev = points_bev[i]
                
                # A. Kiểm tra xe có trong làn đường/vùng đo không
                in_lane = utils.is_inside_lane(boxes[i], mask_img)
                color = (0, 255, 0) if in_lane else (0, 0, 255) # Xanh nếu trong làn, đỏ nếu ngoài
                
                # B. Ước lượng tốc độ (Đã có Kalman Filter & Sliding Window bên trong)
                speed_kmh = speed_estimator.update(tid, point_bev, frame_count)
                
                # C. Vẽ Bounding Box và Thông tin
                label = f"ID:{tid} {model.names[cls_id]}"
                if in_lane and speed_kmh > 0:
                    label += f" | {int(speed_kmh)} km/h"
                
                # Vẽ khung xe
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ nền cho chữ để dễ đọc
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_display, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(frame_display, label, (x1, y1 - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 6. Hiển thị kết quả
        cv2.imshow('Optimized Traffic Speed Monitor', frame_display)
        
        # Dọn dẹp bộ nhớ bbox_history cho các ID đã biến mất (Tối ưu RAM)
        active_ids = set(track_ids) if results[0].boxes.id is not None else set()
        for tid in list(bbox_history.keys()):
            if tid not in active_ids and frame_count % 100 == 0: # Kiểm tra mỗi 100 frame
                del bbox_history[tid]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- KẾT THÚC ---")

if __name__ == "__main__":
    main()