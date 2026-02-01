import cv2
import numpy as np
from ultralytics import YOLO
import torch

import config
import utils
from transformer import ViewTransformer
from speed_estimator import SpeedEstimator

def main():
    # 1. Init Modules
    model = YOLO(config.MODEL_PATH)
    
    # Khởi tạo Transformer (Cần điền đúng SOURCE_POINTS trong config.py)
    transformer = ViewTransformer(
        source_points=config.SOURCE_POINTS,
        target_width=config.TARGET_WIDTH,
        target_height=config.TARGET_HEIGHT
    )
    
    # Khởi tạo bộ ước lượng tốc độ
    speed_estimator = SpeedEstimator(fps=config.VIDEO_FPS, meters_per_pixel=config.METERS_PER_PIXEL)

    # 2. Load Video & Mask
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    mask_img = cv2.imread(config.MASK_PATH, 0)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # Resize frame và Mask (Quan trọng)
        # Giả sử resize về 1280x720 hoặc giữ nguyên, nếu resize thì nhớ resize mask cho đúng (gt_data.pkl cũng phải tương ứng)
        # frame = cv2.resize(frame, (960, 540)) 
        mask_img = utils.resize_mask_to_frame(mask_img, frame)

        # 3. YOLO Tracking (Quan trọng: dùng track thay vì predict)
        # persist=True giúp giữ ID qua các frame
        results = model.track(frame, classes=config.TARGET_CLASSES, persist=True, verbose=False, tracker="bytetrack.yaml")

        # Vẽ Mask lên frame
        colored_mask = np.zeros_like(frame)
        colored_mask[mask_img > 127] = [0, 255, 0]
        frame = cv2.addWeighted(frame, 1, colored_mask, 0.3, 0)

        if results[0].boxes.id is not None:
            # Lấy data từ YOLO
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            cls_ids = results[0].boxes.cls.int().cpu().numpy()
            
            # --- GIAI ĐOẠN TRANSFORMATION ---
            # Lấy danh sách các điểm đáy của xe (chân xe) để transform
            bottom_centers = []
            for box in boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = y2 
                bottom_centers.append([cx, cy])
            
            # Transform sang BEV một lần cho nhanh (Vectorization)
            points_bev = transformer.transform_points(bottom_centers)
            
            # --- VÒNG LẶP VẼ VÀ TÍNH TOÁN ---
            for box, track_id, cls_id, point_bev in zip(boxes, track_ids, cls_ids, points_bev):
                x1, y1, x2, y2 = map(int, box)
                
                # A. Kiểm tra làn đường
                in_lane = utils.is_inside_lane(box, mask_img)
                color = (0, 0, 255) if in_lane else (255, 0, 0)
                
                # B. Tính tốc độ
                speed = speed_estimator.update(track_id, point_bev, frame_count)
                
                # C. Hiển thị
                label = f"ID:{track_id} {model.names[cls_id]}"
                if in_lane:
                    label += f" | {int(speed)} km/h"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Traffic Monitor', frame)
        
        # Nhấn 's' để skip nhanh nếu debug
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()