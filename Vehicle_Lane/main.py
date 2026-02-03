import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch

import config
from transformer import ViewTransformer
from speed_estimator import SpeedEstimator

def main():
    # 1. Kiểm tra thiết bị
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"--- Đang chạy trên: {device} ---")

    # 2. Khởi tạo Model & Modules
    # Lưu ý: Nếu config.MODEL_PATH là .onnx thì task='detect' là bắt buộc
    model = YOLO(config.MODEL_PATH, task='detect')
    
    # Lấy thông tin video để khởi tạo các công cụ vẽ
    video_info = sv.VideoInfo.from_video_path(config.VIDEO_PATH)
    
    # Module biến đổi góc nhìn (Bird's Eye View)
    transformer = ViewTransformer(
        source_points=config.SOURCE_POINTS,
        target_width=config.TARGET_WIDTH,
        target_height=config.TARGET_HEIGHT
    )
    
    # Module ước lượng tốc độ (Dùng bản EMA tối ưu từ speed_estimator.py)
    speed_estimator = SpeedEstimator(fps=video_info.fps, meters_per_pixel=config.METERS_PER_PIXEL)

    # 3. Cấu hình Supervision
    # a. Tạo vùng đo tốc độ (Dùng chính tọa độ hình thang trong config)
    # Đây chính là "Mask" của bạn, nhưng dưới dạng vector, chính xác tuyệt đối
    polygon = np.array(config.SOURCE_POINTS)
    zone = sv.PolygonZone(polygon=polygon)
    
    # b. Các công cụ vẽ (Annotators)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.ColorPalette.DEFAULT
    )
    
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=10,
        text_position=sv.Position.TOP_CENTER,
        color=sv.ColorPalette.DEFAULT
    )
    
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=video_info.fps * 2, # Đuôi dài 2 giây
        position=sv.Position.BOTTOM_CENTER, # Vẽ đuôi từ bánh xe (chuẩn nhất)
        color=sv.ColorPalette.DEFAULT
    )
    
    # Vẽ vùng zone lên màn hình để dễ căn chỉnh (có thể tắt nếu muốn)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.GREEN,
        thickness=2,
        text_thickness=0, # Không cần hiện số đếm
        text_scale=0
    )

    # 4. Chạy Video (Vòng lặp tối ưu)
    # frame_generator giúp code ngắn gọn, tự xử lý việc đọc video
    frame_generator = sv.get_video_frames_generator(config.VIDEO_PATH)
    
    # Tạo cửa sổ hiển thị có thể thay đổi kích thước
    cv2.namedWindow("Supervision Speed Cam", cv2.WINDOW_NORMAL)

    print("--- BẮT ĐẦU XỬ LÝ ---")
    
    for i, frame in enumerate(frame_generator):
        # A. Tracking bằng YOLO
        # persist=True để giữ ID xe qua các frame
        result = model.track(
            frame,
            classes=config.TARGET_CLASSES,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
            device=device
        )[0]

        # B. Chuyển đổi kết quả sang Supervision Detections (Bước quan trọng)
        detections = sv.Detections.from_ultralytics(result)
        
        # Nếu không có xe nào thì bỏ qua vòng này để tránh lỗi
        if detections.tracker_id is None:
            cv2.imshow("Supervision Speed Cam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # C. Lọc xe nằm trong Làn đường (Zone)
        # mask_zone là mảng True/False: True nếu xe nằm trong hình thang
        mask_zone = zone.trigger(detections=detections)
        
        # Chỉ lấy các xe nằm TRONG làn để tính tốc độ
        # Xe ở ngoài lề sẽ bị loại bỏ khỏi danh sách 'valid_detections'
        valid_detections = detections[mask_zone]

        # D. Tính toán tốc độ cho các xe hợp lệ
        labels = []
        
        # Lấy tọa độ chân xe (Bottom Center) để tính toán chính xác nhất
        # Thay thế hoàn toàn đoạn code tính 'cx, cy' thủ công trong utils.py
        points = valid_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        # Biến đổi sang góc nhìn từ trên cao (BEV)
        points_bev = transformer.transform_points(points)

        for tid, point_bev in zip(valid_detections.tracker_id, points_bev):
            # Gọi hàm update trong speed_estimator.py của bạn
            speed = speed_estimator.update(tid, point_bev, i)
            
            # Tạo nhãn hiển thị: ID + Tốc độ
            labels.append(f"#{tid} {int(speed)} km/h")

        # E. Vẽ lên màn hình (Thứ tự vẽ quan trọng: Dưới lên trên)
        
        # 1. Vẽ vùng đo (hình thang xanh lá)
        frame = zone_annotator.annotate(scene=frame)
        
        # 2. Vẽ vết chuyển động (đuôi xe) - Vẽ cho TẤT CẢ xe (cả trong lẫn ngoài zone cho đẹp)
        frame = trace_annotator.annotate(scene=frame, detections=detections)
        
        # 3. Vẽ khung xe (Bounding Box) - Chỉ vẽ xe TRONG zone (đang được đo tốc độ)
        frame = box_annotator.annotate(scene=frame, detections=valid_detections)
        
        # 4. Vẽ nhãn thông tin (ID + KM/H)
        frame = label_annotator.annotate(
            scene=frame, 
            detections=valid_detections, 
            labels=labels
        )

        # Hiển thị
        cv2.imshow("Supervision Speed Cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("--- KẾT THÚC ---")

if __name__ == "__main__":
    main()