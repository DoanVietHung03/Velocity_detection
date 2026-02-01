import cv2
import numpy as np
import pickle
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tùy chỉnh session nếu cần
SESSION_DIR = os.path.join(PROJECT_ROOT, "Video_test", "BrnoCompSpeedSubset", "session_0")

VIDEO_PATH = os.path.join(SESSION_DIR, "video.avi")
DATA_PKL_PATH = os.path.join(SESSION_DIR, "gt_data.pkl")

# Biến toàn cục
clicked_points = []
frame_display = None
calibration_data = []

def load_ground_truth():
    global calibration_data
    print(f"Loading data from: {DATA_PKL_PATH}")
    try:
        with open(DATA_PKL_PATH, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            if 'distanceMeasurement' in data:
                for item in data['distanceMeasurement']:
                    p1 = np.array(item['p1']).flatten()[:2]
                    p2 = np.array(item['p2']).flatten()[:2]
                    dist = item['distance']
                    calibration_data.append((p1, p2, dist))
        print(f"--> Đã load {len(calibration_data)} đoạn thẳng mẫu.")
    except Exception as e:
        print(f"Lỗi đọc pickle: {e}")

def optimize_transform(source_points):
    """
    Chạy vòng lặp để tìm Tỷ lệ khung hình (Aspect Ratio) tốt nhất
    """
    src_pts = np.float32(source_points)
    target_w = 200 # Fix chiều rộng
    
    best_error = 9999
    best_mpp = 0
    best_target_h = 200
    
    # Thử các tỷ lệ chiều cao khác nhau từ ngắn đến dài
    # Ratio = Height / Width. Thử từ 0.5 (hình chữ nhật nằm ngang) đến 20 (dọc dài)
    print("Đang tối ưu hóa tỷ lệ...", end="")
    
    possible_ratios = np.linspace(0.5, 15.0, 150) # Quét 150 mức tỷ lệ
    
    for ratio in possible_ratios:
        target_h = int(target_w * ratio)
        
        dst_pts = np.float32([
            [0, 0], [target_w, 0],
            [target_w, target_h], [0, target_h]
        ])
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Tính toán sai số cho tỷ lệ này
        mpp_values = []
        for p1, p2, real_dist in calibration_data:
            pts_input = np.float32([p1, p2]).reshape(-1, 1, 2)
            pts_output = cv2.perspectiveTransform(pts_input, M).reshape(-1, 2)
            px_dist = np.linalg.norm(pts_output[0] - pts_output[1])
            
            if px_dist > 1:
                mpp_values.append(real_dist / px_dist)
        
        if not mpp_values: continue
        
        # Độ lệch chuẩn (Standard Deviation) càng thấp nghĩa là MPP càng đồng nhất
        # Giữa các đoạn dọc và ngang
        avg_mpp = np.mean(mpp_values)
        std_dev = np.std(mpp_values)
        
        # Metric đánh giá: Hệ số biến thiên (CV)
        error_score = std_dev / avg_mpp
        
        if error_score < best_error:
            best_error = error_score
            best_mpp = avg_mpp
            best_target_h = target_h

    print(" Xong!")
    
    # Tính lại sai số cụ thể với tham số tốt nhất vừa tìm được
    final_dst_pts = np.float32([
        [0, 0], [target_w, 0],
        [target_w, best_target_h], [0, best_target_h]
    ])
    M_final = cv2.getPerspectiveTransform(src_pts, final_dst_pts)
    
    errors_percent = []
    for p1, p2, real_dist in calibration_data:
        pts_input = np.float32([p1, p2]).reshape(-1, 1, 2)
        pts_output = cv2.perspectiveTransform(pts_input, M_final).reshape(-1, 2)
        px_dist = np.linalg.norm(pts_output[0] - pts_output[1])
        
        est_dist = px_dist * best_mpp
        err = abs(est_dist - real_dist) / real_dist * 100
        errors_percent.append(err)
        
    avg_error_percent = np.mean(errors_percent)
    
    return avg_error_percent, best_mpp, best_target_h

def mouse_callback(event, x, y, flags, param):
    global clicked_points, frame_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("CALIBRATION TOOL V2", frame_display)
            
            if len(clicked_points) == 4:
                cv2.polylines(frame_display, [np.array(clicked_points)], True, (0, 255, 0), 2)
                cv2.imshow("CALIBRATION TOOL V2", frame_display)
                
                print("\n--- KẾT QUẢ ---")
                err, mpp, best_h = optimize_transform(clicked_points)
                print(f"✅ Sai số sau khi tối ưu: {err:.2f}%")
                print(f"📏 MPP chuẩn: {mpp:.5f}")
                print(f"🖼️ Target Height: {best_h} (Width=200)")
                
                if err < 5.0:
                    print("\n--- COPY VÀO CONFIG.PY ---")
                    print(f"SOURCE_POINTS = {clicked_points}")
                    print(f"METERS_PER_PIXEL = {mpp:.5f}")
                    print(f"TARGET_WIDTH = 200")
                    print(f"TARGET_HEIGHT = {best_h}")
                    print("--------------------------")
                else:
                    print("\n⚠️ Sai số vẫn hơi cao. Hãy thử chọn 4 điểm khác chuẩn hơn.")

def main():
    global frame_display, clicked_points
    
    load_ground_truth()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret: return
    cap.release()
    
    frame_display = frame.copy()
    # Vẽ các đoạn thẳng GT để user dễ nhìn
    for p1, p2, dist in calibration_data:
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(frame_display, pt1, pt2, (255, 255, 0), 2)

    print("\n--- HƯỚNG DẪN V2 ---")
    print("Click 4 điểm bao quanh vùng vạch kẻ đường màu xanh lơ.")
    print("Thứ tự: Trái-Trên -> Phải-Trên -> Phải-Dưới -> Trái-Dưới")
    
    cv2.imshow("CALIBRATION TOOL V2", frame_display)
    cv2.setMouseCallback("CALIBRATION TOOL V2", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()