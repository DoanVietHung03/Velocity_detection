# validate_manual.py
import cv2
import numpy as np
import pickle
import config  # <--- Import config để lấy 4 điểm bạn vừa điền

def validate():
    print(f"--- KIỂM TRA ĐỘ CHÍNH XÁC DỰA TRÊN CONFIG.PY ---")
    
    # Load ground truth
    with open(config.DATA_PKL_PATH, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Lấy thông số từ CONFIG (Cái bạn vừa sửa)
    src_pts = np.float32(config.SOURCE_POINTS)
    target_w = config.TARGET_WIDTH
    target_h = config.TARGET_HEIGHT
    mpp = config.METERS_PER_PIXEL

    # Tạo ma trận biến đổi
    dst_pts = np.float32([
        [0, 0], [target_w, 0],
        [target_w, target_h], [0, target_h]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Tính sai số
    errors = []
    print(f"{'Thực tế (m)':<12} | {'Code đo (m)':<12} | {'Sai số (%)':<10}")
    print("-" * 40)

    for item in data['distanceMeasurement']:
        p1 = np.array(item['p1']).flatten()[:2]
        p2 = np.array(item['p2']).flatten()[:2]
        real_dist = item['distance']

        # Transform và đo
        pts_input = np.float32([p1, p2]).reshape(-1, 1, 2)
        pts_output = cv2.perspectiveTransform(pts_input, M).reshape(-1, 2)
        px_dist = np.linalg.norm(pts_output[0] - pts_output[1])
        
        estimated_dist = px_dist * mpp
        
        # Sai số
        err_percent = abs(estimated_dist - real_dist) / real_dist * 100
        errors.append(err_percent)

        print(f"{real_dist:<12.2f} | {estimated_dist:<12.2f} | {err_percent:<10.2f}%")

    print("-" * 40)
    avg_err = np.mean(errors)
    print(f"SAI SỐ TRUNG BÌNH: {avg_err:.2f}%")
    
    if avg_err < 5.0:
        print("✅ TUYỆT VỜI! Config của bạn đã khá chuẩn.")
    else:
        print("⚠️ Vẫn còn sai số. Hãy kiểm tra lại config.py xem đã copy đúng chưa.")

if __name__ == "__main__":
    validate()