import cv2
import numpy as np
import math
import csv
import os
import json  # <--- THÊM MỚI

# ================= CẤU HÌNH =================
IMAGE_PATH = '.\\test_imgs\\test_3.jpg'
CALIB_FILE = 'calibration.json'  # <--- File chứa thông số K, D vừa tạo
CSV_FILE_NAME = 'measurement_data.csv'

# Nhập độ dài 4 cạnh thực tế đã đo (Đơn vị: Mét)
L1 = 1.83  # Cạnh trên 
L2 = 3.72  # Cạnh phải 
L3 = 1.28  # Cạnh dưới
L4 = 4.74  # Cạnh trái
DIAG_13 = 4.9  # Đường chéo từ điểm 1 đến điểm 3
# ============================================

# Biến toàn cục
clicked_points = []
matrix_homography = None
scale_px_per_meter = 100 

def save_data_to_csv(ref_points, p_start, p_end, dist_px, dist_m):
    file_exists = os.path.isfile(CSV_FILE_NAME)
    try:
        with open(CSV_FILE_NAME, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                header = [
                    'Ref_P1', 'Ref_P2', 'Ref_P3', 'Ref_P4',
                    'Config_L1_m', 'Config_L2_m', 'Config_L3_m', 'Config_L4_m', 'Config_Diag_m',
                    'Measure_Start', 'Measure_End',
                    'Distance_Pixel', 'Distance_Calculated_Meter'
                ]
                writer.writerow(header)
            
            row = [
                str(ref_points[0]), str(ref_points[1]), 
                str(ref_points[2]), str(ref_points[3]),
                L1, L2, L3, L4, DIAG_13,
                str(p_start), str(p_end),
                round(dist_px, 2),
                round(dist_m, 4)
            ]
            writer.writerow(row)
            print(f"[CSV] Đã lưu: {dist_m:.2f}m vào file {CSV_FILE_NAME}")
    except Exception as e:
        print(f"Lỗi khi lưu CSV: {e}")
        
def get_quadrilateral_coords(l1, l2, l3, l4, d13):
    if l1 + l2 < d13 or abs(l1 - l2) > d13:
        print("LỖI: Đường chéo sai (không khớp với cạnh L1, L2)")
        return []
    
    p1 = (0.0, 0.0)
    p2 = (l1, 0.0)
    
    cos_alpha = (l1**2 + d13**2 - l2**2) / (2 * l1 * d13)
    alpha = math.acos(cos_alpha)
    p3 = (d13 * math.cos(alpha), d13 * math.sin(alpha))
    
    d = d13
    a = (l4**2 - l3**2 + d**2) / (2*d)
    h = math.sqrt(max(0, l4**2 - a**2))
    
    x0 = p1[0] + a * (p3[0] - p1[0]) / d
    y0 = p1[1] + a * (p3[1] - p1[1]) / d
    rx = -(p3[1] - p1[1]) / d
    ry = (p3[0] - p1[0]) / d
    
    p4_a = (x0 + h * rx, y0 + h * ry)
    p4_b = (x0 - h * rx, y0 - h * ry)
    
    def cross_product(vx, vy, px, py):
        return vx * py - vy * px
        
    if cross_product(p3[0], p3[1], p4_a[0], p4_a[1]) > 0:
        p4 = p4_a
    else:
        p4 = p4_b

    return [p1, p2, p3, p4]

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

def compute_homography(src_pts):
    global matrix_homography
    real_coords_meters = get_quadrilateral_coords(L1, L2, L3, L4, DIAG_13)
    if not real_coords_meters: return

    dst_pts = []
    for pt in real_coords_meters:
        dst_pts.append([pt[0] * scale_px_per_meter, pt[1] * scale_px_per_meter])
    
    dst_pts = np.float32(dst_pts)
    src_pts_arr = np.float32(src_pts)
    matrix_homography = cv2.getPerspectiveTransform(src_pts_arr, dst_pts)
    print("[OK] Đã thiết lập bản đồ sàn nhà chính xác!")

def calculate_distance(p1, p2):
    if matrix_homography is None: return 0.0
    pts = np.float32([p1, p2]).reshape(-1, 1, 2)
    trans_pts = cv2.perspectiveTransform(pts, matrix_homography)
    dist_px = np.linalg.norm(trans_pts[0][0] - trans_pts[1][0])
    return dist_px / scale_px_per_meter

# === HÀM MỚI: LOAD CALIB & UNDISTORT ===
def undistort_image_from_json(img, json_path):
    if not os.path.exists(json_path):
        print(f"[WARNING] Không tìm thấy '{json_path}'. Sẽ dùng ảnh gốc.")
        return img

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data['camera_matrix'])
        dist_coeffs = np.array(data['distortion_coefficients'])
        
        # Lấy độ phân giải lúc Calibrate từ file JSON
        # (Nếu file json cũ không có key này, bạn tự điền số cứng vào)
        if 'image_resolution' in data:
            calib_w, calib_h = data['image_resolution']
        else:
            # Fallback nếu json thiếu: Điền số từ log của bạn (810, 720)
            calib_w, calib_h = 810, 720 
            
        # --- BƯỚC QUAN TRỌNG: Resize ảnh về đúng cỡ Calibrate ---
        h_orig, w_orig = img.shape[:2]
        if w_orig != calib_w or h_orig != calib_h:
            print(f"[INFO] Resize ảnh từ {w_orig}x{h_orig} -> {calib_w}x{calib_h} để khớp Calibration.")
            img = cv2.resize(img, (calib_w, calib_h))
        # --------------------------------------------------------

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newcameramtx, (w,h), 5)
        
        undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        print("[INFO] Đã áp dụng sửa méo (Undistort) thành công.")
        return undistorted_img
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc file Calibration: {e}")
        return img
# ========================================

def main():
    global clicked_points, matrix_homography 
    
    img_orig = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img_orig is None:
        print(f"Lỗi: Không tìm thấy file ảnh {IMAGE_PATH}")
        return

    # --- BƯỚC 1: SỬA MÉO ẢNH TRƯỚC KHI LÀM BẤT CỨ CÁI GÌ ---
    img_orig = undistort_image_from_json(img_orig, CALIB_FILE)
    # -------------------------------------------------------

    # Resize để hiển thị cho vừa màn hình (giữ nguyên logic cũ của bạn)
    MAX_HEIGHT = 800
    MAX_WIDTH = 800 
    h, w = img_orig.shape[:2]
    scale_w = MAX_WIDTH / w
    scale_h = MAX_HEIGHT / h
    scale_display = min(scale_w, scale_h) 
    new_w = int(w * scale_display)
    new_h = int(h * scale_display)
    img_resized = cv2.resize(img_orig, (new_w, new_h))
    
    # Cần lưu lại tỉ lệ resize để khi click chuột (trên ảnh nhỏ) ta map về ảnh gốc (ảnh to undistorted)
    # Tuy nhiên trong logic hiện tại của bạn: Bạn tính Homography dựa trên tọa độ ĐÃ resize
    # Nên ma trận Homography sẽ đúng với ảnh resized. Điều này hoàn toàn OK.
    
    img_display = img_resized.copy()

    cv2.namedWindow("Test Distance")
    cv2.setMouseCallback("Test Distance", mouse_callback)

    print("\n--- HƯỚNG DẪN (Đã tích hợp Calibration) ---")
    print("BƯỚC 1: Click 4 điểm sàn nhà theo thứ tự: Trái trên -> Phải trên -> Phải dưới -> Trái dưới.")
    print("BƯỚC 2: Click vào đáy 2 vật thể bất kỳ để đo khoảng cách.")
    print("Phím 'r': Reset.")
    print("Phím 'q': Thoát.")
    
    measured_count = 0

    while True:
        img_show = img_display.copy()

        for i, pt in enumerate(clicked_points):
            color = (0, 0, 255) if i < 4 else (0, 255, 255)
            cv2.circle(img_show, pt, 5, color, -1)
            cv2.putText(img_show, str(i+1), (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if i > 0 and i < 4:
                cv2.line(img_show, clicked_points[i-1], pt, (0, 255, 0), 2)
            if i == 3:
                cv2.line(img_show, clicked_points[3], clicked_points[0], (0, 255, 0), 2)

        if len(clicked_points) == 4 and matrix_homography is None:
            compute_homography(clicked_points)
            print("\n>>> CHẾ ĐỘ ĐO KÍCH HOẠT <<<")

        if len(clicked_points) > 4:
            points_measurement = clicked_points[4:]
            for k in range(0, len(points_measurement) - 1, 2):
                p_start = points_measurement[k]
                p_end = points_measurement[k+1]
                
                dist_m = calculate_distance(p_start, p_end)
                dist_px = np.linalg.norm(np.array(p_start) - np.array(p_end))
                
                cv2.line(img_show, p_start, p_end, (0, 255, 255), 2)
                mid_x = (p_start[0] + p_end[0]) // 2
                mid_y = (p_start[1] + p_end[1]) // 2
                cv2.putText(img_show, f"{dist_m:.2f}m", (mid_x, mid_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                pair_index = k // 2
                if pair_index >= measured_count:
                    save_data_to_csv(clicked_points[:4], p_start, p_end, dist_px, dist_m)
                    measured_count += 1

        cv2.imshow("Test Distance", img_show)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        if key == ord('r'):
            clicked_points = []
            matrix_homography = None
            measured_count = 0
            img_display = img_resized.copy()
            print("\n--- ĐÃ RESET ---")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()