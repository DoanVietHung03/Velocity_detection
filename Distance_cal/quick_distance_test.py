import cv2
import numpy as np
import math
import csv
import os

# ================= CẤU HÌNH KÍCH THƯỚC THỰC TẾ =================
# Tên file ảnh
IMAGE_PATH = 'D:\\HProjecT\\DHL\\Distance_cal\\test_3.jpg' 

# Nhập độ dài 4 cạnh đã đo (Đơn vị: Mét)
L1 = 1.83  # Cạnh trên 
L2 = 3.72  # Cạnh phải 
L3 = 1.28  # Cạnh dưới
L4 = 4.74  # Cạnh trái
DIAG_13 = 4.9  # Đường chéo từ điểm 1 đến điểm 3
# ===============================================================

# Biến toàn cục
clicked_points = []
matrix_homography = None
scale_px_per_meter = 100 # Quy ước: 100 pixel trên bản đồ ảo = 1 mét thực
CSV_FILE_NAME = '.\\Distance_cal\\measurement_data.csv'

def save_data_to_csv(ref_points, p_start, p_end, dist_px, dist_m):
    """
    Lưu thông tin vào file CSV với định dạng cột gọn hơn.
    """
    file_exists = os.path.isfile(CSV_FILE_NAME)
    
    try:
        with open(CSV_FILE_NAME, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Tạo header nếu file chưa tồn tại
            if not file_exists:
                header = [
                    'Ref_P1', 'Ref_P2', 'Ref_P3', 'Ref_P4', # Gom tọa độ (x,y) vào 1 cột
                    'Config_L1_m', 'Config_L2_m', 'Config_L3_m', 'Config_L4_m', 'Config_Diag_m',
                    'Measure_Start', 'Measure_End',         # Gom tọa độ đo
                    'Distance_Pixel', 'Distance_Calculated_Meter'
                ]
                writer.writerow(header)
            
            # Chuẩn bị dữ liệu ghi dòng (chuyển tuple thành string)
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
    """
    Trả về tọa độ 4 điểm của tứ giác bất kỳ từ độ dài 4 cạnh.
    Tự động tính toán để hình dáng hợp lý nhất.
    """
    # 1. Kiểm tra tính hợp lệ của tam giác P1-P2-P3
    if l1 + l2 < d13 or abs(l1 - l2) > d13:
        print("LỖI: Đường chéo sai (không khớp với cạnh L1, L2)")
        return []
    
    # 2. Đặt P1 tại gốc (0,0), P2 nằm trên trục hoành
    p1 = (0.0, 0.0)
    p2 = (l1, 0.0)
    
    # 3. Tìm P3 (Giao điểm 2 đường tròn: Tâm P1 bk d13, Tâm P2 bk l2)
    # Dùng định lý hàm số Cosin cho tam giác P1P2P3 để tìm góc tại P1
    # l2^2 = l1^2 + d13^2 - 2*l1*d13*cos(alpha)
    cos_alpha = (l1**2 + d13**2 - l2**2) / (2 * l1 * d13)
    alpha = math.acos(cos_alpha)
    
    p3_x = d13 * math.cos(alpha)
    p3_y = d13 * math.sin(alpha)
    p3 = (p3_x, p3_y)
    
    # 4. Tìm P4 (Giao điểm 2 đường tròn: Tâm P1 bk l4, Tâm P3 bk l3)
    # Khoảng cách giữa 2 tâm P1 và P3 chính là d13
    d = d13
    
    # Công thức giao điểm 2 đường tròn
    a = (l4**2 - l3**2 + d**2) / (2*d)
    h = math.sqrt(max(0, l4**2 - a**2))
    
    # Tọa độ điểm cơ sở (chiếu vuông góc) trên đường nối P1-P3
    x0 = p1[0] + a * (p3[0] - p1[0]) / d
    y0 = p1[1] + a * (p3[1] - p1[1]) / d
    
    # Vector pháp tuyến đơn vị của đoạn P1-P3
    rx = -(p3[1] - p1[1]) / d
    ry = (p3[0] - p1[0]) / d
    
    # Có 2 giao điểm P4. Chọn điểm nằm khác phía với P2 so với đường chéo P1-P3 (Tứ giác lồi)
    # Cách đơn giản: P4 thường có x < x3 và y tương đương hoặc lớn hơn
    p4_a = (x0 + h * rx, y0 + h * ry)
    p4_b = (x0 - h * rx, y0 - h * ry)
    
    # Kiểm tra tích có hướng (Cross Product) để xác định phía
    # Vector P1->P3
    v13_x = p3[0]
    v13_y = p3[1]
    
    # Hàm check phía
    def cross_product(vx, vy, px, py):
        return vx * py - vy * px
        
    # P2 nằm ở phía nào của P1-P3?
    cp_p2 = cross_product(v13_x, v13_y, p2[0], p2[1]) # p2_y=0 -> v13_x*0 - v13_y*l1 = âm
    
    # P4 phải nằm khác phía -> cross product phải dương
    if cross_product(v13_x, v13_y, p4_a[0], p4_a[1]) > 0:
        p4 = p4_a
    else:
        p4 = p4_b

    print(f"-> Đã khóa cứng hình học sàn nhà theo đường chéo {d13}m.")
    return [p1, p2, p3, p4]

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

def compute_homography(src_pts):
    global matrix_homography
    
    # 1. Tính toán tọa độ thực tế của 4 điểm sàn nhà
    real_coords_meters = get_quadrilateral_coords(L1, L2, L3, L4, DIAG_13)
    
    if not real_coords_meters: return

    # 2. Chuyển đổi mét sang pixel ảo
    dst_pts = []
    for pt in real_coords_meters:
        dst_pts.append([pt[0] * scale_px_per_meter, pt[1] * scale_px_per_meter])
    
    dst_pts = np.float32(dst_pts)
    src_pts_arr = np.float32(src_pts)
    
    # 3. Tính ma trận
    matrix_homography = cv2.getPerspectiveTransform(src_pts_arr, dst_pts)
    print("[OK] Đã thiết lập bản đồ sàn nhà chính xác!")

def calculate_distance(p1, p2):
    if matrix_homography is None: return 0.0
    
    # Chuyển đổi 2 điểm click sang tọa độ thực tế
    pts = np.float32([p1, p2]).reshape(-1, 1, 2)
    trans_pts = cv2.perspectiveTransform(pts, matrix_homography)
    
    real_p1 = trans_pts[0][0]
    real_p2 = trans_pts[1][0]
    
    # Tính khoảng cách Euclidean
    dist_px = np.linalg.norm(real_p1 - real_p2)
    dist_m = dist_px / scale_px_per_meter
    return dist_m

def main():
    global clicked_points, matrix_homography 
    
    # Load ảnh
    img_orig = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img_orig is None:
        print(f"Lỗi: Không tìm thấy file ảnh {IMAGE_PATH}")
        return

    # --- FIX: Resize ảnh thông minh ---
    MAX_HEIGHT = 800
    MAX_WIDTH = 800 

    h, w = img_orig.shape[:2]
    
    scale_w = MAX_WIDTH / w
    scale_h = MAX_HEIGHT / h
    scale_display = min(scale_w, scale_h) 

    new_w = int(w * scale_display)
    new_h = int(h * scale_display)
    
    img_resized = cv2.resize(img_orig, (new_w, new_h))
    print(f"Kích thước hiển thị: {new_w}x{new_h}")
    # ----------------------------------
    
    img_display = img_resized.copy()

    cv2.namedWindow("Test Distance")
    cv2.setMouseCallback("Test Distance", mouse_callback)

    print("\n--- HƯỚNG DẪN ---")
    print("BƯỚC 1: Click 4 điểm sàn nhà theo thứ tự:")
    print("Trái trên -> Phải trên -> Phải dưới -> Trái dưới.")
    print("BƯỚC 2: Click vào đáy 2 vật thể bất kỳ để đo khoảng cách.")
    print("Phím 'r': Reset.")
    print("Phím 'q': Thoát.")
    
    measured_count = 0

    while True:
        img_show = img_display.copy()

        # Vẽ các điểm đã click
        for i, pt in enumerate(clicked_points):
            color = (0, 0, 255) if i < 4 else (0, 255, 255) # Đỏ (Setup) - Vàng (Đo)
            cv2.circle(img_show, pt, 5, color, -1)
            cv2.putText(img_show, str(i+1), (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Vẽ khung ROI
            if i > 0 and i < 4:
                cv2.line(img_show, clicked_points[i-1], pt, (0, 255, 0), 2)
            if i == 3:
                cv2.line(img_show, clicked_points[3], clicked_points[0], (0, 255, 0), 2)

        # KHI ĐỦ 4 ĐIỂM -> TÍNH MATRIX
        if len(clicked_points) == 4 and matrix_homography is None:
            compute_homography(clicked_points)
            print("\n>>> CHẾ ĐỘ ĐO KÍCH HOẠT <<<")
            print("Hãy click 2 điểm để đo khoảng cách.")

        # KHI ĐO (Cứ mỗi 2 điểm sau 4 điểm đầu)
        if len(clicked_points) > 4:
            points_measurement = clicked_points[4:]
            
            # Vẽ từng cặp điểm đo
            for k in range(0, len(points_measurement) - 1, 2):
                p_start = points_measurement[k]
                p_end = points_measurement[k+1]
                
                dist_m = calculate_distance(p_start, p_end)
                
                # Tính khoảng cách Pixel trên ảnh
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
        if key == ord('r'): # Reset
            clicked_points = []
            matrix_homography = None
            measured_count = 0
            img_display = img_resized.copy()
            print("\n--- ĐÃ RESET ---")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()