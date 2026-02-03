import math
from collections import defaultdict, deque
import numpy as np

class KalmanFilter:
    """Bộ lọc Kalman đơn giản cho mô hình vận tốc không đổi (Constant Velocity)"""
    def __init__(self, dt=1/25, process_noise=1e-2, measurement_noise=1e-1):
        # Trạng thái: [x, y, vx, vy]
        self.dt = dt
        self.X = np.zeros((4, 1)) 
        
        # Ma trận chuyển trạng thái (F)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Ma trận đo lường (H) - chỉ đo được vị trí x, y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.P = np.eye(4) * 1000 # Ma trận hiệp phương sai sai số dự đoán
        self.Q = np.eye(4) * process_noise # Nhiễu hệ thống
        self.R = np.eye(2) * measurement_noise # Nhiễu đo lường

    def predict(self):
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.X

    def update(self, z):
        # z: Phép đo thực tế [x, y]
        z = np.array(z).reshape(2, 1)
        y = z - np.dot(self.H, self.X) # Innovation
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman Gain
        self.X = self.X + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
class SpeedEstimator:
    def __init__(self, fps, meters_per_pixel):
        self.fps = fps
        self.dt = 1.0 / fps
        self.mpp = meters_per_pixel
        
        # Lưu bộ lọc Kalman cho từng xe
        self.filters = {}
        # Window lưu trữ vận tốc tức thời để làm mượt
        self.speed_buffer = defaultdict(lambda: deque(maxlen=15)) 
        
    def update(self, track_id, position_bev, frame_idx):
        x, y = position_bev
        
        # 1. Khởi tạo hoặc cập nhật Kalman Filter
        if track_id not in self.filters:
            kf = KalmanFilter(dt=self.dt)
            kf.X[0:2] = np.array([[x], [y]])
            self.filters[track_id] = kf
            return 0
        
        kf = self.filters[track_id]
        kf.predict()
        kf.update([x, y])
        
        # 2. Lấy vận tốc từ trạng thái của Kalman [vx, vy]
        # vx, vy ở đây đơn vị là Pixel/Frame hoặc Pixel/Second tùy dt
        vx = kf.X[2, 0]
        vy = kf.X[3, 0]
        
        # Tốc độ pixel/giây
        v_pixel_per_sec = np.sqrt(vx**2 + vy**2)
        
        # Đổi sang km/h: (pixel/s * meters/pixel) * 3.6
        speed_kmh = (v_pixel_per_sec * self.mpp) * 3.6
        
        # 3. Sliding Window Average (Làm mượt bước cuối)
        self.speed_buffer[track_id].append(speed_kmh)
        smoothed_speed = sum(self.speed_buffer[track_id]) / len(self.speed_buffer[track_id])
        
        return smoothed_speed