import math
from collections import defaultdict, deque
import numpy as np

class SpeedEstimator:
    def __init__(self, fps, meters_per_pixel):
        self.fps = fps
        self.mpp = meters_per_pixel
        
        # Dictionary lưu lịch sử vị trí: {track_id: [ (x, y, frame_count), ... ]}
        # Chỉ giữ lại buffer ngắn để tính toán
        self.positions = defaultdict(lambda: deque(maxlen=60))
        
        # Lưu tốc độ đã tính để hiển thị cho mượt
        self.speeds = {}

    def update(self, track_id, position_bev, frame_idx):
        """
        track_id: ID của xe
        position_bev: Tọa độ (x, y) sau khi đã transform sang Bird-eye view
        """
        if len(self.positions[track_id]) > 0:
            last_pos, _ = self.positions[track_id][-1]
            
            # Alpha càng nhỏ (0.1-0.3) thì càng mượt nhưng trễ (lag)
            # Alpha càng lớn (0.7-0.9) thì bám sát chuyển động thật nhưng dễ rung
            alpha_pos = 0.2 
            smooth_x = last_pos[0] * (1 - alpha_pos) + position_bev[0] * alpha_pos
            smooth_y = last_pos[1] * (1 - alpha_pos) + position_bev[1] * alpha_pos
            position_bev = [smooth_x, smooth_y]
        
        self.positions[track_id].append((position_bev, frame_idx))

        sample_gap = int(self.fps / 2) # Lấy mẫu nửa giây
        
        # Cần ít nhất 2 điểm dữ liệu để tính tốc độ
        if len(self.positions[track_id]) < sample_gap:
            return 0
        
        # Lấy điểm hiện tại và điểm cách đây sample_gap frame (để tránh jitter)
        current_pos, current_frame = self.positions[track_id][-1]
        prev_pos, prev_frame = self.positions[track_id][-sample_gap] # Lấy mẫu cách nhau 5 frame
        
        # Tính khoảng cách Euclidean trong không gian BEV (Pixel)
        distance_pixels = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
        
        # Đổi sang mét
        distance_meters = distance_pixels * self.mpp
        
        # Tính thời gian (Seconds)
        time_elapsed = (current_frame - prev_frame) / self.fps
        
        if time_elapsed == 0: return 0
        
        # Tốc độ (m/s) -> km/h
        speed_ms = distance_meters / time_elapsed
        speed_kmh = speed_ms * 3.6
        
        # Smoothing: Lấy trung bình cộng nhẹ để số không nhảy lung tung
        if track_id in self.speeds:
            self.speeds[track_id] = 0.8 * self.speeds[track_id] + 0.2 * speed_kmh
        else:
            self.speeds[track_id] = speed_kmh
            
        return self.speeds[track_id]