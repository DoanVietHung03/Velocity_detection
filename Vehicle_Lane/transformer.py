import cv2
import numpy as np

class ViewTransformer:
    def __init__(self, source_points, target_width, target_height):
        """
        source_points: 4 điểm hình thang trên video gốc
        """
        self.src = np.float32(source_points)
        self.dst = np.float32([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ])
        
        # Tính ma trận chuyển đổi
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)

    def transform_points(self, points):
        """
        Input: List các điểm [[x, y], ...]
        Output: List các điểm sau khi transform [[x', y'], ...]
        """
        if len(points) == 0:
            return []
            
        # Reshape để phù hợp với input của cv2.perspectiveTransform
        # Format: (N, 1, 2)
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        transformed_points = cv2.perspectiveTransform(points_array, self.M)
        
        # Reshape lại về (N, 2)
        return transformed_points.reshape(-1, 2)