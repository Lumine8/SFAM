import time
import math
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

# Standard ImageNet stats
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class GestureCapture:
    def __init__(self):
        self.points = [] 
        self.is_drawing = False
        self.canvas_size = (640, 480)

    def add_point(self, x, y):
        t = time.perf_counter()
        # Noise Filter: Decimate points too close
        if len(self.points) > 0:
            last_x, last_y, _ = self.points[-1]
            dist = math.hypot(x - last_x, y - last_y)
            if dist < 2.0: return 
        self.points.append((x, y, t))

    def reset(self):
        self.points = []
        self.is_drawing = False

    def process_gesture(self, device="cpu"):
        if len(self.points) < 10: return None, None, None
        
        # 1. Spatial Image
        img_pil = Image.new("RGB", self.canvas_size, "black")
        draw = ImageDraw.Draw(img_pil)
        xy_points = [(p[0], p[1]) for p in self.points]
        draw.line(xy_points, fill="white", width=8, joint="curve")
        spatial_t = transform_pipeline(img_pil).unsqueeze(0).to(device)
        
        # 2. Physics Extraction
        velocities = []
        accels = []
        path_length = 0
        
        for i in range(1, len(self.points)):
            p1, p2 = self.points[i-1], self.points[i]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            dt = p2[2] - p1[2]
            path_length += dist
            if dt < 1e-5: dt = 1e-5
            v = dist / dt
            velocities.append(v)
            if i > 1:
                dv = velocities[-1] - velocities[-2]
                a = dv / dt
                accels.append(abs(a))
        
        # Log-Scaling
        log_vels = [math.log1p(v) for v in velocities]
        if not log_vels: return None, None, None
        
        # Feature Engineering
        med_v = np.median(log_vels)
        max_v = np.max(log_vels)
        avg_a = math.log1p(np.mean(accels)) if accels else 0
        stability = np.std(log_vels)
        duration = self.points[-1][2] - self.points[0][2]
        
        start, end = self.points[0], self.points[-1]
        euclidean = math.hypot(end[0]-start[0], end[1]-start[1])
        tortuosity = path_length / (euclidean + 1e-5)
        
        # Normalize
        norm_vec = [
            (med_v - 6.0) / 2.0,
            (max_v - 8.0) / 2.0,
            (avg_a - 10.0) / 5.0,
            (stability - 1.0) / 1.0,
            (duration - 1.0) / 2.0,
            (tortuosity - 1.0) / 0.5
        ]
        behavior_t = torch.tensor(norm_vec, dtype=torch.float32).unsqueeze(0).to(device)
        
        return spatial_t, behavior_t, img_pil
