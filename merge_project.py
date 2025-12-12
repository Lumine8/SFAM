import os

# 1. UPDATE THE MODEL ARCHITECTURE
sfam_net_code = """import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SFAM(nn.Module):
    def __init__(self, behavioral_dim=6, secure_dim=256):
        super().__init__()
        
        # A. Spatial Path (GhostNet)
        print("   -> Loading GhostNet Backbone...")
        self.cnn_backbone = timm.create_model('ghostnet_100', pretrained=True, num_classes=0)
        
        # Auto-detect output size
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.cnn_backbone(dummy)
            cnn_dim = out.shape[1]
            
        self.cnn_projector = nn.Linear(cnn_dim, 256)
        
        # B. Behavioral Path (TUNED)
        self.behavior_net = nn.Sequential(
            nn.Linear(behavioral_dim, 128),
            nn.BatchNorm1d(128),
            nn.Mish(), 
            nn.Dropout(0.2), 
            nn.Linear(128, 64)
        )
        
        # C. Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.Mish(),
            nn.Linear(512, secure_dim)
        )
        self.output_dim = secure_dim

    def forward(self, pattern_img, behavior_vec, user_seed):
        # 1. Spatial
        spatial = self.cnn_projector(self.cnn_backbone(pattern_img))
        
        # 2. Behavioral
        behavioral = self.behavior_net(behavior_vec)
        
        # 3. Fuse
        combined = torch.cat([spatial, behavioral], dim=1)
        raw_emb = self.fusion(combined)
        
        # 4. Irreversible Abstraction
        torch.manual_seed(user_seed)
        projection = torch.randn(self.output_dim, self.output_dim, device=raw_emb.device)
        hashed = torch.matmul(raw_emb, projection)
        
        return torch.sign(hashed)
"""

# 2. CREATE THE GESTURE INPUT LOADER
gesture_loader_code = """import time
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
"""

# 3. UPDATE MAIN.PY
main_code = """import cv2
import torch
import numpy as np
from models.sfam_net import SFAM
from data.gesture_loader import GestureCapture

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 SecuADR (Merged) initializing on {DEVICE}...")
    
    # Load the Model from 'models/sfam_net.py'
    model = SFAM().to(DEVICE).eval()
    recorder = GestureCapture()
    
    gui_h, gui_w = 480, 640
    window_name = "SecuADR Professional: Merged Project"
    cv2.namedWindow(window_name)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            recorder.reset()
            recorder.is_drawing = True
            recorder.add_point(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if recorder.is_drawing:
                recorder.add_point(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            recorder.is_drawing = False
            
    cv2.setMouseCallback(window_name, mouse_callback)

    enrolled_hash = None
    last_status = "DRAW & PRESS 'E'"
    last_color = (200, 200, 200)

    while True:
        display = np.zeros((gui_h, gui_w, 3), dtype=np.uint8)
        
        if len(recorder.points) > 1:
            pts = np.array([[p[0], p[1]] for p in recorder.points], np.int32)
            cv2.polylines(display, [pts], False, (0, 255, 0), 2)

        cv2.putText(display, last_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)
        if enrolled_hash is not None:
             cv2.putText(display, "🔒 ENROLLED", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('e') or key == 32:
            s_t, b_t, _ = recorder.process_gesture(device=DEVICE)
            
            if s_t is not None and b_t is not None:
                if key == ord('e'):
                    with torch.no_grad():
                        enrolled_hash = model(s_t, b_t, user_seed=999)
                    last_status = "ENROLLED!"
                    last_color = (0, 255, 255)
                    print("🔒 User Enrolled.")
                elif key == 32 and enrolled_hash is not None:
                    with torch.no_grad():
                        live_hash = model(s_t, b_t, user_seed=999)
                    dist = torch.sum((live_hash * enrolled_hash) < 0).item() / 256.0
                    if dist < 0.15: 
                        last_status = f"GRANTED ({dist:.2f})"
                        last_color = (0, 255, 0)
                        print(f"✅ Access Granted (Score: {dist:.2f})")
                    else:
                        last_status = f"DENIED ({dist:.2f})"
                        last_color = (0, 0, 255)
                        print(f"❌ Access Denied (Score: {dist:.2f})")
            else:
                last_status = "Gesture too short!"

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
"""

# WRITING FILES
# NOTE: Using encoding="utf-8" to handle emojis properly on Windows
print("📦 Merging SecuADR logic into SFAM structure...")

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

with open("models/sfam_net.py", "w", encoding="utf-8") as f:
    f.write(sfam_net_code)
    print("   -> Updated models/sfam_net.py")

with open("data/gesture_loader.py", "w", encoding="utf-8") as f:
    f.write(gesture_loader_code)
    print("   -> Created data/gesture_loader.py")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_code)
    print("   -> Updated main.py")

print("✅ Merge Complete!")
print("   Run: python main.py")