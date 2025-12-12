import cv2
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
