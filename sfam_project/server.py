from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import torch
import numpy as np
import os

# IMPORT YOUR MODULAR ENGINE
# We now import from the 'sfam' package folder
from sfam.models.sfam_net import SFAM
from sfam.data.gesture_loader import GestureCapture

# --- SETUP ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 SecuADR Server initializing on {DEVICE}...")

# Load the Brain
model = SFAM().to(DEVICE)
model.eval()

# In-Memory Database (Stores HASHES, not Gestures)
# Format: { "alice": Tensor([1, -1, 1, ...]) }
user_db = {} 

# --- API MODELS ---
class TouchPoint(BaseModel):
    x: float
    y: float
    timestamp: float

class AuthRequest(BaseModel):
    user_id: str
    points: List[TouchPoint]

app = FastAPI(title="SecuADR Cloud API")

# --- HELPER FUNCTION ---
def get_hash_from_points(points):
    """
    Converts raw JSON points -> SFAM Hash
    """
    # 1. Convert JSON to GestureCapture format
    recorder = GestureCapture()
    # We must adjust timestamps to be relative (start at 0) or standard
    # GestureCapture expects time.perf_counter(), but relative diffs matter most.
    start_time = points[0].timestamp
    for p in points:
        # Re-create the recorder state
        recorder.points.append((p.x, p.y, p.timestamp))
    
    # 2. Process using your robust loader
    # Note: We skip 'add_point' logic here and trust the client sent raw data
    # or we can re-run filters. Let's call process_gesture directly.
    s_t, b_t, _ = recorder.process_gesture(device=DEVICE)
    
    if s_t is None: return None
    
    # 3. Generate Hash
    with torch.no_grad():
        # Using fixed system seed 999
        secure_hash = model(s_t, b_t, user_seed=999)
        
    return secure_hash

# --- ENDPOINTS ---

@app.post("/enroll")
async def enroll_user(req: AuthRequest):
    secure_hash = get_hash_from_points(req.points)
    
    if secure_hash is None:
        raise HTTPException(status_code=400, detail="Gesture too short or noisy")
        
    # STORE ONLY THE HASH
    user_db[req.user_id] = secure_hash
    
    # Return success (Do not return the hash to the client for security)
    return {"status": "success", "message": f"User {req.user_id} enrolled."}

@app.post("/verify")
async def verify_user(req: AuthRequest):
    if req.user_id not in user_db:
        raise HTTPException(status_code=404, detail="User not found")
        
    live_hash = get_hash_from_points(req.points)
    if live_hash is None:
        raise HTTPException(status_code=400, detail="Gesture too short")
        
    stored_hash = user_db[req.user_id]
    
    # COMPARE HASHES (Hamming Distance)
    dist = torch.sum((live_hash * stored_hash) < 0).item() / 256.0
    
    if dist < 0.15:
        return {"auth": True, "score": dist, "message": "Access Granted"}
    else:
        return {"auth": False, "score": dist, "message": "Access Denied"}

if __name__ == "__main__":
    # Get the PORT environment variable, default to 8000 for local testing
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)