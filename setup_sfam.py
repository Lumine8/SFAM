import os

# Define the file contents
files = {
    "sfam_project/requirements.txt": """torch>=2.0.0
numpy
matplotlib
scikit-learn
""",

    "sfam_project/data/__init__.py": "",

    "sfam_project/data/synthetic_data.py": """import torch
import numpy as np
from torch.utils.data import Dataset

class SyntheticBiometricDataset(Dataset):
    \"\"\"
    Generates synthetic 'Face' (Image) and 'Voice' (Vector) data.
    \"\"\"
    def __init__(self, num_users=50, samples_per_user=10):
        self.data = []
        self.labels = []
        
        for user_id in range(num_users):
            # Base identity (random latent vector)
            base_face = np.random.randn(3, 32, 32) # Simulated 32x32 image
            base_voice = np.random.randn(64)       # Simulated voice embedding
            
            for _ in range(samples_per_user):
                # Add noise to simulate different sessions
                face_noise = np.random.normal(0, 0.2, (3, 32, 32))
                voice_noise = np.random.normal(0, 0.2, (64))
                
                self.data.append({
                    'image': torch.FloatTensor(base_face + face_noise),
                    'voice': torch.FloatTensor(base_voice + voice_noise)
                })
                self.labels.append(user_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
""",

    "sfam_project/models/__init__.py": "",

    "sfam_project/models/encoders.py": """import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    \"\"\"Standard CNN for 32x32 images\"\"\"
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128)
        )
    def forward(self, x):
        return self.conv(x)

class AudioEncoder(nn.Module):
    \"\"\"MLP for fixed-size audio vectors\"\"\"
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    def forward(self, x):
        return self.net(x)
""",

    "sfam_project/models/abstraction.py": """import torch
import torch.nn as nn

class IAM_Module(nn.Module):
    \"\"\"
    Irreversible Abstraction Module (IAM)
    Uses BioHashing (Random Projection + Binarization)
    \"\"\"
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, fused_features, user_seed, training=False):
        \"\"\"
        fused_features: [batch, dim]
        user_seed: int (The revokable key)
        training: bool (If True, use Tanh for gradients. If False, use Sign for bits)
        \"\"\"
        # 1. Generate User-Specific Projection Matrix
        # Note: We use the seed to deterministically create the matrix on the fly
        torch.manual_seed(user_seed) 
        projection = torch.randn(self.input_dim, self.output_dim).to(fused_features.device)
        
        # 2. Project
        projected = torch.matmul(fused_features, projection)
        
        # 3. Non-Linearity
        if training:
            return torch.tanh(projected) # Differentiable approximation
        else:
            return torch.sign(projected) # Hard bits for security
""",

    "sfam_project/models/sfam_net.py": """import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import ImageEncoder, AudioEncoder
from .abstraction import IAM_Module

class SFAM(nn.Module):
    def __init__(self, embedding_dim=128, secure_dim=256):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.aud_enc = AudioEncoder()
        
        # Fusion Layer
        self.fusion = nn.Linear(128 + 64, embedding_dim)
        
        # Security Layer
        self.iam = IAM_Module(embedding_dim, secure_dim)

    def forward(self, img, voice, user_keys, training=False):
        # 1. Encode
        i_vec = self.img_enc(img)
        v_vec = self.aud_enc(voice)
        
        # 2. Fuse
        combined = torch.cat([i_vec, v_vec], dim=1)
        fused = F.relu(self.fusion(combined))
        
        # 3. Secure Abstraction (Handle batch keys)
        # Simplified: Assuming scalar key for whole batch or single item logic
        if isinstance(user_keys, int):
             return self.iam(fused, user_keys, training=training)
        
        # If keys is a tensor (batch of different keys), loop (slower but correct)
        outputs = []
        for k in range(fused.shape[0]):
            key_val = user_keys[k].item() if isinstance(user_keys, torch.Tensor) else user_keys
            out = self.iam(fused[k].unsqueeze(0), key_val, training=training)
            outputs.append(out)
            
        return torch.cat(outputs)
""",

    "sfam_project/training/__init__.py": "",

    "sfam_project/training/losses.py": """import torch
import torch.nn.functional as F

def contrastive_loss(embeddings, labels, margin=0.2):
    \"\"\"
    Pull same users together, push different users apart.
    Margin 0.2 means: "Punish if impostors are > 20% similar"
    \"\"\"
    # Normalize
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    
    # Pairwise similarity
    similarity = torch.matmul(norm_emb, norm_emb.T)
    
    # Label mask
    labels = labels.unsqueeze(0)
    mask = (labels == labels.T).float()
    
    # Positive Loss (Minimize distance for same user)
    loss_pos = (1 - similarity) * mask
    
    # Negative Loss (Maximize distance for diff user up to margin)
    loss_neg = torch.clamp(similarity - margin, min=0) * (1 - mask)
    
    return loss_pos.mean() + loss_neg.mean()
""",

    "sfam_project/training/train.py": """import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sfam_net import SFAM
from .losses import contrastive_loss

def train_sfam(dataset, epochs=10, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SFAM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"--- Starting Training on {device} ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_data, batch_labels in dataloader:
            imgs = batch_data['image'].to(device)
            voice = batch_data['voice'].to(device)
            labels = batch_labels.to(device)
            
            # Use a fixed system key for training weights
            keys = torch.full_like(labels, 42)
            
            optimizer.zero_grad()
            # training=True uses Tanh (differentiable)
            embeddings = model(imgs, voice, keys, training=True)
            
            loss = contrastive_loss(embeddings, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return model
""",

    "sfam_project/eval/__init__.py": "",

    "sfam_project/eval/evaluate.py": """import torch
import torch.nn.functional as F

def run_evaluation(model, dataset, device="cpu"):
    model.eval()
    
    # Use User 0 as test subject
    idx = 0
    data_A1 = dataset[idx][0]     # Sample 1
    data_A2 = dataset[idx+1][0]   # Sample 2
    
    # Use User 20 as impostor
    data_B = dataset[20][0]
    
    # Helper to prepare inputs
    def prep(d):
        return d['image'].unsqueeze(0).to(device), d['voice'].unsqueeze(0).to(device)

    img_A1, aud_A1 = prep(data_A1)
    img_A2, aud_A2 = prep(data_A2)
    img_B, aud_B = prep(data_B)

    print("\\n--- Security Evaluation ---")

    # 1. AUTHENTICATION (Same User, Same Key)
    with torch.no_grad():
        emb1 = model(img_A1, aud_A1, 12345)
        emb2 = model(img_A2, aud_A2, 12345)
        sim = F.cosine_similarity(emb1, emb2).item()
        print(f"1. Auth Check (Target > 0.8):   {sim:.4f} " + ("✅ PASS" if sim > 0.8 else "❌ FAIL"))

    # 2. CANCELLABILITY (Same User, Changed Key)
    with torch.no_grad():
        emb_revoked = model(img_A1, aud_A1, 99999) # New Key
        sim = F.cosine_similarity(emb1, emb_revoked).item()
        print(f"2. Revoke Check (Target < 0.2): {sim:.4f} " + ("✅ PASS" if sim < 0.2 else "❌ FAIL"))

    # 3. IMPOSTOR (Diff User, Same Key)
    with torch.no_grad():
        emb_imp = model(img_B, aud_B, 12345)
        sim = F.cosine_similarity(emb1, emb_imp).item()
        print(f"3. Impostor Check (Target < 0.2): {sim:.4f} " + ("✅ PASS" if sim < 0.2 else "❌ FAIL"))
""",

    "sfam_project/main.py": """import torch
from data.synthetic_data import SyntheticBiometricDataset
from training.train import train_sfam
from eval.evaluate import run_evaluation

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Generate Data
    print("Generating Dataset...")
    dataset = SyntheticBiometricDataset()
    
    # 2. Train
    model = train_sfam(dataset, epochs=15, device=DEVICE)
    
    # 3. Evaluate
    run_evaluation(model, dataset, device=DEVICE)

if __name__ == "__main__":
    main()
"""
}

def create_structure():
    print("🚀 Creating SFAM Project Structure...")
    for filepath, content in files.items():
        # Create directories if they don't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   Created directory: {directory}")
        
        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   Created file: {filepath}")
    
    print("\n✅ Setup Complete! To run the project:")
    print("   1. cd sfam_project")
    print("   2. python main.py")

if __name__ == "__main__":
    create_structure()