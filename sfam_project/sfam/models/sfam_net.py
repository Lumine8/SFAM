import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import ImageEncoder, AudioEncoder, TextEncoder
from .abstraction import IAM_Module

class SFAM(nn.Module):
    def __init__(self, embedding_dim=128, secure_dim=256, behavior_dim=1):
        """
        Args:
            embedding_dim: Size of the internal feature vector (default 128)
            secure_dim: Size of the final secure hash (default 256)
            behavior_dim: Number of features in your time-series data. 
                          - Set to 1 for raw Audio/Waveform.
                          - Set to 2 for raw Mouse (x, y).
                          - Set to 64 if using pre-processed features (YOUR CASE).
        """
        super().__init__()
        
        # 1. Image Encoder (Spatial)
        self.img_enc = ImageEncoder(embedding_dim=embedding_dim)
        
        # 2. Audio/Behavior Encoder (Temporal)
        # We pass 'behavior_dim' here so the LSTM fits your data size
        self.aud_enc = AudioEncoder(embedding_dim=embedding_dim, input_channels=behavior_dim)
        
        # Fusion Layer (Concatenates Image + Audio embeddings)
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Security Layer
        self.iam = IAM_Module(embedding_dim, secure_dim)

    def forward(self, img, behavior, user_keys, training=False):
        # 1. Encode
        i_vec = self.img_enc(img)       # Image -> Embedding
        b_vec = self.aud_enc(behavior)  # Behavior -> Embedding
        
        # 2. Fuse
        combined = torch.cat([i_vec, b_vec], dim=1)
        fused = F.relu(self.fusion(combined))
        
        # 3. Secure Abstraction
        if isinstance(user_keys, int) or (isinstance(user_keys, torch.Tensor) and user_keys.numel() == 1):
             return self.iam(fused, user_keys, training=training)
        
        # Handle batch of different keys
        outputs = []
        for k in range(fused.shape[0]):
            key_val = user_keys[k].item() if isinstance(user_keys, torch.Tensor) else user_keys[k]
            out = self.iam(fused[k].unsqueeze(0), key_val, training=training)
            outputs.append(out)
            
        return torch.cat(outputs)