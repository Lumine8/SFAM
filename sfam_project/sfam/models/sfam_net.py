import torch
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
