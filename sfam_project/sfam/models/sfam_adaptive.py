import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import math

# ==========================================
# 1. HELPER MODULES
# ==========================================

class GhostModule(nn.Module):
    """ Lightweight CNN block for efficient visual feature extraction """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class BioHashProjection(nn.Module):
    """ Projects features onto a user-specific orthogonal key """
    def __init__(self, input_dim, output_dim=256):
        super(BioHashProjection, self).__init__()
        self.output_dim = output_dim
    
    def forward(self, features, user_key):
        # features: (Batch, Feature_Dim)
        # user_key: (Batch, Feature_Dim, Output_Dim)
        
        if user_key.dim() == 2:
             user_key = user_key.unsqueeze(0)
             
        # Batch Matrix Multiplication
        # (B, 1, F) @ (B, F, O) -> (B, 1, O)
        projected = torch.bmm(features.unsqueeze(1), user_key).squeeze(1)
        return projected

class AdaptiveFusion(nn.Module):
    """ Attention Mechanism: Decides whether to trust Face or Motion more """
    def __init__(self, visual_dim, behavior_dim, fusion_dim=512):
        super(AdaptiveFusion, self).__init__()
        
        self.vis_proj = nn.Linear(visual_dim, fusion_dim)
        self.beh_proj = nn.Linear(behavior_dim, fusion_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 2), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, visual, behavior):
        v = self.vis_proj(visual) 
        b = self.beh_proj(behavior)
        
        combined = torch.cat([v, b], dim=1)
        weights = self.attention(combined) 
        
        alpha_v = weights[:, 0].unsqueeze(1)
        alpha_b = weights[:, 1].unsqueeze(1)
        
        fused = (alpha_v * v) + (alpha_b * b)
        return fused

# ==========================================
# 2. MAIN MODEL CLASS
# ==========================================

class SFAM_Adaptive(nn.Module):
    def __init__(self, behavioral_dim=6, secure_dim=256):
        super(SFAM_Adaptive, self).__init__()
        
        # A. Visual Encoder (GhostNet)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            GhostModule(16, 24),
            nn.MaxPool2d(2, 2),
            GhostModule(24, 40),
            nn.MaxPool2d(2, 2),
            GhostModule(40, 80),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.visual_dim = 80
        
        # B. Behavioral Encoder (LSTM)
        self.lstm = nn.LSTM(input_size=behavioral_dim, hidden_size=64, num_layers=2, batch_first=True)
        self.behavior_dim = 64
        
        # C. Fusion & Hash
        self.fusion = AdaptiveFusion(self.visual_dim, self.behavior_dim, fusion_dim=256)
        self.biohash = BioHashProjection(256, secure_dim)

    def forward(self, images, motion_seq, user_keys):
        # 1. Visual Features
        v_feat = self.features(images)
        v_feat = v_feat.view(v_feat.size(0), -1) 
        
        # 2. Motion Features
        _, (h_n, _) = self.lstm(motion_seq)
        b_feat = h_n[-1] 
        
        # 3. Adaptive Fusion
        fused = self.fusion(v_feat, b_feat)
        
        # 4. Secure Projection
        secure_hash = self.biohash(fused, user_keys)
        
        return F.normalize(secure_hash, p=2, dim=1)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def generate_user_key(user_seed, salt, dim=256):
    raw_str = f"{user_seed}_{salt}".encode('utf-8')
    seed_hash = hashlib.sha256(raw_str).hexdigest()
    final_seed = int(seed_hash, 16) % (2**32)
    
    rng = torch.Generator()
    rng.manual_seed(final_seed)
    
    W = torch.randn(dim, dim, generator=rng)
    Q, R = torch.linalg.qr(W)
    return Q