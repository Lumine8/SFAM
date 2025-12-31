import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SFAM(nn.Module):
    def __init__(self, behavioral_dim=6, secure_dim=256):
        super().__init__()
        
        # 1. Spatial Path (GhostNet)
        # We use a lightweight backbone for efficiency
        self.cnn_backbone = timm.create_model('ghostnet_100', pretrained=True, num_classes=0)
        
        # Auto-detect output size to prevent shape mismatch errors
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.cnn_backbone(dummy)
            cnn_dim = out.shape[1]
            
        self.cnn_projector = nn.Linear(cnn_dim, 256)
        
        # 2. Behavioral Path (MLP for feature data)
        # Takes velocity, acceleration, jerk, etc.
        self.behavior_net = nn.Sequential(
            nn.Linear(behavioral_dim, 128),
            nn.BatchNorm1d(128),
            nn.Mish(), 
            nn.Dropout(0.2), 
            nn.Linear(128, 64)
        )
        
        # 3. Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.Mish(),
            nn.Linear(512, secure_dim)
        )
        self.output_dim = secure_dim

    def forward(self, pattern_img, behavior_vec, user_seed):
        """
        Forward pass with Cancellable BioHashing.
        user_seed: An integer specific to the user (the "Salt")
        """
        # A. Extract Features
        spatial = self.cnn_projector(self.cnn_backbone(pattern_img))
        behavioral = self.behavior_net(behavior_vec)
        
        # B. Fuse Multimodal Data
        combined = torch.cat([spatial, behavioral], dim=1)
        raw_emb = self.fusion(combined)
        
        # C. Irreversible Abstraction (BioHashing / Salted Projection)
        # We use a LOCAL generator to ensure the seed affects only this layer,
        # preventing the "Global Random State" corruption bug.
        device = raw_emb.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(user_seed)) 
        
        # Generate the user-specific random projection matrix (The "Key")
        with torch.no_grad():
            projection = torch.randn(self.output_dim, self.output_dim, generator=gen, device=device)
            
        hashed = torch.matmul(raw_emb, projection)
        
        # D. Output Logic (Soft for Training, Hard for Security)
        if self.training:
            return torch.tanh(hashed) # Differentiable approximation
        else:
            return torch.sign(hashed) # Binary hash (Hammering distance ready)