import torch
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
