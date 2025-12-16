import torch
import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # 1. Load backbone but FORCE it to pool the output
        # global_pool='avg' turns the 7x7 grid into a 1x1 vector automatically
        self.backbone = timm.create_model(
            'ghostnet_100', 
            pretrained=True, 
            num_classes=0,       # No classification layer
            global_pool='avg'    # <--- VITAL: Averages spatial dims to 1x1
        )
        
        # Get the output channels (usually 1280 for GhostNet, 2048 for ResNet)
        num_features = self.backbone.num_features
        
        # 2. Project to your embedding dimension
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, embedding_dim),
            nn.LayerNorm(embedding_dim), # Good practice for stability
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        features = self.backbone(x) 
        # features shape is now [Batch, num_features] (e.g., [1, 1280])
        
        out = self.project(features)
        return out