import torch
import torch.nn as nn
import timm

# 1. Import the module where SFAM lives
import sfam.models.sfam_net 
from sfam import SFAM, image_fm, gesture_fm 

# 2. Define the FIXED Encoder locally (Correcting the Flatten/Pooling issue)
class FixedImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # FIX: global_pool='avg' forces the 7x7 grid into a 1x1 vector
        self.backbone = timm.create_model(
            'ghostnet_100', 
            pretrained=True, 
            num_classes=0, 
            global_pool='avg' 
        )
        num_features = self.backbone.num_features
        
        # Now the input size matches the linear layer
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.project(self.backbone(x))

# 3. 💪 FORCE PATCH: Overwrite the class inside 'sfam_net'
# This ensures that when SFAM() is called, it uses OUR class, not the old one.
sfam.models.sfam_net.ImageEncoder = FixedImageEncoder
print("✅ Patched ImageEncoder inside sfam_net!")

# 4. Initialize the Engine
device = "cpu"
model = SFAM(
    embedding_dim=64, 
    secure_dim=256
).to(device).eval()

print(f"🚀 SFAM Engine loaded on {device} (with Hot Fix)")