import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    """Standard CNN for 32x32 images"""
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
    """MLP for fixed-size audio vectors"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    def forward(self, x):
        return self.net(x)
