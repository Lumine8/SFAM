import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import math

# =====================================================
# INTERNAL BUILDING BLOCKS (NOT PUBLIC API)
# =====================================================

class GhostModule(nn.Module):
    """Lightweight CNN block for efficient visual feature extraction"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        self.oup = oup

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class AdaptiveFusion(nn.Module):
    """Attention-based fusion between visual and behavioral features"""
    def __init__(self, visual_dim, behavior_dim, fusion_dim=256):
        super().__init__()
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

        weights = self.attention(torch.cat([v, b], dim=1))
        alpha_v = weights[:, 0].unsqueeze(1)
        alpha_b = weights[:, 1].unsqueeze(1)

        return alpha_v * v + alpha_b * b


class BioHashProjection(nn.Module):
    """User-specific orthogonal projection"""
    def __init__(self, output_dim=256):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, features, user_key):
        if user_key.dim() == 2:
            user_key = user_key.unsqueeze(0)

        projected = torch.bmm(features.unsqueeze(1), user_key).squeeze(1)
        return projected


# =====================================================
# INTERNAL MODEL CORE (NOT IMPORTED DIRECTLY)
# =====================================================

class _SFAMAdaptiveCore(nn.Module):
    def __init__(self, behavioral_dim=6, secure_dim=256):
        super().__init__()

        # Visual Encoder (GhostNet-style)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            GhostModule(16, 24),
            nn.MaxPool2d(2),
            GhostModule(24, 40),
            nn.MaxPool2d(2),
            GhostModule(40, 80),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.visual_dim = 80

        # Behavioral Encoder (LSTM)
        self.behavior_encoder = nn.LSTM(
            input_size=behavioral_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.behavior_dim = 64

        # Fusion + Projection
        self.fusion = AdaptiveFusion(self.visual_dim, self.behavior_dim, fusion_dim=256)
        self.biohash = BioHashProjection(output_dim=secure_dim)

    def forward(self, images, motion_seq, user_keys, binarize=False):
        # Visual features
        v = self.visual_encoder(images).view(images.size(0), -1)

        # Behavioral features
        _, (h_n, _) = self.behavior_encoder(motion_seq)
        b = h_n[-1]

        # Adaptive fusion
        fused = self.fusion(v, b)

        # Secure projection
        hashed = self.biohash(fused, user_keys)
        out = F.normalize(hashed, p=2, dim=1)

        if binarize:
            return torch.sign(out)
        return out


# =====================================================
# PUBLIC API (WHAT USERS IMPORT)
# =====================================================

def SFAM_Adaptive(behavioral_dim=6, secure_dim=256):
    """
    Factory function that returns an Adaptive SFAM model.

    This avoids class import conflicts and keeps a clean public API.
    """
    return _SFAMAdaptiveCore(
        behavioral_dim=behavioral_dim,
        secure_dim=secure_dim
    )


def generate_user_key(user_seed, salt, dim=256):
    """Deterministic orthogonal user-specific key"""
    raw = f"{user_seed}_{salt}".encode("utf-8")
    seed = int(hashlib.sha256(raw).hexdigest(), 16) % (2**32)

    g = torch.Generator()
    g.manual_seed(seed)

    W = torch.randn(dim, dim, generator=g)
    Q, _ = torch.linalg.qr(W)
    return Q
