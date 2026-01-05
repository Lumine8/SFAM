import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

# --- SMART IMPORT ---
# This block handles the import regardless of whether you run this 
# as a library (from sfam.models...) or as a script.
try:
    # Try relative import first (Best for package structure)
    from .encoders import ImageEncoder, TemporalEncoder, TextEncoder
except ImportError:
    # Fallback to absolute import (If running script directly in folder)
    from encoders import ImageEncoder, TemporalEncoder, TextEncoder

# ==========================================
# SFAM BASE CLASS
# ==========================================

class SFAM(nn.Module):
    """ 
    Base class containing the core BioHashing security logic. 
    It imports the encoders instead of defining them here.
    """
    def __init__(self, behavioral_dim=6, secure_dim=256, noise_std=0.01):
        super().__init__()
        self.noise_std = noise_std
        self.secure_dim = secure_dim
        
        # 1. Instantiate Encoders (Imported from encoders.py)
        self.image_encoder = ImageEncoder(embedding_dim=128)
        self.behavior_encoder = TemporalEncoder(input_dim=behavioral_dim, embedding_dim=128)
        
        # 2. Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 512),
            nn.Mish(),
            nn.Linear(512, secure_dim) 
        )

    def extract_features(self, pattern_img, behavior_seq):
        """Helper to get raw embeddings before fusion"""
        spatial = self.image_encoder(pattern_img)       
        behavioral = self.behavior_encoder(behavior_seq) 
        return spatial, behavioral

    def biohash(self, raw_emb, projection_matrices):
        """
        The 'Golden Master' BioHashing implementation.
        """
        projection_matrices = projection_matrices.detach()
        
        # Batch safety check
        if raw_emb.shape[0] != projection_matrices.shape[0]:
            if projection_matrices.shape[0] == 1:
                projection_matrices = projection_matrices.expand(raw_emb.shape[0], -1, -1)
            else:
                raise ValueError(f"Batch mismatch: Emb {raw_emb.shape[0]} vs Key {projection_matrices.shape[0]}")

        # Pre-Project Logic
        features = torch.tanh(raw_emb)
        features = F.normalize(features, p=2, dim=1) 

        # Privacy Noise (Training Only)
        if self.training and self.noise_std > 0:
            noise = self.noise_std * torch.randn_like(features)
            features = features + noise
            features = torch.clamp(features, -1.5, 1.5)

        # Orthogonal Projection
        hashed = torch.bmm(features.unsqueeze(1), projection_matrices).squeeze(1)
        
        return hashed

# ==========================================
# SFAM ADAPTIVE (The Logic Core)
# ==========================================

class SFAM_Adaptive(SFAM):
    """
    Adaptive SFAM: Dual-Modal with Attention Gating.
    Dynamically weights Image vs. Behavior based on signal quality.
    """
    def __init__(self, behavioral_dim=6, secure_dim=256, noise_std=0.01):
        super().__init__(behavioral_dim, secure_dim, noise_std)
        
        # Attention Mechanism (Input 256 -> Weights for 2 modalities)
        self.attention_gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 2), 
            nn.Softmax(dim=1)
        )

    def forward(self, pattern_img, behavior_seq, projection_matrices, binarize=False):
        # 1. Encode (Using imported encoders)
        spatial, behavioral = self.extract_features(pattern_img, behavior_seq)
        
        # 2. Adaptive Weighting
        context = torch.cat([spatial, behavioral], dim=1)
        weights = self.attention_gate(context) # (B, 2)
        
        w_spatial = weights[:, 0].unsqueeze(1)
        w_behavior = weights[:, 1].unsqueeze(1)
        
        # 3. Weighted Fusion
        combined = torch.cat([spatial * w_spatial, behavioral * w_behavior], dim=1)
        
        # 4. Hash & Output
        raw_emb = self.fusion(combined)
        hashed = self.biohash(raw_emb, projection_matrices)
        out = torch.tanh(hashed)
        
        if binarize:
            return torch.sign(out)
        return out

# ==========================================
# HELPER: Key Generator
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