import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

# Ensure encoders.py is in the same directory
from .encoders import ImageEncoder, TemporalEncoder, TextEncoder

class SFAM(nn.Module):
    """
    Standard SFAM: Dual-Modal Static Fusion (Image + Behavior).
    Base class containing the core BioHashing security logic.
    
    SECURITY WARNINGS:
    1. Unique Projections: 'projection_matrices' must be unique per sample/user.
       Shape: (Batch_Size, Input_Dim, Output_Dim). Do not broadcast a single 
       matrix across the entire batch unless simulating a shared-key attack.
       
    2. Empirical Privacy: This architecture provides computational resistance 
       to inversion. It does NOT claim information-theoretic indistinguishability.
    """
    def __init__(self, behavioral_dim=6, secure_dim=256, noise_std=0.01):
        super().__init__()
        
        self.noise_std = noise_std
        self.secure_dim = secure_dim
        
        # --- 1. Encoders ---
        # Image Path (GhostNet)
        self.image_encoder = ImageEncoder(embedding_dim=128)
        
        # Behavior Path (LSTM for sequences)
        # Input: (Batch, Seq_Len, 6) -> Output: (Batch, 128)
        self.behavior_encoder = TemporalEncoder(input_dim=behavioral_dim, embedding_dim=128)
        
        # --- 2. Fusion ---
        # 128 (Image) + 128 (Behavior) = 256
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
        Pipeline: Tanh -> Normalize -> Noise(Train) -> Ortho-Project -> Tanh
        """
        # A. Detach Matrix (Safety: Prevent gradient leaks into keys)
        projection_matrices = projection_matrices.detach()
        
        # B. Assertions
        B, D = raw_emb.shape
        P_B, P_Dim1, P_Dim2 = projection_matrices.shape
        # Ensure we have one matrix per user in the batch
        assert B == P_B, f"CRITICAL: Batch size mismatch. Emb {B} vs Proj {P_B}. Keys must be unique per user."
        assert D == P_Dim1, f"Dimension Mismatch: Emb {D} vs Proj {P_Dim1}"

        # C. Pre-Project Logic (Non-linearity + Stability)
        features = torch.tanh(raw_emb)
        features = F.normalize(features, p=2, dim=1) # Project to unit hypersphere

        # D. Privacy Noise (Training Only)
        if self.training and self.noise_std > 0:
            noise = self.noise_std * torch.randn_like(features)
            features = features + noise
            features = torch.clamp(features, -1.5, 1.5) # Stability guard

        # E. Orthogonal Projection (Optimized via Einsum)
        # b=batch, i=input_dim, j=output_dim
        # Equivalent to: (Batch, 1, D) @ (Batch, D, D_out) -> (Batch, 1, D_out)
        hashed = torch.einsum('bi,bij->bj', features, projection_matrices)
        
        return hashed

    def forward(self, pattern_img, behavior_seq, projection_matrices, binarize=False):
        # 1. Encode
        spatial, behavioral = self.extract_features(pattern_img, behavior_seq)
        
        # 2. Fuse (Static Concatenation)
        combined = torch.cat([spatial, behavioral], dim=1)
        raw_emb = self.fusion(combined)
        
        # 3. Hash
        hashed = self.biohash(raw_emb, projection_matrices)
        out = torch.tanh(hashed)
        
        if binarize:
            return torch.sign(out)
        return out


class SFAM_Adaptive(SFAM):
    """
    Adaptive SFAM: Dual-Modal with Attention Gating.
    Dynamically weights Image vs. Behavior based on signal quality/noise.
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
        spatial, behavioral = self.extract_features(pattern_img, behavior_seq)
        
        # --- Adaptive Logic ---
        context = torch.cat([spatial, behavioral], dim=1)
        weights = self.attention_gate(context) # (B, 2)
        
        w_spatial = weights[:, 0].unsqueeze(1)
        w_behavior = weights[:, 1].unsqueeze(1)
        
        # Weighted Fusion
        combined = torch.cat([spatial * w_spatial, behavioral * w_behavior], dim=1)
        
        # Standard Pipeline
        raw_emb = self.fusion(combined)
        hashed = self.biohash(raw_emb, projection_matrices)
        out = torch.tanh(hashed)
        
        if binarize:
            return torch.sign(out)
        return out


class SFAM_TriModule(SFAM):
    """
    Tri-Modal SFAM: Image + Behavior + Symbolic (Text/Key).
    
    USAGE WARNING:
    The 'text_seq' input handles Symbolic Secrets (passwords, seeds).
    It is NOT a biometric trait (it is exact, not fuzzy). 
    Do not use this class if you want pure biometric authentication.
    """
    def __init__(self, behavioral_dim=6, secure_dim=256, noise_std=0.01, vocab_size=1000):
        super().__init__(behavioral_dim, secure_dim, noise_std)
        
        # Additional Text Encoder
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embedding_dim=128)
        
        # Override Fusion Layer to handle 3 inputs
        # 128 (Img) + 128 (Beh) + 128 (Text) = 384
        self.fusion = nn.Sequential(
            nn.Linear(384, 512),
            nn.Mish(),
            nn.Linear(512, secure_dim)
        )

    def forward(self, pattern_img, behavior_seq, text_seq, projection_matrices, binarize=False):
        """
        Args:
            text_seq: (Batch, Seq_Len) indices from text_fm.py
        """
        # 1. Encode All
        spatial = self.image_encoder(pattern_img)
        behavioral = self.behavior_encoder(behavior_seq)
        symbolic = self.text_encoder(text_seq) # (B, 128)
        
        # 2. Tri-Modal Fusion
        combined = torch.cat([spatial, behavioral, symbolic], dim=1)
        raw_emb = self.fusion(combined)
        
        # 3. Hash & Output
        hashed = self.biohash(raw_emb, projection_matrices)
        out = torch.tanh(hashed)
        
        if binarize:
            return torch.sign(out)
        return out


# --- Key Generator Helper ---
def generate_user_key(user_seed, salt, dim=256):
    """
    Generates revocable, orthogonal matrices (QR Decomposition).
    Output: (Dim, Dim) - Projection Matrix.
    """
    # Deterministic Seeding via SHA256
    raw_str = f"{user_seed}_{salt}".encode('utf-8')
    seed_hash = hashlib.sha256(raw_str).hexdigest()
    final_seed = int(seed_hash, 16) % (2**32)
    
    g = torch.Generator()
    g.manual_seed(final_seed)
    
    # Orthogonal Initialization
    W = torch.randn(dim, dim, generator=g)
    Q, R = torch.linalg.qr(W)
    return Q