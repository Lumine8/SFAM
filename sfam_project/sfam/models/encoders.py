import torch
import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    """
    Handles 2D spatial data (Face, Iris, Fingerprint images).
    Uses a lightweight CNN backbone (GhostNet) for efficient feature extraction.
    """
    def __init__(self, embedding_dim=128, model_name='ghostnet_100'):
        super().__init__()
        
        # 1. Load Backbone (Pretrained, stripped head)
        # num_classes=0 returns the global pooled feature vector (B, C)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # 2. Dynamic Shape Inference
        # Avoids hardcoding input dimensions; makes swapping backbones safe.
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone(dummy)
            real_in_features = out.shape[1]
            
        # 3. Projection Head
        # Projects high-dim CNN features down to the shared embedding space.
        self.project = nn.Sequential(
            nn.Linear(real_in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            # [Fix] Tanh ensures bounded embeddings [-1, 1], 
            # which aligns better with SFAM's cosine/projection logic than ReLU.
            nn.Tanh() 
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.project(features)


class TemporalEncoder(nn.Module):
    """
    Handles 1D time-series data (Voice, Mouse Dynamics, Gesture sequences).
    Renamed from 'AudioEncoder' to reflect generic temporal capabilities.
    """
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=64, num_layers=2):
        super().__init__()
        
        # Bi-directional LSTM captures context from both past and future
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Projection Head
        # Input is hidden_dim * 2 (because of bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim)
        """
        # lstm_out: (Batch, Seq_Len, Hidden_Dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # [Fix] Mean Pooling
        # Instead of taking the last hidden state (which is noisy for gestures),
        # we average across the entire sequence. This makes the embedding
        # robust to speed variations and end-of-sequence jitter.
        pooled = lstm_out.mean(dim=1)
        
        return self.fc(pooled)


class TextEncoder(nn.Module):
    """
    Handles symbolic data (Passwords, PINs, static tokens).
    
    SECURITY NOTE: 
    This encoder treats text as a symbolic secret, not a biometric trait.
    It learns a fixed mapping for tokens but does not inherently capture
    'behavioral' text typing patterns (use TemporalEncoder for keystroke dynamics).
    """
    def __init__(self, vocab_size=1000, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        # x: (Batch, Seq_Len) indices
        # Simple averaging of token embeddings
        embedded = self.embedding(x)
        return embedded.mean(dim=1)