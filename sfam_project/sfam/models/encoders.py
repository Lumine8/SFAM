import torch
import torch.nn as nn
import timm

# --- 1. SPATIAL ENCODER (Images, Face, Retina) ---
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Load backbone (GhostNet, ResNet, etc.)
        # global_pool='avg' forces the output to be a vector, not a grid
        self.backbone = timm.create_model(
            'ghostnet_100', 
            pretrained=True, 
            num_classes=0, 
            global_pool='avg' 
        )
        
        # --- ROBUST SIZE DETECTION ---
        # Instead of guessing 960 or 1280, we run a fake image through 
        # to see exactly what the model outputs. This prevents runtime errors.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            real_in_features = out.shape[1]
            
        # Project to shared embedding space
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(real_in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        features = self.backbone(x)
        return self.project(features)


# --- 2. TEMPORAL ENCODER (Voice, Mouse Gestures) ---
class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Standard LSTM for time-series data
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        # x: [Batch, Sequence_Length, 1]
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]) # Take the last hidden state


# --- 3. SYMBOLIC ENCODER (Text, Passwords) ---
# Added for future compatibility (v1.2+)
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.transformer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        # x: [Batch, Sequence_Length] (Integer Tokens)
        x = self.embedding(x)      # -> [Batch, Seq, 64]
        x = self.transformer(x)    # -> [Batch, Seq, 64]
        
        # Permute for pooling: [Batch, 64, Seq]
        x = x.permute(0, 2, 1)     
        x = self.pool(x).squeeze(-1) # -> [Batch, 64]
        
        return self.fc(x)