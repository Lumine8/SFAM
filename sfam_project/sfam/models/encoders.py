import torch
import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # FIX: global_pool='avg' prevents the 100,352 shape error
        self.backbone = timm.create_model(
            'ghostnet_100', 
            pretrained=True, 
            num_classes=0, 
            global_pool='avg' 
        )
        
        num_features = self.backbone.num_features
        
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.project(features)

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, 1]
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])