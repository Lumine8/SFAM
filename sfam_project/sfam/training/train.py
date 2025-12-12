import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.sfam_net import SFAM
from .losses import contrastive_loss

def train_sfam(dataset, epochs=10, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SFAM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"--- Starting Training on {device} ---")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_data, batch_labels in dataloader:
            imgs = batch_data['image'].to(device)
            voice = batch_data['voice'].to(device)
            labels = batch_labels.to(device)
            
            # Use a fixed system key for training weights
            keys = torch.full_like(labels, 42)
            
            optimizer.zero_grad()
            # training=True uses Tanh (differentiable)
            embeddings = model(imgs, voice, keys, training=True)
            
            loss = contrastive_loss(embeddings, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return model
