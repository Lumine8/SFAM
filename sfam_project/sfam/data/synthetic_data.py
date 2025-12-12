import torch
import numpy as np
from torch.utils.data import Dataset

class SyntheticBiometricDataset(Dataset):
    """
    Generates synthetic 'Face' (Image) and 'Voice' (Vector) data.
    """
    def __init__(self, num_users=50, samples_per_user=10):
        self.data = []
        self.labels = []
        
        for user_id in range(num_users):
            # Base identity (random latent vector)
            base_face = np.random.randn(3, 32, 32) # Simulated 32x32 image
            base_voice = np.random.randn(64)       # Simulated voice embedding
            
            for _ in range(samples_per_user):
                # Add noise to simulate different sessions
                face_noise = np.random.normal(0, 0.2, (3, 32, 32))
                voice_noise = np.random.normal(0, 0.2, (64))
                
                self.data.append({
                    'image': torch.FloatTensor(base_face + face_noise),
                    'voice': torch.FloatTensor(base_voice + voice_noise)
                })
                self.labels.append(user_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
