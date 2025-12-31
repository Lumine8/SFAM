import torch
import torch.nn as nn

class IAM_Module(nn.Module):
    """
    Irreversible Abstraction Module (IAM)
    Uses BioHashing (Random Projection + Binarization)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, fused_features, user_seed, training=False):
        """
        fused_features: [batch, dim] 
        user_seed: int (The revokable key)
        """
        device = fused_features.device
        
        # 🛑 FIX: Use a Local Generator
        # This creates randomness ONLY for this specific matrix,
        # leaving the global PyTorch training random state untouched.
        gen = torch.Generator(device=device)
        gen.manual_seed(int(user_seed)) 
        
        # 1. Generate User-Specific Projection Matrix
        # We don't need gradients for the random matrix itself (it's fixed per user)
        with torch.no_grad():
            projection = torch.randn(self.input_dim, self.output_dim, generator=gen, device=device)
        
        # 2. Project
        # [1, 128] x [128, 256] -> [1, 256]
        projected = torch.matmul(fused_features, projection)
        
        # 3. Non-Linearity (Relaxation)
        if training:
            # Tanh allows gradients to flow back to the encoders
            return torch.tanh(projected) 
        else:
            # Sign creates the hard binary hash for the database
            return torch.sign(projected)