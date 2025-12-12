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
        training: bool (If True, use Tanh for gradients. If False, use Sign for bits)
        """
        # 1. Generate User-Specific Projection Matrix
        # Note: We use the seed to deterministically create the matrix on the fly
        torch.manual_seed(user_seed) 
        projection = torch.randn(self.input_dim, self.output_dim).to(fused_features.device)
        
        # 2. Project
        projected = torch.matmul(fused_features, projection)
        
        # 3. Non-Linearity
        if training:
            return torch.tanh(projected) # Differentiable approximation
        else:
            return torch.sign(projected) # Hard bits for security
