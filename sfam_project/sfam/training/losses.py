import torch
import torch.nn as nn
import torch.nn.functional as F

class BiometricTripletLoss(nn.Module):
    """
    Triplet Loss optimized for BioHashing.
    Formula: max(0, dist(a, p) - dist(a, n) + margin)
    
    Uses Cosine Distance (1 - cos_sim) because SFAM outputs 
    are normalized/tanh'd vectors.
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def cosine_distance(self, x1, x2):
        # Returns distance between 0 (identical) and 2 (opposite)
        return 1.0 - F.cosine_similarity(x1, x2)

    def forward(self, anchor, positive, negative):
        # 1. Distance between User A (Session 1) and User A (Session 2)
        # Should be SMALL
        dist_pos = self.cosine_distance(anchor, positive)
        
        # 2. Distance between User A and User B
        # Should be LARGE
        dist_neg = self.cosine_distance(anchor, negative)
        
        # 3. Compute Loss
        # We want: dist_neg > dist_pos + margin
        losses = torch.relu(dist_pos - dist_neg + self.margin)
        
        return losses.mean()