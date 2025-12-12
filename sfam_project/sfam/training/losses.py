import torch
import torch.nn.functional as F

def contrastive_loss(embeddings, labels, margin=0.2):
    """
    Pull same users together, push different users apart.
    Margin 0.2 means: "Punish if impostors are > 20% similar"
    """
    # Normalize
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    
    # Pairwise similarity
    similarity = torch.matmul(norm_emb, norm_emb.T)
    
    # Label mask
    labels = labels.unsqueeze(0)
    mask = (labels == labels.T).float()
    
    # Positive Loss (Minimize distance for same user)
    loss_pos = (1 - similarity) * mask
    
    # Negative Loss (Maximize distance for diff user up to margin)
    loss_neg = torch.clamp(similarity - margin, min=0) * (1 - mask)
    
    return loss_pos.mean() + loss_neg.mean()
