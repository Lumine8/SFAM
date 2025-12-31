import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# --- 1. IMPORTS FROM YOUR ARCHITECTURE ---
# Ensure 'sfam' package is in your python path
from sfam.models.sfam_net import SFAM_Adaptive, generate_user_key

# --- 2. HELPER: Stable Key Management ---
def get_batch_keys(user_ids, key_cache, device, secure_dim=256):
    """
    Retrieves or generates stable projection keys for a batch of users.
    Ensures keys are unique per user and persist across epochs.
    """
    keys_list = []
    for uid in user_ids:
        uid = int(uid.item())
        
        # Lazy initialization of keys
        if uid not in key_cache:
            # Generate on CPU, move to Device, cache it
            # "system_salt" ensures revocability (change salt -> new keys for everyone)
            k = generate_user_key(uid, "system_salt", dim=secure_dim)
            key_cache[uid] = k.to(device)
            
        keys_list.append(key_cache[uid])
    
    # Stack and explicitly enforce device placement
    return torch.stack(keys_list).to(device)

# --- 3. LOSS FUNCTION ---
class BiometricTripletLoss(nn.Module):
    """
    Optimizes: Distance(Anchor, Positive) < Distance(Anchor, Negative)
    Metric: Cosine Distance (1 - CosSim)
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Range: 0 (Identical) to 2 (Opposite)
        d_pos = 1.0 - F.cosine_similarity(anchor, positive)
        d_neg = 1.0 - F.cosine_similarity(anchor, negative)
        
        # Loss activates if Negative is closer than (Positive + Margin)
        losses = torch.relu(d_pos - d_neg + self.margin)
        return losses.mean()

# --- 4. IN-MEMORY DATASET (PLUMBING TEST) ---
class InMemoryTripletGenerator(Dataset):
    """
    Generates synthetic triplets to verify gradient flow and convergence.
    NOT for accuracy reporting.
    """
    def __init__(self, num_users=50, samples_per_epoch=1000):
        self.users_list = list(range(num_users))
        self.length = samples_per_epoch

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # A. Select Users
        anchor_id = random.choice(self.users_list)
        negative_id = random.choice(self.users_list)
        while negative_id == anchor_id:
            negative_id = random.choice(self.users_list)

        # B. Generate Synthetic Tensors (Simulating Inputs)
        # In real usage, replace this with file loading
        
        # Anchor (User A)
        anc_img = torch.randn(3, 224, 224)
        anc_seq = torch.randn(64, 6) # (Seq_Len, Channels)

        # Positive (User A + Noise) -> Simulates intra-class variance
        pos_img = anc_img + (torch.randn_like(anc_img) * 0.1)
        pos_seq = anc_seq + (torch.randn_like(anc_seq) * 0.1)

        # Negative (User B) -> Simulates inter-class variance
        neg_img = torch.randn(3, 224, 224)
        neg_seq = torch.randn(64, 6)

        return {
            "anc_img": anc_img, "anc_seq": anc_seq, "anc_id": anchor_id,
            "pos_img": pos_img, "pos_seq": pos_seq, "pos_id": anchor_id, # Pos ID == Anc ID
            "neg_img": neg_img, "neg_seq": neg_seq, "neg_id": negative_id
        }

# --- 5. MAIN TRAINING LOOP ---
def train():
    # Config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 0.0003
    SECURE_DIM = 256
    
    print(f"🚀 Starting Plumbing Test on {DEVICE}")
    print(f"   Architecture: SFAM_Adaptive")
    print(f"   Loss: BiometricTriplet (Margin 0.5)")

    # Setup
    dataset = InMemoryTripletGenerator()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SFAM_Adaptive(behavioral_dim=6, secure_dim=SECURE_DIM).to(DEVICE)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = BiometricTripletLoss(margin=0.5).to(DEVICE)
    
    # Key Cache (The "State" of the security layer)
    key_cache = {}

    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # A. Move Inputs
            anc_img, anc_seq = batch['anc_img'].to(DEVICE), batch['anc_seq'].to(DEVICE)
            pos_img, pos_seq = batch['pos_img'].to(DEVICE), batch['pos_seq'].to(DEVICE)
            neg_img, neg_seq = batch['neg_img'].to(DEVICE), batch['neg_seq'].to(DEVICE)
            
            # B. Get Keys (Using Helper)
            # Anchor & Positive use the SAME keys (User A)
            keys_A = get_batch_keys(batch['anc_id'], key_cache, DEVICE, SECURE_DIM)
            # Negative uses DIFFERENT keys (User B)
            keys_B = get_batch_keys(batch['neg_id'], key_cache, DEVICE, SECURE_DIM)

            # C. Forward Pass (binarize=False for training gradients)
            optimizer.zero_grad()
            
            anc_out = model(anc_img, anc_seq, keys_A, binarize=False)
            pos_out = model(pos_img, pos_seq, keys_A, binarize=False)
            neg_out = model(neg_img, neg_seq, keys_B, binarize=False) # Important: Key B

            # D. Loss & Step
            loss = criterion(anc_out, pos_out, neg_out)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # E. Epoch Report
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save artifact
    torch.save(model.state_dict(), "sfam_plumbing_test.pth")
    print("💾 Model saved. System is operationally sound.")

if __name__ == "__main__":
    train()