import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models.sfam_net import SFAM
from data.synthetic_data import SyntheticBiometricDataset

def visualize_space():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data & Model
    dataset = SyntheticBiometricDataset(num_users=10, samples_per_user=20)
    model = SFAM().to(DEVICE)
    # Note: In a real run, you would load 'sfam_trained.pth'. 
    # For now, we rely on the architecture's untrained separation or you can save/load weights.
    # Let's assume you want to see the "untrained" vs "trained" difference, 
    # but to make this script standalone useful, let's just run forward pass.
    
    embeddings = []
    labels = []
    colors = []
    
    print("Generating embeddings...")
    with torch.no_grad():
        for i in range(len(dataset)):
            data, label = dataset[i]
            img = data['image'].unsqueeze(0).to(DEVICE)
            voice = data['voice'].unsqueeze(0).to(DEVICE)
            
            # 1. Normal User (Key A)
            emb = model(img, voice, user_keys=12345, training=True)
            embeddings.append(emb.cpu().numpy()[0])
            labels.append(f"User {label}")
            colors.append(label)

            # 2. Add ONE Revoked version for User 0
            if label == 0:
                emb_rev = model(img, voice, user_keys=99999, training=True) # Diff Key
                embeddings.append(emb_rev.cpu().numpy()[0])
                labels.append(f"User {label} (Revoked)")
                colors.append(100) # Distinct color

    # 2. Reduce to 2D (PCA)
    print("Computing PCA...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    
    # 3. Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=colors, cmap='tab20', alpha=0.7)
    
    # Highlight User 0 vs User 0 (Revoked)
    plt.title("SFAM Latent Space: Identity vs Key Revocation")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    
    # Annotate User 0 Clusters
    # (Simplified annotation logic)
    plt.text(reduced[0,0], reduced[0,1], "User 0 (Valid)", fontsize=12, fontweight='bold')
    
    # Find the Revoked points (which we assigned color 100)
    revoked_indices = [i for i, c in enumerate(colors) if c == 100]
    if revoked_indices:
        idx = revoked_indices[0]
        plt.text(reduced[idx,0], reduced[idx,1], "User 0 (Revoked)", color='red', fontsize=12, fontweight='bold')

    plt.savefig("sfam_visualization.png")
    print("✅ Plot saved to 'sfam_visualization.png'")
    plt.show()

if __name__ == "__main__":
    visualize_space()