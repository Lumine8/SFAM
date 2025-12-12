import torch
import torch.nn.functional as F

def run_evaluation(model, dataset, device="cpu"):
    model.eval()
    
    # Use User 0 as test subject
    idx = 0
    data_A1 = dataset[idx][0]     # Sample 1
    data_A2 = dataset[idx+1][0]   # Sample 2
    
    # Use User 20 as impostor
    data_B = dataset[20][0]
    
    # Helper to prepare inputs
    def prep(d):
        return d['image'].unsqueeze(0).to(device), d['voice'].unsqueeze(0).to(device)

    img_A1, aud_A1 = prep(data_A1)
    img_A2, aud_A2 = prep(data_A2)
    img_B, aud_B = prep(data_B)

    print("\n--- Security Evaluation ---")

    # 1. AUTHENTICATION (Same User, Same Key)
    with torch.no_grad():
        emb1 = model(img_A1, aud_A1, 12345)
        emb2 = model(img_A2, aud_A2, 12345)
        sim = F.cosine_similarity(emb1, emb2).item()
        print(f"1. Auth Check (Target > 0.8):   {sim:.4f} " + ("✅ PASS" if sim > 0.8 else "❌ FAIL"))

    # 2. CANCELLABILITY (Same User, Changed Key)
    with torch.no_grad():
        emb_revoked = model(img_A1, aud_A1, 99999) # New Key
        sim = F.cosine_similarity(emb1, emb_revoked).item()
        print(f"2. Revoke Check (Target < 0.2): {sim:.4f} " + ("✅ PASS" if sim < 0.2 else "❌ FAIL"))

    # 3. IMPOSTOR (Diff User, Same Key)
    with torch.no_grad():
        emb_imp = model(img_B, aud_B, 12345)
        sim = F.cosine_similarity(emb1, emb_imp).item()
        print(f"3. Impostor Check (Target < 0.2): {sim:.4f} " + ("✅ PASS" if sim < 0.2 else "❌ FAIL"))
