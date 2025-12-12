import torch
from data.synthetic_data import SyntheticBiometricDataset
from training.train import train_sfam
from eval.evaluate import run_evaluation

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Generate Data
    print("Generating Dataset...")
    dataset = SyntheticBiometricDataset()
    
    # 2. Train
    model = train_sfam(dataset, epochs=15, device=DEVICE)
    
    # 3. Evaluate
    run_evaluation(model, dataset, device=DEVICE)

if __name__ == "__main__":
    main()
