
import torch
from models import EEGSNet
from train import mixup_data

def test_model():
    print("Testing EEGSNet with Coordinate Attention...")
    model = EEGSNet()
    # Mock Input: (Batch=2, Seq=5, Channels=3, Freq=76, Time=60)
    x = torch.randn(2, 5, 3, 76, 60)
    
    try:
        out, aux = model(x)
        print(f"Forward Pass Successful.")
        print(f"Output Shape: {out.shape} (Expected [2, 5])")
        print(f"Aux Output Shape: {aux.shape} (Expected [2, 5, 5])")
    except Exception as e:
        print(f"Forward Pass Failed: {e}")
        raise e

def test_mixup():
    print("\nTesting Mixup Logic...")
    x = torch.randn(4, 5, 3, 76, 60)
    y = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0,0,0,0,0], [1,1,1,1,1]]) # (B, Seq)
    
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0, use_cuda=False)
    
    print(f"Mixed Shape: {mixed_x.shape}")
    print(f"Lambda: {lam}")
    print("Mixup Test Passed.")

if __name__ == "__main__":
    test_model()
    test_mixup()
