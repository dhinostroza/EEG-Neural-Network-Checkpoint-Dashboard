import time
import numpy as np
import torch

def benchmark():
    t_start = time.time()
    
    # Simulate 1000 epochs of data (approx one night)
    # Each epoch is 1520x3 = 4560 float features (Spectrogram flat)
    # or whatever the input dimension is.
    # The app uses 4560 features.
    n_epochs = 3000 # ~1.5 nights
    n_features = 4560
    
    print(f"Generating random data for {n_epochs} epochs...")
    data_values = np.random.randn(n_epochs, n_features).astype(np.float32)
    
    print("Benchmarking Vectorized Preprocessing + Inference Mock...")
    
    batch_size = 64
    total_time = 0
    start_time = time.time()
    
    # 1. Vectorized Preprocessing
    # Mimic app.py logic
    # Mean/Std per sample (axis 1)
    mean = data_values.mean(axis=1, keepdims=True)
    std = data_values.std(axis=1, keepdims=True)
    normalized = (data_values - mean) / (std + 1e-6)
    
    # Reshape
    # (N, 76, 60)
    spectrograms_2d = normalized.reshape(-1, 1, 76, 60)
    
    # Convert to Tensor (CPU)
    t_tensor = torch.from_numpy(spectrograms_2d)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Total Preprocessing Time: {duration:.4f} seconds")
    print(f"Time per epoch: {duration/n_epochs*1000:.4f} ms")
    print(f"Processing Rate: {n_epochs/duration:.2f} epochs/sec")
    
    return duration

if __name__ == "__main__":
    benchmark()
