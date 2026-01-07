
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
DATA_FILE = '/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/SleepEDFX8_processed_parallel.mat'
OUTPUT_IMG = '/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/frontend/public/img/spectrogram_sample.png'

def generate_n1_sample():
    print(f"Loading data from {DATA_FILE}...")
    try:
        with h5py.File(DATA_FILE, 'r') as f:
            # Load Labels
            # Labels are 0-4 (W, N1, N2, N3, REM)
            raw_labels = f['all_labels'][:] # Shape (1, N)
            labels = raw_labels.flatten() # Already 0-indexed based on previous check
            
            # Find indices for N1 (Label 1)
            n1_indices = np.where(labels == 1)[0]
            
            if len(n1_indices) == 0:
                print("Error: No N1 stages found in dataset.")
                return

            # Pick a "nice" random one or the first one
            # Let's pick the 10th one to avoid potentially noisy starts, or just random
            idx = n1_indices[10] if len(n1_indices) > 10 else n1_indices[0]
            
            print(f"Selected N1 Epoch at Index: {idx}")
            
            # Load Spectrogram for this index
            # Dataset: 'all_spectrograms'
            # Shape inspected previously: (3, 60, 76, N) -> (3 channels, 60 time, 76 freq, N epochs)
            # wait, data_loader said: (3, 60, 76, 15206)
            # So we access [:, :, :, idx]
            
            spec_data = f['all_spectrograms'][:, :, :, idx] 
            # Shape: (3, 60, 76) assuming (Channel, Time, Freq) or (Channel, Freq, Time)?
            # Matlab usually saves (H, W, C, N). h5py sees (C, W, H, N)? 
            # Let's check `data_loader.py` logic again:
            # "spectrograms = raw_specs.transpose(3, 0, 2, 1)" -> (N, 3, 76, 60)
            # Original raw: (3, 60, 76, N)
            
            # So for single index `idx`:
            # raw_slice = f['all_spectrograms'][:, :, :, idx] -> (3, 60, 76)
            # We want (Frequency, Time) for plotting.
            # Channel 0 is usually Fpz-Cz.
            
            channel_0 = spec_data[0, :, :] # (60, 76) -> (Time, Freq)?
            # If `data_loader` transposes to (76, 60), then dim 2 is Freq (76).
            # So here: dim 1 (60) is Time, dim 2 (76) is Freq.
            
            # We want Freq on Y, Time on X.
            # So transpose it: (76, 60)
            spec_plot = channel_0.T # (76, 60)
            
            # Plotting
            plt.figure(figsize=(10, 4))
            # origin='lower' puts index 0 at bottom.
            plt.imshow(spec_plot, aspect='auto', origin='lower', cmap='jet')
            plt.colorbar(label='Power (dB)')
            plt.title(f"Real N1 Sleep Stage Spectrogram (Epoch {idx})")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.tight_layout()
            
            # Save
            os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
            plt.savefig(OUTPUT_IMG)
            print(f"Saved N1 Spectrogram to {OUTPUT_IMG}")
            plt.close()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_n1_sample()
