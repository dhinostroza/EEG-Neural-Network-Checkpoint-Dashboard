
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
DATA_FILE = '/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/SleepEDFX8_processed_parallel.mat'
OUTPUT_DIR = '/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/frontend/public/img'

CLASS_MAP = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

def generate_gallery():
    print(f"Loading data from {DATA_FILE}...")
    try:
        with h5py.File(DATA_FILE, 'r') as f:
            raw_labels = f['all_labels'][:] # Shape (1, N)
            labels = raw_labels.flatten()
            
            # Iterate through all classes
            for class_idx, class_name in CLASS_MAP.items():
                print(f"Processing Class: {class_name} ({class_idx})...")
                
                indices = np.where(labels == class_idx)[0]
                if len(indices) == 0:
                    print(f"Warning: No samples found for {class_name}")
                    continue
                
                # Pick a representative sample (skip first few to avoid start artifacts)
                # For Wake, picking something later might be better to avoid initial noise
                selection_idx = 50 if len(indices) > 50 else 0
                idx = indices[selection_idx]
                
                print(f"  Selected Epoch Index: {idx}")
                
                # Load Spectrogram: (3, 60, 76, N) -> slice [:, :, :, idx]
                spec_data = f['all_spectrograms'][:, :, :, idx] # (3, 60, 76)
                
                # Channel 0 (Fpz-Cz)
                channel_0 = spec_data[0, :, :] # (60, 76) -> (Time, Freq)
                
                # Transpose for plotting: (Freq, Time)
                # Original matlab: (60 time, 76 freq). 
                # We want Freq on Y axis.
                spec_plot = channel_0.T # (76, 60)
                
                # Apply Log-Transform for visualization (matches what the model sees/user expects)
                spec_plot = np.log1p(spec_plot)
                
                # Plotting
                # Plotting
                plt.figure(figsize=(10, 6)) # Larger figure for details
                # origin='lower' puts index 0 at bottom (0 Hz)
                im = plt.imshow(spec_plot, aspect='auto', origin='lower', cmap='jet', extent=[0, 60, 0, 76])
                cbar = plt.colorbar(im)
                cbar.set_label('Power (dB)', rotation=270, labelpad=15)
                
                # Full Title (User requested "Add information... remove 'Real' ... add 'Fpz-Cz'")
                plt.title(f"{class_name} Sleep Stage (Fpz-Cz) Spectrogram (Epoch {idx})", fontsize=12, fontweight='bold')
                
                # Restore Axes Labels
                plt.ylabel('Frequency (Hz)', fontsize=10)
                plt.xlabel('Time (s)', fontsize=10)
                plt.tight_layout()
                
                # Save
                filename = f"spec_{class_name}.png"
                out_path = os.path.join(OUTPUT_DIR, filename)
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
                print(f"  Saved to {out_path}")
                plt.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_gallery()
