
import h5py
import numpy as np

DATA_FILE = '/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/SleepEDFX8_processed_parallel.mat'

try:
    with h5py.File(DATA_FILE, 'r') as f:
        labels = f['all_labels'][:]
        print(f"Labels Shape: {labels.shape}")
        print(f"Min Label: {np.min(labels)}")
        print(f"Max Label: {np.max(labels)}")
        print(f"Unique Labels: {np.unique(labels)}")
except Exception as e:
    print(f"Error: {e}")
