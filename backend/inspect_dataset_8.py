import h5py
import scipy.io
import os

file_path = "../01_matlab_eeg/processed_data/SleepEDFX8_processed_parallel.mat"

print(f"Inspecting {file_path}...")

try:
    # Try h5py first (v7.3)
    with h5py.File(file_path, 'r') as f:
        print("Opened with h5py (v7.3)")
        print("Keys:", list(f.keys()))
        if 'all_spectrograms' in f:
            print("all_spectrograms shape:", f['all_spectrograms'].shape)
        if 'all_labels' in f:
            print("all_labels shape:", f['all_labels'].shape)
            # check unique values
except OSError:
    try:
        # Try scipy.io (v7)
        data = scipy.io.loadmat(file_path)
        print("Opened with scipy.io (v7)")
        print("Keys:", data.keys())
        if 'all_spectrograms' in data:
            print("all_spectrograms shape:", data['all_spectrograms'].shape)
        if 'all_labels' in data:
            print("all_labels shape:", data['all_labels'].shape)
    except Exception as e:
        print(f"Error: {e}")
