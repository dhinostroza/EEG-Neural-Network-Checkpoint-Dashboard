import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os

class SleepDataset(Dataset):
    def __init__(self, spectrograms, labels, seq_len=1):
        """
        spectrograms: Numpy array (N, 3, 76, 60)
        labels: Numpy array (N,)
        seq_len: Length of sequence (e.g. 5)
        """
        self.seq_len = seq_len
        self.data = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
        
        # Valid indices: We can only start a sequence if we have enough previous data?
        # Or future data?
        # Convention: Input [t-seq_len+1 : t+1] -> Label t?
        # Or Input [t : t+seq_len] -> Label t+seq_len?
        # Let's say index `i` corresponds to the END of the sequence.
        # So we range from `seq_len-1` to `N-1`.
        # Total samples = N - seq_len + 1.
        self.valid_indices = list(range(len(labels) - seq_len + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map logical index to start index in raw data
        # idx 0 -> start at 0 (if valid_indices logic above matches)
        # Actually my logic above `valid_indices` is just 0..M. 
        # So `start_idx = idx`. `end_idx = idx + seq_len`.
        
        start_idx = idx
        end_idx = idx + self.seq_len
        
        X_seq = self.data[start_idx:end_idx] # (Seq, 3, 76, 60)
        y_seq = self.labels[start_idx:end_idx] # (Seq,)
        
        if self.seq_len == 1:
            # Backwards compatibility: Return (3, 76, 60) and Label scalar
            return X_seq[0], y_seq[0]
        else:
            return X_seq, y_seq

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        # Load Raw Data
        # Shapes in file (h5py view of matlab):
        # all_spectrograms: (3, 60, 76, 15206) -> (Channels, Time, Freq, N) -- Check assumption!
        # Actually h5py transposes. Matlab (N, 76, 60, 3)? 
        # Inspect output showed: (3, 60, 76, 15206)
        # Matlab typically saves as (H, W, C, N).
        # h5py reads with dimensions reversed? No, h5py reads structured as (C_dim, W_dim, H_dim, N_dim) if it's v7.3?
        # Let's stick to inspecting explicitly again or relying on robust transpose.
        
        # Let's assume the inspection was accurate: (3, 60, 76, 15206)
        # 3 Channels, 60 Time, 76 Freq, 15206 Epochs.
        # We need (N, 3, 76, 60) for PyTorch.
        # Transpose needed: (3, 2, 1, 0) -> (15206, 3, 76, 60).
        
        raw_specs = f['all_spectrograms'][:]
        raw_labels = f['all_labels'][:] # (1, 15206)
        
        # Handle References for Subject Info
        # processed_subject_info is a cell array of structs.
        # accessing it via h5py is tricky (object references).
        
        # Robust Transpose
        # Determine N. It's usually the largest dim or matches label count.
        n_epochs = raw_labels.size
        
        print(f"Raw Specs Shape: {raw_specs.shape}")
        
        # Transpose logic
        if raw_specs.shape[-1] == n_epochs:
            # (3, 60, 76, N) -> (N, 3, 76, 60)
            print("Transposing (3, 60, 76, N) -> (N, 3, 76, 60)")
            spectrograms = raw_specs.transpose(3, 0, 2, 1)
        else:
            # Maybe (N, 3, 60, 76)?
            print("Assuming (N, ...) structure already?")
            spectrograms = raw_specs
            
        # --- Normalization (v1.1) ---
        print("Applying Log-Transform + Z-Score Normalization...")
        # 1. Log-Transform (stabilize variance)
        spectrograms = np.log1p(spectrograms) 
        
        # 2. Z-Score (Global Standardization)
        mean_val = np.mean(spectrograms)
        std_val = np.std(spectrograms)
        print(f"Global Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        spectrograms = (spectrograms - mean_val) / (std_val + 1e-6)
        
        labels = raw_labels.flatten() # Labels are already 0-indexed (0-4)
        
        # Temporary Subject Splitting Logic
        # Since reading nested HDF5 structs for Subject IDs is complex without dedicated tools,
        # we will use a simple heuristic if we can't parse metadata:
        # We know there are 8 subjects.
        # We can try to load `processed_subject_info` but if it fails, we warn.
        
        # Try to reconstruct subject mapping
        # In current Matlab code, total epochs = 15206.
        # 8 subjects. ~1900 per subject.
        # LOSO requires accurate mapping.
        
        # Let's try to extract 'num_valid_epochs' from subject info
        subj_epoch_counts = []
        try:
            ref_array = f['processed_subject_info'][:] # (1, 8)
            for ref in ref_array[0]:
                obj = f[ref]
                # 'num_valid_epochs' field
                # Depending on how it was saved, fields are datasets or groups
                # Note: Matlab structs in v7.3 are groups
                # Let's check keys of the dereferenced object
                # This is tricky without iterative inspection.
                # Simplification: Assume roughly equal distribution or try to read 'num_valid_epochs'
                pass
        except Exception as e:
            print(f"Metadata read error: {e}")

        # FALLBACK: If we can't read metadata, we CANNOT do LOSO accurately.
        # However, for this task, I will simply create a synthetic mapping if read fails,
        # OR I can rely on a separate 'data_split.json' if I had one.
        # Better: I'll use `scipy.io` to read the legacy (non-7.3) file if available? 
        # No, inspection said it's v7.3
        
    return spectrograms, labels

def decode_subject_info(file_path):
    # Specialized function to just get epoch counts using h5py dereferencing
    counts = []
    with h5py.File(file_path, 'r') as f:
        refs = f['processed_subject_info'][0]
        for ref in refs:
            # dereference
            struct = f[ref]
            # num_valid_epochs might be a dataset inside
            # need to read it. 
            # In v7.3, scalar might be [[val]]
            val = struct['num_valid_epochs'][0][0]
            counts.append(int(val))
    return counts

def get_data_splits(file_path):
    specs, labels = load_data(file_path)
    try:
        counts = decode_subject_info(file_path)
        print(f"Subject Epoch Counts: {counts}")
    except:
        print("Could not decode subject info. Using equal split (DANGEROUS).")
        counts = [len(labels) // 8] * 8 # Fallback
    
    # Create start/end indices
    splits = []
    current = 0
    for c in counts:
        splits.append((current, current + c))
        current += c
        
    return specs, labels, splits
