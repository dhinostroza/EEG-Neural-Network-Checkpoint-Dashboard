import scipy.io
import h5py
import os
import numpy as np

# Inspecting Stage 1 and Stage 2 for metadata
files = [
    "../01_matlab_eeg/processed_data/Stage1_CNN_Training_Results_SleepEDFX_SC40_processed_parallel.mat_Sequential.mat",
    "../01_matlab_eeg/processed_data/Stage2_LSTM_Training_Results_SleepEDFX_SC40_processed_parallel.mat.mat"
]

for fp in files:
    print(f"\n--- Scanning {os.path.basename(fp)} ---")
    try:
        data = scipy.io.loadmat(fp)
        # Find results key
        key = 'results_cnn' if 'results_cnn' in data else 'results_stage2' if 'results_stage2' in data else None
        
        if key:
            res = data[key]
            print(f"Result struct found: {key}")
            # Try to access TrainInfo
            info_key = 'TrainInfo' if 'TrainInfo' in res.dtype.names else 'TrainInfoLSTM' if 'TrainInfoLSTM' in res.dtype.names else None
            
            if info_key:
                info = res[0,0][info_key]
                print(f"Found {info_key}. Fields:")
                print(info.dtype.names)
                
                # Print values for non-array fields (to find scalars like LearnRate, Epochs etc)
                for field in info.dtype.names:
                    val = info[0,0][field]
                    if val.size < 10: # Only print small stuff
                        print(f"  {field}: {val.flatten()}")
                    else:
                        print(f"  {field}: [Array size {val.shape}]")
    except Exception as e:
        print(f"Error: {e}")
