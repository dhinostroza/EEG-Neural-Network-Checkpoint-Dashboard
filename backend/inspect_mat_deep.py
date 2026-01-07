
import scipy.io
import os
import numpy as np

files = [
    "../01_matlab_eeg/processed_data/Stage1_CNN_Training_Results_SleepEDFX_SC40_processed_parallel.mat_Sequential.mat",
    "../01_matlab_eeg/processed_data/Stage2_LSTM_Training_Results_SleepEDFX_SC40_processed_parallel.mat.mat"
]

def search_keys(obj, path="/"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'conf' in k.lower() or 'mat' in k.lower() or 'cm' == k.lower():
                print(f"FOUND POTENTIAL MATCH: {path}{k} (Type: {type(v)})")
            
            if isinstance(v, (dict, np.void)) or (isinstance(v, np.ndarray) and v.dtype.names):
                 # Recurse if struct/dict
                 pass 
            
            # If numpy array of objects, might need to iterate?
            
    elif isinstance(obj, np.void): # Matlab struct
         for k in obj.dtype.names:
            if 'conf' in k.lower() or 'mat' in k.lower() or 'cm' == k.lower():
                print(f"FOUND POTENTIAL MATCH: {path}{k}")
            # Recurse?

for fpath in files:
    print(f"Scanning {fpath}...")
    try:
        mat = scipy.io.loadmat(fpath)
        # Search root
        for k in mat.keys():
            if k.startswith('__'): continue
            print(f"  Root key: {k}")
            val = mat[k]
            # Try to look inside root struct
            if isinstance(val, np.ndarray) and val.shape[0] > 0:
                print(f"    Scanning inside {k} (first element)...")
                # Assume it's a struct array
                if val.dtype.names:
                    print(f"    Fields: {val.dtype.names}")
                    for field in val.dtype.names:
                        if 'conf' in field.lower() or 'matrix' in field.lower():
                            print(f"    !!! FOUND: {field}")
                        
                        # Look one level deeper if possible
                        # inner = val[0,0][field]
                        # ...
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 20)
