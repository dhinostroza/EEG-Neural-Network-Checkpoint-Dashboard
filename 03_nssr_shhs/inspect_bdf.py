import mne
import pandas as pd
import numpy as np
import os

# Path to BDF
bdf_path = "Registro de EEG-HECAM_processed.bdf"
output_parquet = "Registro_EEG_HECAM.parquet"

if not os.path.exists(bdf_path):
    print(f"File not found: {bdf_path}")
    exit(1)

print(f"Attempting to read {bdf_path} with MNE...")

try:
    # Try reading raw
    raw = mne.io.read_raw_bdf(bdf_path, preload=True)
    print("Read success!")
    print(raw.info)
    
    # Simple extraction for demo/test purposes - we need to cut into 30s epochs
    # This is a simplified logic from the app's preprocessor
    sfreq = raw.info['sfreq']
    print(f"Sampling Rate: {sfreq} Hz")
    
    # Assuming we just want to verify it can be read and maybe convert it
    # If the app failed, maybe it's 0 bytes or corrupt?
    # But files listing said 4.8MB. 
    # A 4.8MB BDF is very small for overnight EEG. Might be just a few minutes?
    duration_sec = raw.times[-1]
    print(f"Duration: {duration_sec} seconds ({duration_sec/60:.2f} mins)")
    
    # If it's valid, let's try to make a dummy parquet with just the raw data for inference?
    # Real preprocessing is complex (filtering, spectrograms). 
    # I should check if there is a pre-processing script available to reuse.
    # 'pre_shhs_edf2parquet.py' seems relevant.
    
except Exception as e:
    print(f"Failed to read BDF: {e}")
    # Check file header or size
    with open(bdf_path, 'rb') as f:
        head = f.read(100)
        print(f"First 100 bytes: {head}")
