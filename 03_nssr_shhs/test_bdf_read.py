
import mne
import os

BDF_FILE = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/Registro de EEG-HECAM_processed.bdf"

print(f"Reading {BDF_FILE}...")
try:
    # Attempt 1: Standard
    raw = mne.io.read_raw_bdf(BDF_FILE, preload=True, verbose=True)
    print("Success 1")
except Exception as e:
    print(f"Fail 1: {e}")
    
try:
    # Attempt 2: Ignore header date? MNE doesn't have a direct flag for this in read_raw_bdf usually, 
    # but let's see if we can catch it.
    pass
except Exception as e:
    pass
