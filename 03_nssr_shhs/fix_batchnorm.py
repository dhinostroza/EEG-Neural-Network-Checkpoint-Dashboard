import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import glob

# Add path to import inference util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import get_model, load_checkpoint_weights, preprocess_spectrogram, detect_architecture
from torch.utils.data import DataLoader, Dataset

# --- Configuration ---
BASE_DIR = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs"
INPUT_CKPT = os.path.join(BASE_DIR, "checkpoint_files/2000 files/2026-01-10_Modelo_Definitivo_Fusionado.ckpt")
OUTPUT_CKPT = os.path.join(BASE_DIR, "checkpoint_files/2000 files/2026-01-11_Modelo_Fusionado_Recalibrated.ckpt")
DATA_PATTERN = os.path.join(BASE_DIR, "shhs*_processed.parquet") # Use SHHS data for calibration
BATCH_SIZE = 32
MAX_BATCHES = 200 # Don't need infinite data, just enough to stabilize stats (approx 6000 samples)

# --- Dataset ---
class RecalibrationDataset(Dataset):
    def __init__(self, file_pattern):
        files = glob.glob(file_pattern)
        print(f"Found {len(files)} files for recalibration.")
        
        self.data = []
        for f in files[:5]: # Use first 5 files to be fast but representative
            try:
                df = pd.read_parquet(f)
                # Drop non-feature columns
                cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage', 'true_label'] if c in df.columns]
                vals = df.drop(columns=cols_to_drop).values.astype(np.float32)
                self.data.append(vals)
                print(f"Loaded {len(vals)} samples from {os.path.basename(f)}")
            except Exception as e:
                print(f"Skipping {f}: {e}")
                
        if self.data:
            self.data = np.concatenate(self.data, axis=0)
        else:
            self.data = np.array([])
            
        print(f"Total calibration samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flat = self.data[idx]
        return preprocess_spectrogram(flat)

# --- Calibration Logic ---
def recalibrate_bn(model, loader):
    print("Recalibrating BatchNorm statistics...")
    model.train() # BN layers update running stats in train mode
    
    # Reset running stats for fresh calibration?
    # Sometimes it's better to reset to calculate fresh mean/var from the SHHS data
    # instead of doing a moving average with the old skewed stats.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None # Use simple average (cumulative), not exponential moving average
            
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
        
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            _ = model(batch)
            count += 1
            if count % 10 == 0:
                print(f"Processed batch {count}...", end='\r')
            if count >= MAX_BATCHES:
                break
    print("\nCalibration finished.")

def save_checkpoint(model, original_ckpt_path, output_path):
    print("Saving new checkpoint...")
    # Load original container to preserve metadata/hyperparams
    checkpoint = torch.load(original_ckpt_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        # Update the state_dict with the new model weights (which contain updated BN stats)
        # Note: model.state_dict() keys might lack "model." prefix if we removed it during load
        # Let's handle prefix matching carefully.
        
        new_state = model.state_dict()
        old_state_keys = checkpoint['state_dict'].keys()
        
        # Check if old keys have 'model.' prefix
        has_prefix = any(k.startswith('model.') for k in old_state_keys)
        
        final_state = {}
        for k, v in new_state.items():
            save_key = f"model.{k}" if has_prefix else k
            final_state[save_key] = v
            
        checkpoint['state_dict'] = final_state
        torch.save(checkpoint, output_path)
        print(f"Saved to: {output_path}")
    else:
        print("Error: Original checkpoint structure unknown.")

def main():
    if not os.path.exists(INPUT_CKPT):
        print(f"Input checkpoint not found: {INPUT_CKPT}")
        return

    # 1. Setup Data
    dataset = RecalibrationDataset(DATA_PATTERN)
    if len(dataset) == 0:
        print("No data found for calibration!")
        return
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Setup Model
    model_name = detect_architecture(INPUT_CKPT)
    print(f"Architecture: {model_name}")
    model = get_model(model_name=model_name, num_classes=5)
    model, _ = load_checkpoint_weights(model, INPUT_CKPT)
    
    # 3. Recalibrate
    recalibrate_bn(model, loader)
    
    # 4. Save
    save_checkpoint(model, INPUT_CKPT, OUTPUT_CKPT)

if __name__ == "__main__":
    main()
