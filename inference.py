import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
import argparse
import sys
import os
import logging
import warnings

# Optional imports for EDF conversion
try:
    import mne
    from scipy import signal
    from skimage.transform import resize
    HAS_EDF_LIBS = True
except ImportError:
    HAS_EDF_LIBS = False

# ==============================================================================
# 1. MODEL DEFINITION
# ==============================================================================
def get_model(model_name='convnext_tiny', num_classes=5, pretrained=False):
    """
    Creates a ConvNeXT model adapted for sleep stage classification.
    Matches the architecture defined in the training notebook.
    """
    # ... (same logic as before for model creation)
    # Defaulting to tiny 
    if 'base' in model_name:
         timm_name = 'convnextv2_base' 
    elif 'tiny' in model_name:
         timm_name = 'convnextv2_tiny'
    else:
         # Fallback default or mapped from other names
         timm_name = 'convnextv2_tiny'

    # Create model without silent fallback. If this fails, we want to know.
    model = timm.create_model(timm_name, pretrained=pretrained)

    # Modify first layer for 1 channel input
    original_conv = model.stem[0]
    new_first_conv = nn.Conv2d(1, original_conv.out_channels, 
                               kernel_size=original_conv.kernel_size, 
                               stride=original_conv.stride, 
                               padding=original_conv.padding, 
                               bias=(original_conv.bias is not None))
    
    model.stem[0] = new_first_conv
    
    # Modify head
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ==============================================================================
# 2. PREPROCESSING
# ==============================================================================
def preprocess_spectrogram(spectrogram_flat):
    spectrogram_flat = spectrogram_flat.astype(np.float32)
    mean, std = spectrogram_flat.mean(), spectrogram_flat.std()
    spectrogram_normalized = (spectrogram_flat - mean) / (std + 1e-6)
    spectrogram_2d = spectrogram_normalized.reshape(1, 76, 60)
    return torch.from_numpy(spectrogram_2d)

# ==============================================================================
# 3. EDF CONVERSION (Derived from user script)
# ==============================================================================
def convert_edf_to_parquet(edf_path, output_path):
    """
    Converts a single .edf file to .parquet spectrograms for inference.
    Does NOT require XML annotations (uses dummy labels).
    """
    if not HAS_EDF_LIBS:
        raise ImportError("Missing libraries for EDF conversion. Please install: mne, scipy, scikit-image")
        
    try:
        # Suppress MNE info messages
        mne.set_log_level('WARNING') 
        
        # 1. Read EDF
        # verbose=False to keep logs clean
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # 2. Pick Channels
        # The user script uses ['EEG']. We should try to be robust if 'EEG' isn't exact.
        # But for now, let's stick to the script's logic.
        if 'EEG' in raw.ch_names:
            raw.pick_channels(['EEG'])
        elif 'EEG(sec)' in raw.ch_names:
             raw.pick_channels(['EEG(sec)'])
        else:
            # Fallback: select first channel if 'EEG' not found? 
            # Or raise error. Let's try to map common names or just take the first one.
            # SHHS usually has 'EEG'.
            available = raw.ch_names
            print(f"Warning: 'EEG' channel not found in {available}. Using first channel: {available[0]}")
            raw.pick_channels([available[0]])
            
        # 3. Resample to 64Hz (from script)
        raw.resample(64, npad="auto")
        
        # 4. Epoching (30 seconds * 64 Hz)
        # Note: raw.get_data() returns (n_channels, n_times)
        # We assume 1 channel.
        data_full = raw.get_data() # (1, total_samples)
        total_samples = data_full.shape[1]
        epoch_len = 30 * 64
        
        # Drop residual samples
        n_epochs = total_samples // epoch_len
        epochs_data = data_full[0, :n_epochs * epoch_len].reshape(-1, epoch_len)
        
        # 5. Spectrogram Generation
        spectrograms = []
        for epoch_data in epochs_data:
            f, t, Sxx = signal.spectrogram(epoch_data, fs=64, nperseg=128, noverlap=64)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            Sxx_resized = resize(Sxx_db, (76, 60), anti_aliasing=True, mode='reflect')
            spectrograms.append(Sxx_resized.flatten())
            
        # 6. Create DataFrame
        df_out = pd.DataFrame(spectrograms)
        # Add dummy valid dummy labels (-1) since we don't have annotations
        df_out['label'] = -1 
        
        # Save
        df_out.to_parquet(output_path)
        return True, f"Converted {n_epochs} epochs."
        
    except Exception as e:
        return False, str(e)

# ==============================================================================
# 4. MAIN INFERENCE
# ==============================================================================
def load_checkpoint_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint # fallback if it IS the state dict
        
    # Remove 'model.' prefix if present (common in Lightning)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('loss_fn'):
            continue # Skip loss function weights stored in checkpoint
            
        if k.startswith('model.'):
            new_key = k[6:] # remove 'model.'
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    # strict=False allows ignoring keys like 'loss_fn.weight' if they slipped through
    # or if there are slight mismatches in head layers etc.
    keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")
    
    # Try to extract model name from Hparams if available, else guess from filename
    ckpt_model_name = checkpoint.get('hyper_parameters', {}).get('model_name', 'convnext_tiny')
    return model, ckpt_model_name

def detect_architecture(checkpoint_path):
    """
    Robustly detects if the checkpoint is ConvNeXt Tiny, Base, etc.
    Prioritizes filename hints over internal metadata (which can be unreliable).
    """
    filename = os.path.basename(checkpoint_path).lower()
    
    # 1. Trust Filename First
    if 'convnext_base' in filename or 'base' in filename:
        print(f"Info: Detected 'base' in filename '{filename}'. Forcing ConvNeXt Base.")
        return 'convnext_base'
    if 'convnext_tiny' in filename or 'tiny' in filename:
        return 'convnext_tiny'
        
    # 2. Fallback to Metadata
    try:
        tmp = torch.load(checkpoint_path, map_location='cpu')
        return tmp.get('hyper_parameters', {}).get('model_name', 'convnext_tiny')
    except:
        return 'convnext_tiny'

def run_inference(checkpoint_path, parquet_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # We need to instantiate the model structure FIRST. 
    model_name = detect_architecture(checkpoint_path)
    print(f"Detected architecture: {model_name}")
    
    # Create model
    model = get_model(model_name=model_name, num_classes=5, pretrained=False)
    
    # Load weights
    model, _ = load_checkpoint_weights(model, checkpoint_path)
    model.eval()
    
    # Load Data
    print(f"Loading data: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Found {len(df)} samples.")
    
    predictions = []
    
    with torch.no_grad():
        for i in range(len(df)):
            row = df.iloc[i]
            # Handle different label column potential names or absence
            cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage'] if c in df.columns]
            if cols_to_drop:
                flat_data = row.drop(cols_to_drop).values
            else:
                flat_data = row.values
                
            input_tensor = preprocess_spectrogram(flat_data)
            input_tensor = input_tensor.unsqueeze(0) # (1, 1, 76, 60)
            
            logits = model(input_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            predictions.append(pred_idx)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(df)}...", end='\r')
                
    print(f"\nInference complete.")
    
    # Add predictions
    df['predicted_mid'] = predictions
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    df['predicted_label'] = df['predicted_mid'].map(stage_map)
    
    output_path = parquet_path.replace(".parquet", "_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Sleep Stage Classifier")
    parser.add_argument("checkpoint", help="Path to .ckpt file")
    parser.add_argument("input_file", help="Path to input .parquet file (spectrograms)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.checkpoint) and os.path.exists(args.input_file):
        run_inference(args.checkpoint, args.input_file)
    else:
        print("Error: Checkpoint or Input file not found.")
