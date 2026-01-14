
import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import re
import time
import matplotlib.pyplot as plt
import sys

# Add path to import inference util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import preprocess_spectrogram

# ==============================================================================
# CONFIG
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "ensemble_model_scripted.pt")
BATCH_SIZE = 128
SQL_FILE = "predictions.sql"
LOG_FILE = "processed_ensemble_files.log"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def preprocess_batch_features(df_values):
    """
    Vectorized preprocessing for the entire dataframe at once.
    Input: Numpy array of shape (N, 4560) or similar
    Output: Torch tensor of shape (N, 1, 76, 60)
    """
    # 1. Cast to float32
    data = df_values.astype(np.float32)
    
    # 2. Compute Mean and Std per sample (axis 1)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    
    # 3. Normalize
    normalized = (data - mean) / (std + 1e-6)
    
    # 4. Reshape to (N, 1, 76, 60)
    try:
        reshaped = normalized.reshape(-1, 1, 76, 60)
    except ValueError:
        # Fallback for unexpected shapes
        print(f"    ⚠️ Warning: Data shape {data.shape} mismatch. Padding or truncating...")
        # (Naive fallback not implemented for speed, assume correct Parquet)
        raise
    
    return torch.from_numpy(reshaped)

def generate_and_save_hypnogram(filename, predictions, confidence=None, title="Hypnogram"):
    """Generates and saves a Hypnogram PNG."""
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Data
        x = range(len(predictions))
        y = predictions
        
        # Plot Predictions
        ax.step(x, y, where='post', label='Ensemble Prediction', color='#2196f3', linewidth=1.5)
        
        if confidence is not None:
            # Maybe overlay confidence as alpha or separate plot? 
            # For now keep simple as per app.py
            pass

        # Formatting
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
        ax.set_title(title)
        ax.set_xlabel("Epoch (30s)")
        ax.set_ylabel("Sleep Stage")
        ax.invert_yaxis() 
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Save
        png_path = os.path.splitext(filename)[0] + "_hypnogram.png"
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=100)
        plt.close(fig)
        return png_path
        
    except Exception as e:
        print(f"Error generating PNG: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Ensemble Batch Inference")
    parser.add_argument("--pattern", default="parquet_files/shhs*.parquet", help="Glob pattern for files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("--target", default=None, help="Specific filename to process (overrides pattern)")
    parser.add_argument("--device", default=None, help="Force device (mps, cuda, cpu)")
    
    args = parser.parse_args()
    
    # 1. Setup Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"🚀 Using Device: {device}")
    
    # 2. Load TorchScript Model
    if not os.path.exists(SCRIPT_PATH):
        print(f"❌ Error: Scripted model not found at {SCRIPT_PATH}")
        return
        
    print(f"Loading Model: {SCRIPT_PATH} ...")
    model = torch.jit.load(SCRIPT_PATH)
    model.to(device)
    model.eval()
    
    # 3. Find Files
    if args.target:
        # Check if full path or just name
        if os.path.exists(args.target):
            files = [args.target]
        else:
            # Try finding in parquet_files/
            sub = os.path.join(BASE_DIR, "parquet_files", args.target)
            if os.path.exists(sub):
                files = [sub]
            else:
                 print(f"❌ Target file not found: {args.target}")
                 return
    else:
        files = glob.glob(os.path.join(BASE_DIR, args.pattern))
        files = sorted(files)
        
    if not files:
        print("No files found matching pattern.")
        return
        
    if args.limit:
        files = files[:args.limit]
        
    print(f"files to process: {len(files)}")
    print("-" * 50)
    
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    
    # 4. Processing Loop
    processed_count = 0
    
    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"Processing: {fname} ...")
        
        try:
            start_t = time.time()
            
            # Load Data
            df = pd.read_parquet(fpath)
             # Drop non-feature columns
            cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage', 'true_label'] if c in df.columns]
            if cols_to_drop:
                feature_data = df.drop(columns=cols_to_drop).values
            else:
                feature_data = df.values
                
            # Preprocess
            input_tensor = preprocess_batch_features(feature_data)
            input_tensor = input_tensor.to(device)
            
            # Inference
            all_preds = []
            all_confs = []
            
            with torch.no_grad():
                # Loop in batches
                for i in range(0, len(input_tensor), BATCH_SIZE):
                     batch = input_tensor[i:i+BATCH_SIZE]
                     
                     # Model forward (Scripted model returns probabilities or logits? 
                     # The scripted model exported from ensemble_logic usually performs the averaging internally 
                     # IF it was scripted from the Python class. 
                     # BUT usually we script the individual models or a wrapper.
                     # Let's assume the scripted model returns generic output.
                     # Wait, checking app.py... 
                     # It says: "It returns probabilities. Argmax(probs) is valid."
                     
                     output = model(batch)
                     
                     # If output is logits, softmax. If probs, just use.
                     # Safer to softmax if not sure, but if it's already probs, softmaxing twice squashes.
                     # Assuming Logits for safety unless proven otherwise from app.py comments (line 2083 says it returns probs).
                     # Actually line 2084 app.py comment says: "It returns probabilities."
                     # Let's verify sum.
                     # However, safe to take argmax either way.
                     
                     probs = output # Assuming it is probabilities as per comment
                     
                     # If validation needed, check output.sum(dim=1).mean() -> should be 1.0 approx
                     
                     preds = torch.argmax(probs, dim=1).cpu().numpy()
                     confs = torch.max(probs, dim=1).values.cpu().numpy()
                     
                     all_preds.extend(preds)
                     all_confs.extend(confs)
            
            # Save to SQL
            mode = 'a'
            with open(SQL_FILE, mode) as f:
                header = "INSERT INTO sleep_predictions (patient_id, filename, epoch_index, predicted_stage, confidence, model_used) VALUES\n"
                f.write(f"-- Batch Data for {fname}\n")
                f.write(header)
                
                rows = []
                for i, (p, c) in enumerate(zip(all_preds, all_confs)):
                    lbl = stage_map.get(p, "Unknown")
                    # Patient ID extract
                    pat_match = re.search(r"(shhs\d+-\d+)", fname)
                    pid = pat_match.group(1) if pat_match else "UNKNOWN"
                    
                    row = f"('{pid}', '{fname}', {i}, '{lbl}', {c:.4f}, 'Ensemble_Scripted')"
                    rows.append(row)
                
                f.write(",\n".join(rows))
                f.write(";\n")
                
            # Save PNG
            png_path = generate_and_save_hypnogram(fpath, all_preds, title=f"Ensemble: {fname}")
            
            duration = time.time() - start_t
            print(f"   ✅ Done in {duration:.2f}s. Saved SQL & PNG ({png_path})")
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nBatch Completed. {processed_count}/{len(files)} success.")

if __name__ == "__main__":
    main()
