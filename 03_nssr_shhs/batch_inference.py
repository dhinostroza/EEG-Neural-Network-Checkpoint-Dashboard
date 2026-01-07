import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import re
import time
from analyzer import CheckpointAnalyzer
from inference import get_model, load_checkpoint_weights, detect_architecture

# ==============================================================================
# CONFIG & OPTIMIZATIONS
# ==============================================================================
BATCH_SIZE = 128  # Adjust based on VRAM/RAM
LOG_FILE = "processed_files.log"

def get_best_model(base_dir):
    print("Scanning for best model...")
    analyzer = CheckpointAnalyzer(base_dir)
    df = analyzer.scan()
    
    if df.empty:
        raise ValueError("No checkpoints found!")
        
    valid_df = df[df['val_loss'].notna()].copy()
    if valid_df.empty:
        raise ValueError("No checkpoints with valid validation loss found!")
        
    best_row = valid_df.sort_values(by='val_loss', ascending=True).iloc[0]
    return best_row

def preprocess_batch(df_values):
    """
    Vectorized preprocessing for the entire dataframe at once.
    Input: Numpy array of shape (N, 4560)
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
    reshaped = normalized.reshape(-1, 1, 76, 60)
    
    return torch.from_numpy(reshaped)

def load_processed_files():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def mark_as_processed(filename):
    with open(LOG_FILE, 'a') as f:
        f.write(filename + "\n")

def main():
    parser = argparse.ArgumentParser(description="Optimized Batch Inference")
    parser.add_argument("--checkpoint_dir", default="./checkpoint_files", help="Directory containing .ckpt files")
    parser.add_argument("--data_dir", default="./parquet_files", help="Directory containing .parquet files")
    parser.add_argument("--output", default="predictions.sql", help="Output SQL file")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--rename_suffix", default="_processed", help="Suffix to append to processed files (e.g. file.parquet -> file_processed.parquet)")
    
    args = parser.parse_args()
    
    # Enable Cudnn Benchmark for speed
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 1. Select Best Model
    try:
        best_model_meta = get_best_model(args.checkpoint_dir)
        print(f"🏆 Best Model: {best_model_meta['filename']}")
        print(f"   Val Loss: {best_model_meta['val_loss']}")
    except Exception as e:
        print(f"Error selecting model: {e}")
        return

    # 2. Load Model
    model_path = os.path.join(args.checkpoint_dir, best_model_meta['filename'])
    arch = detect_architecture(model_path)
    print(f"   Architecture: {arch}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model = get_model(model_name=arch, num_classes=5, pretrained=False)
    model, _ = load_checkpoint_weights(model, model_path)
    model.to(device)
    model.eval()
    
    # 3. File Discovery & State Filtering
    processed_set = load_processed_files()
    search_pattern = os.path.join(args.data_dir, "**", "*.parquet")
    all_files = glob.glob(search_pattern, recursive=True)
    
    # Filter only relevant files (SC*)
    dataset_files = [f for f in all_files if "SC" in os.path.basename(f)]
    
    # Filter out already processed
    files_to_process = [f for f in dataset_files if os.path.basename(f) not in processed_set]
    
    print(f"Total Files: {len(dataset_files)}")
    print(f"Already Processed: {len(dataset_files) - len(files_to_process)}")
    print(f"Remaining: {len(files_to_process)}")
    
    if args.limit:
        files_to_process = files_to_process[:args.limit]
        print(f"Limit applied: Processing {len(files_to_process)} files.")
    
    if not files_to_process:
        print("No files to process!")
        return
        
    # 4. Processing Loop
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    
    # Open SQL file in append mode if it exists so we don't overwrite previous work
    mode = 'a' if os.path.exists(args.output) else 'w'
    
    with open(args.output, mode) as sql_file:
        if mode == 'w':
            sql_file.write("-- Automated Sleep Stage Predictions (Optimized)\n")
            sql_file.write("CREATE TABLE IF NOT EXISTS sleep_predictions (\n")
            sql_file.write("    id SERIAL PRIMARY KEY,\n")
            sql_file.write("    patient_id VARCHAR(50),\n")
            sql_file.write("    filename VARCHAR(255),\n")
            sql_file.write("    epoch_index INT,\n")
            sql_file.write("    predicted_stage VARCHAR(10),\n")
            sql_file.write("    confidence FLOAT,\n")
            sql_file.write("    model_used VARCHAR(255)\n")
            sql_file.write(");\n\n")
            
        start_time = time.time()
        
        for idx, fpath in enumerate(files_to_process):
            fname = os.path.basename(fpath)
            patient_match = re.search(r"(SC\d+)", fname)
            patient_id = patient_match.group(1) if patient_match else "UNKNOWN"
            
            file_start = time.time()
            try:
                # Load Data
                df = pd.read_parquet(fpath)
                
                # Drop non-feature columns
                cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage'] if c in df.columns]
                if cols_to_drop:
                    feature_data = df.drop(columns=cols_to_drop).values
                else:
                    feature_data = df.values
                
                # VECTORIZED PREPROCESSING (The Speedup Magic ⚡️)
                input_tensor = preprocess_batch(feature_data).to(device)
                
                # Batch Inference
                all_preds = []
                all_confs = []
                
                with torch.no_grad():
                    # Process in chunks to avoid VRAM OOM
                    for i in range(0, len(input_tensor), BATCH_SIZE):
                        batch = input_tensor[i:i+BATCH_SIZE]
                        logits = model(batch)
                        probs = torch.softmax(logits, dim=1)
                        
                        preds = torch.argmax(probs, dim=1).cpu().numpy()
                        confs = torch.max(probs, dim=1).values.cpu().numpy()
                        
                        all_preds.extend(preds)
                        all_confs.extend(confs)

                # Write SQL Block
                sql_file.write(f"-- Data for {fname}\n")
                sql_file.write("INSERT INTO sleep_predictions (patient_id, filename, epoch_index, predicted_stage, confidence, model_used) VALUES\n")
                
                values_list = []
                for i, (pred, conf) in enumerate(zip(all_preds, all_confs)):
                    stage_label = stage_map.get(pred, "Unknown")
                    val_str = f"('{patient_id}', '{fname}', {i}, '{stage_label}', {conf:.4f}, '{best_model_meta['filename']}')"
                    values_list.append(val_str)
                
                # Write chunks
                sql_file.write(",\n".join(values_list))
                sql_file.write(";\n")
                
                # Mark as done
                mark_as_processed(fname)
                
                # RENAME FILE (User Request for Cleanup)
                if args.rename_suffix:
                    try:
                        folder = os.path.dirname(fpath)
                        base_name_no_ext = os.path.splitext(fname)[0]
                        ext = os.path.splitext(fname)[1]
                        new_name = f"{base_name_no_ext}{args.rename_suffix}{ext}"
                        new_path = os.path.join(folder, new_name)
                        os.rename(fpath, new_path)
                        # print(f"   -> Renamed to {new_name}")
                    except Exception as rename_err:
                        print(f"   ⚠️ Failed to rename {fname}: {rename_err}")

                duration = time.time() - file_start
                print(f"[{idx+1}/{len(files_to_process)}] {fname}: {len(all_preds)} epochs in {duration:.2f}s ({len(all_preds)/duration:.1f} epoch/s)")
                
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")
                sql_file.write(f"-- Error processing {fname}: {e}\n")

    total_time = time.time() - start_time
    print(f"\nBatch Processing Complete in {total_time:.2f}s")

if __name__ == "__main__":
    main()
