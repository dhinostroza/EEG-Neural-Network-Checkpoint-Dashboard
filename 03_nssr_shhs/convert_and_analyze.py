import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.jit
import matplotlib.pyplot as plt
import sys
import time

# Force libraries check
try:
    import mne
    import scipy
    import skimage
    HAS_EDF_LIBS = True
except ImportError:
    HAS_EDF_LIBS = False

# Add path to import inference util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import convert_edf_to_parquet, get_model, load_checkpoint_weights, preprocess_spectrogram, detect_architecture


# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PNG_DIR = os.path.join(BASE_DIR, "png")
os.makedirs(PNG_DIR, exist_ok=True)

# Input Files
BDF_FILE = os.path.join(BASE_DIR, "Registro de EEG-HECAM_processed.bdf")
SC_PATTERN = os.path.join(BASE_DIR, "parquet_files/SC*.parquet") # Assuming they are in parquet_files or root?
# Check where SC files are. Based on file list, they are in root of 03_nssr_shhs
SC_PATTERN_ROOT = os.path.join(BASE_DIR, "SC*.parquet")

# Models
CKPT_OLD = os.path.join(BASE_DIR, "checkpoint_files/2000 files/2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt")
ENSEMBLE_SCRIPT = os.path.join(BASE_DIR, "checkpoint_files/2000 files/2026-01-11_convnext_base_ensemble.pt")

SQL_FILE = os.path.join(BASE_DIR, "comparative_predictions.sql")

STAGE_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Ground Truth Paths
HYPNO_PATH_CASSETTE = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/Bases_datos/sleep-edf-database-expanded-1.0.0/sleep-cassette"
HYPNO_PATH_TELEMETRY = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/Bases_datos/sleep-edf-database-expanded-1.0.0/sleep-telemetry"

def load_external_hypnogram(filename, expected_epochs):
    """
    Attempts to load the matching Hypnogram EDF for a given SC file.
    """
    if not HAS_EDF_LIBS:
        print("  Warning: MNE not available, cannot load external hypnogram.")
        return None
        
    # filename: SC4001E.parquet or SC4001E_processed.parquet
    # Hypnogram format: SC4001EC-Hypnogram.edf
    
    # 1. Extract Subject ID
    base = os.path.basename(filename).replace(".parquet", "").replace("_processed", "")
    # Usually SC4001E -> SC4001E...
    # The file pattern is usually SC4001E0-PSG.edf -> SC4001EC-Hypnogram.edf
    # But we only have SC4001E.
    # We should search for *SC4001*Hypnogram.edf
    
    # Take first 6 chars? SC4001
    subject_id = base[:6] 
    
    pattern = f"{subject_id}*Hypnogram.edf"
    
    # Search in both folders
    candidates = []
    candidates.extend(glob.glob(os.path.join(HYPNO_PATH_CASSETTE, pattern)))
    candidates.extend(glob.glob(os.path.join(HYPNO_PATH_TELEMETRY, pattern)))
    
    if not candidates:
        print(f"  No Hypnogram file found for {base} (Pattern: {pattern})")
        return None
        
    hypno_path = candidates[0]
    print(f"  Found External Hypnogram: {os.path.basename(hypno_path)}")
    
    try:
        annot = mne.read_annotations(hypno_path)
        
        # Parse Annotations to 30s epochs
        # MNE Annotations: onset, duration, description
        
        # Standard Sleep-EDF logic:
        # We need to construct a vector of 30s labels.
        # But we must be careful: Sleep-EDF annotations often include "Sleep stage ?" or long distinct periods.
        # Also, the recording start time might differ? 
        # Usually for Cassette, they align from t=0 of the PSG file.
        
        labels_30s = []
        
        # Duration of the hypnogram coverage
        total_duration = annot.onset[-1] + annot.duration[-1]
        num_epochs_hypno = int(total_duration // 30)
        
        print(f"    Hypnogram Duration: {total_duration}s ({num_epochs_hypno} epochs)")
        print(f"    Expected Epochs (Parquet): {expected_epochs}")
        
        # Create empty array
        hypno_labels = np.full(num_epochs_hypno + 100, -1) # Buffer
        
        for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
            start_epoch = int(onset // 30)
            end_epoch = int((onset + duration) // 30)
            
            # Map description
            val = -1
            d = desc.lower()
            if 'wake' in d or 'w' in d: val = 0
            elif '1' in d: val = 1
            elif '2' in d: val = 2
            elif '3' in d: val = 3
            elif '4' in d: val = 4
            elif 'rem' in d or 'r' in d: val = 4
            elif '?' in d: val = -1
            else: val = -1
            
            # Fill
            # Check bounds
            if start_epoch < 0: start_epoch = 0
            
            # Sleep-EDF expanded: last annotation might be "Sleep stage ?" or similar
            # Just fill standard
            hypno_labels[start_epoch:end_epoch] = val
        
        print(f"    Unique labels in Hypnogram: {np.unique(hypno_labels)}")
            
        # Crop to expected size
            
        # Crop to expected size
        if expected_epochs <= len(hypno_labels):
            final_gt = hypno_labels[:expected_epochs]
        else:
            # Pad with -1
            padding = np.full(expected_epochs - len(hypno_labels), -1)
            final_gt = np.concatenate([hypno_labels, padding])
            
        return final_gt
        
    except Exception as e:
        print(f"  Error loading Hypnogram: {e}")
        return None

def run_single_model_inference(model, input_tensor, device):
    """Run inference for a single PyTorch model (Old Model)"""
    model.to(device)
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        logits = model(input_tensor) # (N, 5)
        probs = torch.softmax(logits, dim=1) # Models usually output logits
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confs = torch.max(probs, dim=1).values.cpu().numpy()
    return preds, confs

def run_ensemble_inference(script_model, input_tensor, device):
    """Run inference for Scripted Ensemble Model"""
    # Scripted model expects (N, 1, 76, 60) and normally handles internal batching or we batch here
    # It returns averaged probabilities
    script_model.to(device)
    all_preds = []
    all_confs = []
    
    BATCH_SIZE = 128
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        for i in range(0, len(input_tensor), BATCH_SIZE):
            batch = input_tensor[i:i+BATCH_SIZE]
            probs = script_model(batch) # Returns probabilities
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1).values.cpu().numpy()
            all_preds.extend(preds)
            all_confs.extend(confs)
            
    return np.array(all_preds), np.array(all_confs)

def generate_comparative_png(filename, preds_old, preds_new, gt=None, title="Comparison"):
    """
    Generates a PNG showing Old vs New predictions.
    If GT is present, it can be added too.
    """
    # Increased width significantly to avoid "compressed" look for long recordings (~2700 epochs)
    fig, ax = plt.subplots(figsize=(24, 6))
    
    x = range(len(preds_new))
    
    # Plot Old (Dashed, Lighter)
    ax.step(x, preds_old, where='post', label='Old Model (Baseline)', color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot New (Solid, Blue) - Thinner line for better detail in dense areas
    ax.step(x, preds_new, where='post', label='Ensemble (New)', color='#2196f3', linewidth=1.2, alpha=0.9)
    
    if gt is not None:
         ax.step(x, gt, where='post', label='Ground Truth', color='#4caf50', linestyle=':', linewidth=2, alpha=0.8)
    
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    ax.set_title(title)
    ax.set_xlabel("Epoch (30s)")
    ax.set_ylabel("Stage")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    out_name = os.path.basename(filename).replace(".parquet", "") + "_comparison.png"
    save_path = os.path.join(PNG_DIR, out_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

    plt.close(fig)
    return save_path

def generate_comparative_stats_png(filename, preds_old, preds_new, gt_labels, title="Stats Comparison"):
    """
    Generates a PNG mimicking the Dashboard 'Analysis of Results' for Three-Way Comparison:
    - Ground Truth (Original)
    - Old Model
    - Ensemble Model
    """
    # 1. Calculate Counts
    stages = [0, 1, 2, 3, 4]
    stage_names = ['Vigilia', 'N1', 'N2', 'N3', 'REM'] # Spanish
    
    counts_old = [np.sum(preds_old == i) for i in stages]
    counts_new = [np.sum(preds_new == i) for i in stages]
    
    # Handle GT (might be -1 for unknown, filter those?)
    # usually we compare distributions on valid epochs.
    # But for bar chart, simple counts of what's labeled 0-4 is fine.
    if gt_labels is not None:
        counts_gt = [np.sum(gt_labels == i) for i in stages]
    else:
        counts_gt = [0] * 5
        
    fig = plt.figure(figsize=(16, 12)) 
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    
    # --- Plot 1: Grouped Bar Chart (GT vs Old vs New) ---
    ax1 = fig.add_subplot(gs[0])
    
    x = np.arange(len(stages))
    width = 0.25 # thinner to fit 3
    
    # Colors
    c_gt = '#4CAF50' # Green
    c_old = '#9E9E9E' # Gray
    c_new = '#2196F3' # Blue
    
    rects1 = ax1.bar(x - width, counts_gt, width, label='Ground Truth', color=c_gt, edgecolor='black', alpha=0.8)
    rects2 = ax1.bar(x, counts_old, width, label='Old Model', color=c_old, edgecolor='black', alpha=0.8)
    rects3 = ax1.bar(x + width, counts_new, width, label='Ensemble', color=c_new, edgecolor='black', alpha=0.8)
    
    ax1.set_ylabel('Conteo')
    ax1.set_title('Distribución de Etapas: Original vs Old vs Ensemble', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_names)
    ax1.legend()
    
    ax1.bar_label(rects1, padding=3)
    ax1.bar_label(rects2, padding=3)
    ax1.bar_label(rects3, padding=3)
    
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Plot 2: Metrics Table ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    # Table Data
    table_data = []
    # Header
    table_data.append(["Etapa", "Ground Truth", "Old Model", "Ensemble", "Diff (Ens-GT)"])
    
    for i in range(len(stages)):
        diff = counts_new[i] - counts_gt[i]
        diff_str = f"+{diff}" if diff > 0 else f"{diff}"
        
        row = [
            stage_names[i],
            str(counts_gt[i]) if gt_labels is not None else "N/A",
            str(counts_old[i]),
            str(counts_new[i]),
            diff_str
        ]
        table_data.append(row)
        
    # Add Total
    if gt_labels is not None:
        # Count only valid 0-4
         total_gt = sum(counts_gt)
    else:
         total_gt = "N/A"
         
    table_data.append(["TOTAL", str(total_gt), str(sum(counts_old)), str(sum(counts_new)), "-"])

    # Draw Table
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2) 
    
    # Style Header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold')
        if row == len(table_data)-1: 
            cell.set_text_props(weight='bold')

    ax2.set_title("Tabla Comparativa de Conteo", pad=20)

    # Overall Title
    fig.suptitle(f"Reporte Comparativo Completo: {os.path.basename(filename)}", fontsize=16, y=0.98)

    out_name = os.path.basename(filename).replace(".parquet", "").replace(".bdf", "") + "_stats_comparison.png"
    save_path = os.path.join(PNG_DIR, out_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def main():
    print("--- Starting Comparative Analysis ---")
    
    # 1. Device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    
    # 2. Files
    files_to_process = []
    
    # A. BDF Conversion/Detection
    if os.path.exists(BDF_FILE):
        print(f"Checking BDF: {BDF_FILE}")
        
        # Check if actually Parquet
        is_fake_bdf = False
        try:
            with open(BDF_FILE, 'rb') as f:
                if f.read(4) == b'PAR1':
                    is_fake_bdf = True
        except:
            pass
            
        if is_fake_bdf:
             print("  File is actually a Parquet file (Mislabeled). Loading directly.")
             files_to_process.append(BDF_FILE)
        else:
            parquet_out = BDF_FILE.replace(".bdf", ".parquet")
            try:
                success, msg = convert_edf_to_parquet(BDF_FILE, parquet_out)
                if success:
                    print(f"  Conversion Success: {parquet_out}")
                    files_to_process.append(parquet_out)
                else:
                    print(f"  Conversion Failed: {msg}")
            except Exception as e:
                print(f"  Error converting BDF: {e}")
    else:
        print(f"Warning: BDF file not found at {BDF_FILE}")

    # B. SC Files
    sc_files_root = glob.glob(SC_PATTERN_ROOT)
    sc_files_subdir = glob.glob(os.path.join(BASE_DIR, "parquet_files/SC*.parquet"))
    # Combine both
    sc_files = sc_files_root + sc_files_subdir
    
    print(f"Found {len(sc_files)} SC files.")
    files_to_process.extend(sc_files)
    
    files_to_process = sorted(list(set(files_to_process))) # Unique
    
    if not files_to_process:
        print("No files to process. Exiting.")
        return

    # 3. Load Models
    print("Loading Old Model...")
    if os.path.exists(CKPT_OLD):
        arch = detect_architecture(CKPT_OLD)
        old_model = get_model(arch, num_classes=5)
        old_model, _ = load_checkpoint_weights(old_model, CKPT_OLD)
        old_model.eval()
    else:
        print(f"Error: Old checkpoint not found: {CKPT_OLD}")
        return

    print("Loading Ensemble Model...")
    if os.path.exists(ENSEMBLE_SCRIPT):
        ensemble_model = torch.jit.load(ENSEMBLE_SCRIPT)
        ensemble_model.eval()
    else:
         print(f"Error: Ensemble script not found: {ENSEMBLE_SCRIPT}")
         return

    # 4. Processing Loop
    sql_buffer = []
    
    for fpath in files_to_process:
        fname = os.path.basename(fpath)
        
        csv_name = f"comparison_results_{fname.replace('.parquet', '').replace('.bdf', '')}.csv"
        csv_path = os.path.join(BASE_DIR, csv_name)
        
        if os.path.exists(csv_path):
             print(f"Skipping {fname} (Already Processed: {csv_name})")
             continue
             
        print(f"\nProcessing {fname}...")
        
        try:
            # Handle reading fake bdf with pandas read_parquet
            df = pd.read_parquet(fpath)
            
            # Extract features
            cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage', 'true_label'] if c in df.columns]
            if cols_to_drop:
                data = df.drop(columns=cols_to_drop).values
            else:
                data = df.values
                
            # Preprocess to Tensor (N, 1, 76, 60)
            # preprocess_spectrogram handles one item or we can map it
            # Vectorized helper inside? 
            # inference.preprocess_spectrogram takes 1D array.
            # We need batch preprocessing logic like in batch_ensemble.py
            
            # Quick batch preprocess:
            data = data.astype(np.float32)
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            normalized = (data - mean) / (std + 1e-6)
            input_tensor = torch.from_numpy(normalized.reshape(-1, 1, 76, 60))
            
            # --- Inference ---
            # Old
            preds_old, confs_old = run_single_model_inference(old_model, input_tensor, device)
            
            # New
            preds_new, confs_new = run_ensemble_inference(ensemble_model, input_tensor, device)
            
            # --- Ground Truth ---
            gt = None
            # Check cols
            for c in ['label', 'true_label', 'stage']:
                if c in df.columns:
                    # Check if valid
                    temp_gt = df[c].values
                    # If strictly ALL -1, then it's invalid.
                    if np.any(temp_gt != -1):
                        gt = temp_gt
                        break
            
            # Fallback: Load External Hypnogram
            if gt is None:
                print(f"  Internal Ground Truth missing or invalid. Attempting to load external Hypnogram...")
                gt = load_external_hypnogram(fname, expected_epochs=len(preds_new))
                if gt is not None:
                    print(f"  External Hypnogram loaded successfully (Length: {len(gt)})")
                else:
                    print(f"  Comparison will be without Ground Truth.")
            
            # --- Outputs ---
            
            # --- Outputs ---
            
            # 1. CSV
            # Add columns to DF (or new DF)
            df_out = pd.DataFrame({
                'epoch': range(len(preds_new)),
                'pred_old': preds_old,
                'conf_old': confs_old,
                'pred_ensemble': preds_new,
                'conf_ensemble': confs_new
            })
            if gt is not None:
                df_out['true_label'] = gt
            
            csv_name = f"comparison_results_{fname.replace('.parquet', '')}.csv"
            df_out.to_csv(os.path.join(BASE_DIR, csv_name), index=False)
            print(f"  Saved CSV: {csv_name}")
            
            # 2. SQL
            # We only generate SQL for the NEW (Ensemble) model as that's the desired "production" output usually
            # Or both? "The results should be stored in the .cvs and .sql files."
            # Usually SQL stores the active prediction. I'll store Ensemble.
            
            # Header only if file empty? No, we append.
            header = "INSERT INTO sleep_predictions (patient_id, filename, epoch_index, predicted_stage, confidence, model_used) VALUES"
            sql_rows = []
            for i, (p, c) in enumerate(zip(preds_new, confs_new)):
                lbl = STAGE_MAP.get(p, "Unknown")
                # Clean patient ID from filename?
                pid = fname.split('.')[0]
                row = f"('{pid}', '{fname}', {i}, '{lbl}', {c:.4f}, 'Ensemble')"
                sql_rows.append(row)
            
            if sql_rows:
                 # Append to buffer or write chunk
                 sql_buffer.append(f"-- {fname}\n" + header + "\n" + ",\n".join(sql_rows) + ";\n")
            
            # 3. PNG (Hypnogram)
            png_path = generate_comparative_png(fname, preds_old, preds_new, gt, title=f"Comparison: {fname}")
            print(f"  Saved Graph: {png_path}")
            
            # 4. PNG (Stats)
            stats_path = generate_comparative_stats_png(fname, preds_old, preds_new, gt, title=f"Stats: {fname}")
            print(f"  Saved Stats: {stats_path}")

            
        except Exception as e:
            print(f"  Failed to process {fname}: {e}")
            import traceback
            traceback.print_exc()

    # Write SQL
    if sql_buffer:
        print(f"Writing SQL to {SQL_FILE}...")
        with open(SQL_FILE, "a") as f:
            for block in sql_buffer:
                f.write(block)
        print("SQL Write Complete.")

if __name__ == "__main__":
    main()
