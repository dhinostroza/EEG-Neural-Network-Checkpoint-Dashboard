import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.jit
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
from datetime import datetime

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
SC_PATTERN_ROOT = os.path.join(BASE_DIR, "SC*.parquet")
SHHS_PATTERN_ROOT = os.path.join(BASE_DIR, "shhs1*.parquet")

# Models
CKPT_OLD = os.path.join(BASE_DIR, "checkpoint_files/2000 files/2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt")
ENSEMBLE_SCRIPT = os.path.join(BASE_DIR, "checkpoint_files/2000 files/2026-01-11_convnext_base_ensemble.pt")

SQL_FILE = os.path.join(BASE_DIR, "comparative_predictions.sql")

STAGE_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Ground Truth Paths
HYPNO_PATH_CASSETTE = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/Bases_datos/sleep-edf-database-expanded-1.0.0/sleep-cassette"
HYPNO_PATH_TELEMETRY = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/Bases_datos/sleep-edf-database-expanded-1.0.0/sleep-telemetry"
HYPNO_PATH_SHHS = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-profusion/shhs1"

# Load Helper
def load_external_hypnogram(filename, expected_epochs):
    if not HAS_EDF_LIBS:
        return None
    
    base = os.path.basename(filename).replace(".parquet", "").replace("_processed", "")
    
    # Logic for Sleep-EDF
    if base.startswith("SC") or base.startswith("ST"):
        subject_id = base[:6]
        pattern = f"{subject_id}*Hypnogram.edf"
        candidates = glob.glob(os.path.join(HYPNO_PATH_CASSETTE, pattern)) + \
                     glob.glob(os.path.join(HYPNO_PATH_TELEMETRY, pattern))
    # Logic for SHHS
    elif base.startswith("shhs"):
         parts = base.split('-')
         if len(parts) >= 2:
             xml_name = f"{parts[0]}-{parts[1]}-profusion.xml"
             # Use XML path defined or just use the local file logic if XML parser available
             # But here we are looking for EDF hypnograms? SHHS usually provides XML.
             # The user code had XML parser. Let's stick to simple EDF if possible or return None.
             # SHHS usually doesn't have Hypnogram.edf, it has XML.
             # We will support XML if needed but for now return None for SHHS here unless we merge the XML logic.
             return None
    else:
        return None

    if not candidates:
        return None
        
    hypno_path = candidates[0]
    try:
        annot = mne.read_annotations(hypno_path)
        total_duration = annot.onset[-1] + annot.duration[-1]
        num_epochs_hypno = int(total_duration // 30)
        
        hypno_labels = np.full(num_epochs_hypno + 200, -1)
        
        for onset, duration, desc in zip(annot.onset, annot.duration, annot.description):
            start_epoch = int(onset // 30)
            end_epoch = int((onset + duration) // 30)
            val = -1
            d = desc.lower()
            if 'wake' in d or 'w' in d: val = 0
            elif '1' in d: val = 1
            elif '2' in d: val = 2
            elif '3' in d: val = 3
            elif '4' in d: val = 3 # Map 4 to 3
            elif 'rem' in d or 'r' in d: val = 4
            elif '?' in d: val = -1
            
            if start_epoch < len(hypno_labels):
                 end_real = min(end_epoch, len(hypno_labels))
                 hypno_labels[start_epoch:end_real] = val
                 
        if expected_epochs <= len(hypno_labels):
            return hypno_labels[:expected_epochs]
        else:
            padding = np.full(expected_epochs - len(hypno_labels), -1)
            padding_len = len(padding)
            # If padding is huge, something is wrong, but we'll accept it
            return np.concatenate([hypno_labels[:expected_epochs], padding])
    except:
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
    script_model.to(device)
    all_preds = []
    all_confs = []
    BATCH_SIZE = 64
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        for i in range(0, len(input_tensor), BATCH_SIZE):
            batch = input_tensor[i:i+BATCH_SIZE]
            probs = script_model(batch)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1).values.cpu().numpy()
            all_preds.extend(preds)
            all_confs.extend(confs)
    return np.array(all_preds), np.array(all_confs)

def generate_composite_report(filename, preds, preds_old, gt, accuracy, acc_old, lang='es'):
    """
    Generates a detailed composite PNG with 5 components:
    1. Hypnogram (Top)
    2. Metrics & Accuracy (Middle Left)
    3. Comparative Table (Middle Right)
    4. Confusion Matrix (Bottom Left)
    5. Class Distribution (Bottom Right)
    """
    # Config
    diff = accuracy - acc_old
    sign = "+" if diff >= 0 else ""
    
    if lang == 'es':
        title_main = f"Comparación Global de Precisión: Modelo Base ({acc_old:.2f}%) vs Ensemble ({accuracy:.2f}%)\nMejora: {sign}{diff:.2f}%"
        labels = ['Vigilia', 'N1', 'N2', 'N3', 'REM']
        label_gt = "Ground Truth"
        label_pred = "Ensemble (Nuevo)"
        label_old = "ConvNeXT Base"
        head_table = ["Etapa", "GT", "ConvNeXT", "Ensemble", "Diff (Ens-GT)"]
        title_cm = "Matriz de Confusión (Ensemble vs GT)"
        title_dist = "Distribución de Clases"
        axis_predicted = "Predicción Ensemble"
        axis_real = "Etapa Real"
        lbl_acc = "EXACTITUD GENERAL"
        lbl_kappa = "INDICE KAPPA"
        footer_text = f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Archivo: {filename}"
    else:
        title_main = f"Global Accuracy Comparison: Base Model ({acc_old:.2f}%) vs Ensemble ({accuracy:.2f}%)\nImprovement: {sign}{diff:.2f}%"
        labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
        label_gt = "Ground Truth"
        label_pred = "Ensemble (New)"
        label_old = "ConvNeXT Base"
        head_table = ["Stage", "GT", "ConvNeXT", "Ensemble", "Diff (Ens-GT)"]
        title_cm = "Confusion Matrix (Ensemble vs GT)"
        title_dist = "Class Distribution"
        axis_predicted = "Ensemble Prediction"
        axis_real = "True Stage"
        lbl_acc = "OVERALL ACCURACY"
        lbl_kappa = "KAPPA INDEX"
        footer_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | File: {filename}"

    # Setup Figure (Tall format)
    fig = plt.figure(figsize=(20, 24))
    # Grid: 
    # Row 0: Hypnogram (Height 2)
    # Row 1: Table & Metrics (Height 1)
    # Row 2: Charts (Height 1.5)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.8, 1.2], hspace=0.3)

    # ==========================
    # 1. Hypnogram (Top, Full Width)
    # ==========================
    ax_hypno = fig.add_subplot(gs[0, :])
    x = range(len(preds))
    
    # Old (Gray)
    ax_hypno.step(x, preds_old, where='post', label=label_old, color='#BDBDBD', linestyle='--', linewidth=1.5, alpha=0.7)
    # New (Blue)
    ax_hypno.step(x, preds, where='post', label=label_pred, color='#1976D2', linewidth=1.5, alpha=0.9)
    # GT (Green)
    if gt is not None:
         valid_mask = gt != -1
         gt_clean = gt.astype(float)
         gt_clean[~valid_mask] = np.nan
         ax_hypno.step(x, gt_clean, where='post', label=label_gt, color='#388E3C', linestyle=':', linewidth=2.5, alpha=0.9)

    ax_hypno.set_yticks([0, 1, 2, 3, 4])
    ax_hypno.set_yticklabels(labels)
    ax_hypno.set_title(title_main, fontsize=18, weight='bold', pad=20)
    ax_hypno.grid(True, alpha=0.3)
    ax_hypno.invert_yaxis()
    ax_hypno.legend(loc='lower left', ncol=3)
    ax_hypno.set_xlim(0, len(preds))
    ax_hypno.set_xlabel("Epoch (30s)" if lang=='en' else "Época (30s)")

    # ==========================
    # 2. Metrics Box (Middle Left)
    # ==========================
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_metrics.axis('off')
    
    # Calculate Kappa if GT exists
    kappa_val = 0.0
    if gt is not None:
        mask = gt != -1
        if mask.any():
            from sklearn.metrics import cohen_kappa_score
            kappa_val = cohen_kappa_score(gt[mask], preds[mask])

    # Draw Text
    ax_metrics.text(0.5, 0.7, f"{accuracy:.2f}%", ha='center', va='center', fontsize=60, weight='bold', color='#1976D2')
    ax_metrics.text(0.5, 0.55, lbl_acc, ha='center', va='center', fontsize=14, color='gray')
    
    if gt is not None:
        ax_metrics.text(0.5, 0.35, f"κ = {kappa_val:.3f}", ha='center', va='center', fontsize=30, weight='bold', color='#555')
        ax_metrics.text(0.5, 0.25, lbl_kappa, ha='center', va='center', fontsize=12, color='gray')
    else:
        ax_metrics.text(0.5, 0.3, "Ground Truth Missing", ha='center', va='center', fontsize=12, color='red')

    # ==========================
    # 3. Comparative Table (Middle Right)
    # ==========================
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis('off')
    
    counts_pred = [np.sum(preds == i) for i in range(5)]
    counts_old = [np.sum(preds_old == i) for i in range(5)]
    data_table = []
    
    total_gt = 0
    total_old = sum(counts_old)
    total_new = sum(counts_pred)
    
    # Header
    data_table.append(head_table)
    
    for i in range(5):
        c_gt = np.sum(gt == i) if gt is not None else 0
        c_old = counts_old[i]
        c_new = counts_pred[i]
        
        diff = c_new - c_gt
        diff_str = f"+{diff}" if diff > 0 else f"{diff}"
        if gt is None: diff_str = "-"
        
        row = [labels[i], c_gt if gt is not None else "-", c_old, c_new, diff_str]
        data_table.append(row)
        total_gt += c_gt

    # Footer Row
    data_table.append(["TOTAL", total_gt if gt is not None else "-", total_old, total_new, "-"])

    # Draw Table
    table = ax_table.table(cellText=data_table, loc='center', cellLoc='center', colWidths=[0.2]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    # Style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold')
        if row == len(data_table)-1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#F5F5F5')
            
    ax_table.set_title("Tabla Comparativa de Conteo" if lang == 'es' else "Comparative Count Table", 
                       fontsize=14, pad=10, weight='bold', color='#444')

    # ==========================
    # 4. Confusion Matrix (Bottom Left)
    # ==========================
    ax_cm = fig.add_subplot(gs[2, 0])
    
    if gt is not None:
        mask = gt != -1
        y_true = gt[mask]
        y_pred = preds[mask]
        if len(y_true) > 0:
            from sklearn.metrics import confusion_matrix
            cm_val = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
            sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 12})
            ax_cm.set_title(title_cm, fontsize=14, weight='bold', color='#444')
            ax_cm.set_ylabel(axis_real)
            ax_cm.set_xlabel(axis_predicted)
    else:
        ax_cm.text(0.5, 0.5, "No GT", ha='center')

    # ==========================
    # 5. Distribution Bar Chart (Bottom Right)
    # ==========================
    ax_dist = fig.add_subplot(gs[2, 1])
    
    x_pos = np.arange(len(labels))
    width = 0.25
    
    counts_gt_plot = [np.sum(gt == i) for i in range(5)] if gt is not None else [0]*5
    
    ax_dist.bar(x_pos - width, counts_gt_plot, width, label='GT', color='#388E3C', alpha=0.7)
    ax_dist.bar(x_pos, counts_old, width, label='ConvNeXT', color='#BDBDBD', alpha=0.7)
    ax_dist.bar(x_pos + width, counts_pred, width, label='Ensemble', color='#1976D2', alpha=0.8)
    
    ax_dist.set_xticks(x_pos)
    ax_dist.set_xticklabels(['W','N1','N2','N3','R'])
    ax_dist.legend()
    ax_dist.set_title(title_dist, fontsize=14, weight='bold', color='#444')
    ax_dist.grid(axis='y', linestyle='--', alpha=0.3)

    # Footer
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

def main():
    print("--- Starting Optimized Analysis ---")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    
    # 1. Identify Files (Limit 2 SC, 2 SHHS)
    all_sc = sorted(glob.glob(SC_PATTERN_ROOT)) + sorted(glob.glob(os.path.join(BASE_DIR, "parquet_files/SC*.parquet")))
    all_shhs = sorted(glob.glob(SHHS_PATTERN_ROOT)) + sorted(glob.glob(os.path.join(BASE_DIR, "parquet_files/shhs1*.parquet")))
    
    # Deduplicate
    all_sc = sorted(list(set(all_sc)))
    all_shhs = sorted(list(set(all_shhs)))
    
    # Process ALL SC files and ALL SHHS
    files_to_process = all_sc + all_shhs
    print(f"Selected {len(files_to_process)} files for processing: {[os.path.basename(f) for f in files_to_process]}")
    
    # 2. Load Models
    # Old
    print("Loading Old Model...")
    if os.path.exists(CKPT_OLD):
        arch = detect_architecture(CKPT_OLD)
        old_model = get_model(arch, num_classes=5)
        old_model, _ = load_checkpoint_weights(old_model, CKPT_OLD)
        old_model.eval()
    else:
        print(f"Error: Old checkpoint not found: {CKPT_OLD}")
        return

    # Ensemble
    print("Loading Ensemble Model...")
    if os.path.exists(ENSEMBLE_SCRIPT):
        model = torch.jit.load(ENSEMBLE_SCRIPT)
        model.eval()
    else:
        print("Ensemble model not found.")
        return

    # 3. Process
    for fpath in files_to_process:
        fname = os.path.basename(fpath)
        done_file = fpath + ".done"
        
        # Check if done
        if os.path.exists(done_file):
            print(f"Skipping {fname} (Found .done file)")
            continue
            
        print(f"\nProcessing {fname}...")
        try:
            df = pd.read_parquet(fpath)
            
            # Prepare Data
            cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage', 'true_label'] if c in df.columns]
            if cols_to_drop:
                data = df.drop(columns=cols_to_drop).values
            else:
                data = df.values
                
            data = data.astype(np.float32)
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            normalized = (data - mean) / (std + 1e-6)
            input_tensor = torch.from_numpy(normalized.reshape(-1, 1, 76, 60))
            
            # Inference Old
            preds_old, confs_old = run_single_model_inference(old_model, input_tensor, device)
            
            # Inference Ensemble
            preds, confs = run_ensemble_inference(model, input_tensor, device)
            
            # Ground Truth
            gt = None
            for c in ['label', 'true_label', 'stage']:
                if c in df.columns:
                    temp = df[c].values
                    if np.any(temp != -1):
                        gt = temp
                        break
            
            if gt is None:
                gt = load_external_hypnogram(fname, len(preds))
                
            # Calc Accuracy for Title
            acc = 0.0
            acc_old = 0.0
            if gt is not None:
                mask = gt != -1
                if mask.sum() > 0:
                    acc = (preds[mask] == gt[mask]).mean() * 100
                    acc_old = (preds_old[mask] == gt[mask]).mean() * 100
            
            # Generate Reports
            current_date = datetime.now().strftime("%Y-%m-%d")
            base_clean = fname.replace(".parquet", "").replace("_processed", "")
            
            # Spanish Report
            fig_es = generate_composite_report(fname, preds, preds_old, gt, accuracy=acc, acc_old=acc_old, lang='es')
            path_es = os.path.join(PNG_DIR, f"{current_date}_{base_clean}_es.png")
            fig_es.savefig(path_es, dpi=100)
            plt.close(fig_es)
            print(f"  Saved: {os.path.basename(path_es)}")
            
            # English Report
            fig_en = generate_composite_report(fname, preds, preds_old, gt, accuracy=acc, acc_old=acc_old, lang='en')
            path_en = os.path.join(PNG_DIR, f"{current_date}_{base_clean}_en.png")
            fig_en.savefig(path_en, dpi=100)
            plt.close(fig_en)
            print(f"  Saved: {os.path.basename(path_en)}")
            
            # Mark Done (Create Empty File)
            with open(done_file, 'w') as f:
                f.write(f"Processed on {datetime.now()}")
            print(f"  Marked as done: {os.path.basename(done_file)}")
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
