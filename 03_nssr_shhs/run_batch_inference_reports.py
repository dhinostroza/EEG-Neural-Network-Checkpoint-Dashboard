
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import xml.etree.ElementTree as ET
import timm

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Paths
SOURCE_DIR = "parquet_files"
CHECKPOINT_DIR = "checkpoint_files/2000 files"
BASELINE_CKPT = os.path.join(CHECKPOINT_DIR, "2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt")
ENSEMBLE_MODEL = os.path.join(CHECKPOINT_DIR, "2026-01-11_convnext_base_ensemble.pt")
SQL_PATH = "predictions.sql"

BATCH_LIMIT = 10000
DATE_STAMP = "2026-01-11"

# Map string labels to int
STAGE_MAP_REV = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# TEXT DICTIONARIES
TEXT_ES = {
    "title_global": "Comparación de Precisión Global: Base ({:.2f}%) vs Ensamble ({:.2f}%)",
    "improvement": "Mejora: +{:.2f}%",
    "decline": "Disminución: {:.2f}%",
    "cm_title": "Matriz de Confusión ({})",
    "dist_title": "Distribución de Clases ({})",
    "hyp_title": "Hipnograma Predicho ({})",
    "gt_label": "Ground Truth",
    "baseline_label": "Base",
    "ensemble_label": "Ensamble",
    "y_label": "Clase Real",
    "x_label_base": "Predicción (Base)",
    "x_label_ens": "Predicción (Ensamble)",
    "epoch_label": "Época (30s)",
    "stages": ["Vigilia", "N1", "N2", "N3", "REM"],
    "stages_short": ["W", "N1", "N2", "N3", "R"],
    "no_baseline": "Sin Datos Base"
}

TEXT_EN = {
    "title_global": "Global Accuracy Comparison: Baseline ({:.2f}%) vs Ensemble ({:.2f}%)",
    "improvement": "Improvement: +{:.2f}%",
    "decline": "Decline: {:.2f}%",
    "cm_title": "Confusion Matrix ({})",
    "dist_title": "Class Distribution ({})",
    "hyp_title": "Predicted Hypnogram ({})",
    "gt_label": "Ground Truth",
    "baseline_label": "Baseline",
    "ensemble_label": "Ensemble",
    "y_label": "True Class",
    "x_label_base": "Predicted (Baseline)",
    "x_label_ens": "Predicted (Ensemble)",
    "epoch_label": "Epoch (30s)",
    "stages": ["Wake", "N1", "N2", "N3", "REM"],
    "stages_short": ["W", "N1", "N2", "N3", "R"],
    "no_baseline": "No Baseline Data Found"
}

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_model(model_name='convnext_base', num_classes=5, pretrained=False):
    """
    Creates a ConvNeXT v2 Base model adapted for single-channel input (spectrograms).
    """
    if model_name == 'convnext_base':
        # We use pretrained=False because we will load our own weights
        model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained)

        # Adapt first layer for 1 channel
        original_conv = model.stem[0]
        new_first_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )
        model.stem[0] = new_first_conv

        # Adapt head
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, num_classes)
        return model
    else:
        raise ValueError(f"Model {model_name} not supported.")

def load_checkpoint_weights(model, checkpoint_path, device):
    """
    Loads weights from a PyTorch Lightning checkpoint into a vanilla PyTorch model.
    """
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # State dict handling
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Remove 'model.' prefix and exclude 'loss_fn' params
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("loss_fn."):
            continue
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

# =============================================================================
# SQL HELPERS
# =============================================================================

def check_if_in_sql(filename):
    """
    Checks if the filename (or core ID) is already present in predictions.sql.
    Pure text search for speed/simplicity as per existing pattern.
    """
    if not os.path.exists(SQL_PATH):
        return False
        
    core_id = filename.replace(".parquet", "")
    # Robust check: look for exact filename match in VALUES
    # e.g. 'shhs1-200207.parquet'
    
    try:
        # Using grep might be faster for huge files, but python is safer for logic
        # For 500MB+ sql files, reading line by line is better.
        # But here we assume it's manageable.
        # Let's try a quick grep via shell if possible, or just read line by line.
        # We'll stick to python line-by-line with early exit.
        with open(SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if filename in line:
                    return True
        return False
    except FileNotFoundError:
        return False

def save_to_sql(filename, predictions, confidences, model_ckpt_name, patient_id):
    """
    Appends predictions to predictions.sql in the format:
    ('patient_id', 'filename', epoch_index, 'predicted_stage', confidence, 'model_used', NULL),
    """
    print(f"Archiving {len(predictions)} predictions to SQL for {filename}...")
    
    entries = []
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        stage_str = STAGE_MAP_REV.get(pred, "Unknown")
        # Format: ('SC4072', 'SC4072E.parquet', 0, 'Wake', 0.9997, 'ckpt_name', NULL)
        entry = f"('{patient_id}', '{filename}', {i}, '{stage_str}', {conf:.4f}, '{model_ckpt_name}', NULL)"
        entries.append(entry)
        
    if not entries:
        return

    # Batch insert statement
    # We'll split huge batches if necessary, but 1000 epochs is fine for one statement?
    # Actually, let's just write them cleanly.
    
    with open(SQL_PATH, 'a') as f:
        f.write(f"\n-- Auto-generated predictions for {filename}\n")
        f.write("INSERT INTO sleep_predictions (patient_id, filename, epoch_index, predicted_stage, confidence, model_used, true_stage) VALUES\n")
        f.write(",\n".join(entries) + ";\n")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_gt_from_xml(xml_path):
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        stages_nodes = root.findall(".//{*}SleepStage")
        if not stages_nodes:
            stages_nodes = root.findall(".//SleepStage")
            
        stages = []
        for stage in stages_nodes:
            try:
                val = int(stage.text)
                if val == 0: val = 0 
                elif val == 1: val = 1 
                elif val == 2: val = 2 
                elif val == 3: val = 3 
                elif val == 4: val = 3 
                elif val == 5: val = 4 
                else: val = -1 
                stages.append(val)
            except:
                stages.append(-1)
        return stages
    except Exception as e:
        print(f"XML Error: {e}")
        return []

def generate_comparative_report(output_base, gt_labels, predictions, prev_predictions, prev_model_name, acc_ensemble, acc_prev, acc_prev_str, lang='ES'):
    T = TEXT_ES if lang == 'ES' else TEXT_EN
    
    # --- METRICS & ALIGNMENT ---
    L = min(len(gt_labels), len(predictions))
    if len(prev_predictions) > 0:
        L = min(L, len(prev_predictions))
        
    y_t_acc = gt_labels[:L]
    y_pr_acc = prev_predictions[:L] if len(prev_predictions) >= L else []
    y_p_acc = predictions[:L]
    
    # Filter invalid GT for accuracy calculation
    valid_indices = [i for i, x in enumerate(y_t_acc) if x != -1]
    
    y_t_valid = [y_t_acc[i] for i in valid_indices]
    y_pr_valid = [y_pr_acc[i] for i in valid_indices] if y_pr_acc else []
    y_p_valid = [y_p_acc[i] for i in valid_indices]
    
    final_acc_ens = accuracy_score(y_t_valid, y_p_valid) if y_p_valid else 0.0
    final_acc_prev = accuracy_score(y_t_valid, y_pr_valid) if y_pr_valid else 0.0
    
    diff = final_acc_ens - final_acc_prev
    diff_text = T["improvement"].format(diff * 100) if diff >= 0 else T["decline"].format(abs(diff) * 100)
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(16, 16))
    
    # Layout:
    # Row 0: Title (Global)
    # Row 1: Baseline (CM + Dist)
    # Row 2: Baseline Hypnogram
    # Row 3: Ensemble (CM + Dist)
    # Row 4: Ensemble Hypnogram
    
    gs = gridspec.GridSpec(5, 2, height_ratios=[0.1, 1, 0.8, 1, 0.8])
    
    # ROW 0: TITLE
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_str = T["title_global"].format(final_acc_prev*100, final_acc_ens*100) + "\n" + diff_text
    ax_title.text(0.5, 0.5, title_str, ha='center', va='center', fontsize=16, fontweight='bold')

    # Shared Config
    x = np.arange(len(T["stages_short"]))
    w = 0.35
    t_counts = [y_t_valid.count(l) for l in [0,1,2,3,4]]
    time_axis = np.arange(len(y_p_acc))
    y_gt_mapped = [x if x != -1 else None for x in y_t_acc]

    # ROW 1: BASELINE METRICS
    ax_prev_cm = fig.add_subplot(gs[1, 0])
    ax_prev_dist = fig.add_subplot(gs[1, 1])
    
    if y_pr_valid:
        # CM
        cm_prev = confusion_matrix(y_t_valid, y_pr_valid, labels=[0,1,2,3,4])
        cax_p = ax_prev_cm.matshow(cm_prev, cmap='Oranges') 
        fig.colorbar(cax_p, ax=ax_prev_cm, fraction=0.046, pad=0.04)
        ax_prev_cm.set_title(T["cm_title"].format(prev_model_name), fontweight='bold', color='#666666')
        ax_prev_cm.set_xticks(np.arange(5))
        ax_prev_cm.set_yticks(np.arange(5))
        ax_prev_cm.set_xticklabels(T["stages"])
        ax_prev_cm.set_yticklabels(T["stages"])
        ax_prev_cm.set_ylabel(T["y_label"])
        ax_prev_cm.set_xlabel(T["x_label_base"])
        
        thresh_p = cm_prev.max() / 2.
        for i in range(cm_prev.shape[0]):
            for j in range(cm_prev.shape[1]):
                ax_prev_cm.text(j, i, str(cm_prev[i, j]), va='center', ha='center',
                                color="white" if cm_prev[i, j] > thresh_p else "black")
        
        # Dist
        p_counts = [y_pr_valid.count(l) for l in [0,1,2,3,4]]
        ax_prev_dist.bar(x - w/2, t_counts, w, label=T["gt_label"], color='#4caf50')
        ax_prev_dist.bar(x + w/2, p_counts, w, label=T["baseline_label"], color='#ff9800') 
        ax_prev_dist.set_title(T["dist_title"].format(prev_model_name), fontweight='bold', color='#666666')
        ax_prev_dist.set_xticks(x)
        ax_prev_dist.set_xticklabels(T["stages_short"])
        ax_prev_dist.legend()
    else:
        ax_prev_cm.text(0.5, 0.5, T["no_baseline"], ha='center', va='center')
        ax_prev_cm.axis('off')
        ax_prev_dist.axis('off')

    # ROW 2: BASELINE HYPNOGRAM
    ax_prev_hyp = fig.add_subplot(gs[2, :])
    if y_pr_valid:
        y_prev_mapped = [x if x != -1 else None for x in y_p_acc] # Wait, y_p_acc is predictions (Ensemble)? No, need y_pr_acc equivalent
        # Re-map baseline full array
        y_pr_full = prev_predictions[:L] if len(prev_predictions) >= L else []
        y_pr_mapped = [x if x != -1 else None for x in y_pr_full]
        
        # Use simple mapping if len mismatch isn't an issue (L is min len)
        # Re-slice to L to be safe
        y_pr_mapped = y_pr_mapped[:len(time_axis)] # Should match time_axis which is len(y_p_acc) which is L
        
        ax_prev_hyp.step(time_axis, y_pr_mapped, where='post', label=T["baseline_label"], color='#ff9800', lw=1.5, alpha=0.9)
        ax_prev_hyp.step(time_axis, y_gt_mapped, where='post', label=T["gt_label"], color='#4caf50', linestyle='--', lw=1.5, alpha=0.7)
        
        prev_mismatches = [i for i, (p, t) in enumerate(zip(y_pr_mapped, y_gt_mapped)) 
                           if p is not None and t is not None and p != t]
        if len(prev_mismatches) > 0:
             ax_prev_hyp.scatter(prev_mismatches, [y_pr_mapped[i] for i in prev_mismatches], 
                                color='red', s=10, marker='x', alpha=0.5, zorder=5)

        ax_prev_hyp.set_yticks(np.arange(5))
        ax_prev_hyp.invert_yaxis()
        ax_prev_hyp.set_yticklabels(T["stages_short"]) 
        
        ax_prev_hyp.set_title(T["hyp_title"].format(f"{final_acc_prev*100:.2f}%"), fontweight='bold', color='#ff9800')
        ax_prev_hyp.set_xlabel(T["epoch_label"])
        ax_prev_hyp.legend(loc='upper right')
        ax_prev_hyp.grid(True, alpha=0.3)
    else:
        ax_prev_hyp.text(0.5, 0.5, T["no_baseline"], ha='center', va='center')
        ax_prev_hyp.axis('off')

    # ROW 3: ENSEMBLE METRICS
    ax_ens_cm = fig.add_subplot(gs[3, 0])
    ax_ens_dist = fig.add_subplot(gs[3, 1])
    
    cm_ens = confusion_matrix(y_t_valid, y_p_valid, labels=[0,1,2,3,4])
    cax_e = ax_ens_cm.matshow(cm_ens, cmap='Blues')
    fig.colorbar(cax_e, ax=ax_ens_cm, fraction=0.046, pad=0.04)
    ax_ens_cm.set_title(T["cm_title"].format("Ensemble"), fontweight='bold', color='#2196f3')
    ax_ens_cm.set_xticks(np.arange(5))
    ax_ens_cm.set_yticks(np.arange(5))
    ax_ens_cm.set_xticklabels(T["stages"])
    ax_ens_cm.set_yticklabels(T["stages"])
    ax_ens_cm.set_ylabel(T["y_label"])
    ax_ens_cm.set_xlabel(T["x_label_ens"])
    
    thresh_e = cm_ens.max() / 2.
    for i in range(cm_ens.shape[0]):
        for j in range(cm_ens.shape[1]):
            ax_ens_cm.text(j, i, str(cm_ens[i, j]), va='center', ha='center',
                           color="white" if cm_ens[i, j] > thresh_e else "black")
                           
    ens_counts = [y_p_valid.count(l) for l in [0,1,2,3,4]]
    ax_ens_dist.bar(x - w/2, t_counts, w, label=T["gt_label"], color='#4caf50')
    ax_ens_dist.bar(x + w/2, ens_counts, w, label=T["ensemble_label"], color='#2196f3')
    ax_ens_dist.set_title(T["dist_title"].format("Ensemble"), fontweight='bold', color='#2196f3')
    ax_ens_dist.set_xticks(x)
    ax_ens_dist.set_xticklabels(T["stages_short"])
    ax_ens_dist.legend()

    # ROW 4: ENSEMBLE HYPNOGRAM
    ax_ens_hyp = fig.add_subplot(gs[4, :])
    
    y_ens_mapped = [x if x != -1 else None for x in y_p_acc]
    
    ax_ens_hyp.step(time_axis, y_ens_mapped, where='post', label=T["ensemble_label"], color='#2196f3', lw=1.5, alpha=0.9)
    ax_ens_hyp.step(time_axis, y_gt_mapped, where='post', label=T["gt_label"], color='#4caf50', linestyle='--', lw=1.5, alpha=0.7)
    
    mismatches = [i for i, (p, t) in enumerate(zip(y_ens_mapped, y_gt_mapped)) 
                  if p is not None and t is not None and p != t]
    if len(mismatches) > 0:
         ax_ens_hyp.scatter(mismatches, [y_ens_mapped[i] for i in mismatches], 
                            color='red', s=10, marker='x', alpha=0.5, zorder=5)

    ax_ens_hyp.set_yticks(np.arange(5))
    ax_ens_hyp.invert_yaxis()
    ax_ens_hyp.set_yticklabels(T["stages_short"]) 
    
    ax_ens_hyp.set_title(T["hyp_title"].format(f"{final_acc_ens*100:.2f}%"), fontweight='bold', color='#2196f3')
    ax_ens_hyp.set_xlabel(T["epoch_label"])
    ax_ens_hyp.legend(loc='upper right')
    ax_ens_hyp.grid(True, alpha=0.3)
    
    plt.tight_layout()
    lang_code = lang.lower()
    final_path = f"{output_base}_{lang_code}.png"
    plt.savefig(final_path, dpi=150)
    plt.close(fig)
    
    print(f"Saved {lang} REPORT to {final_path}")

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def main():
    # Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    # Load Baseline Model
    print("Initializing Baseline Model...")
    baseline_model = get_model('convnext_base', num_classes=5, pretrained=False)
    load_checkpoint_weights(baseline_model, BASELINE_CKPT, device)
    print("Baseline Model Loaded.")
    
    # Load Ensemble Model
    print("Loading Ensemble Model...")
    ensemble_model = torch.jit.load(ENSEMBLE_MODEL)
    ensemble_model.to(device)
    ensemble_model.eval()
    print("Ensemble Model Loaded.")
    
    all_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "shhs1-*.parquet")))
    processed_count = 0
    os.makedirs("png", exist_ok=True)
    
    baseline_ckpt_name = os.path.basename(BASELINE_CKPT)
    
    for fpath in all_files:
        if processed_count >= BATCH_LIMIT:
            break
            
        fname = os.path.basename(fpath)
        done_marker = fpath + ".done"
        
        if os.path.exists(done_marker):
            print(f"Skipping {fname} (Already processed).")
            continue
            
        print(f"Processing {fname} ({processed_count + 1})...")
        
        try:
            # Check SQL presence
            in_sql = check_if_in_sql(fname)
            
            # Load Data
            df = pd.read_parquet(fpath)
            
            # Preprocess
            cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage', 'true_label'] if c in df.columns]
            if cols_to_drop:
                dvals = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number]).values
            else:
                dvals = df.select_dtypes(include=[np.number]).values
                
            X = dvals.astype(np.float32)
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True)
            spectrogram_n = (X - mean) / (std + 1e-6)
            spectrogram_2d = spectrogram_n.reshape(-1, 1, 76, 60)
            
            # --- INFERENCE ---
            preds_base = []
            confs_base = []
            preds_ens = []
            chunk_size = 128
            
            with torch.no_grad():
                t_tensor = torch.from_numpy(spectrogram_2d).to(device)
                total_len = len(t_tensor)
                
                for i in range(0, total_len, chunk_size):
                    batch = t_tensor[i:i+chunk_size]
                    
                    # Baseline
                    out_base = baseline_model(batch)
                    probs_base = torch.softmax(out_base, dim=1)
                    conf_b, p_base = torch.max(probs_base, 1)
                    preds_base.extend(p_base.cpu().numpy().tolist())
                    confs_base.extend(conf_b.cpu().numpy().tolist())
                    
                    # Ensemble
                    out_ens = ensemble_model(batch)
                    _, p_ens = torch.max(out_ens, 1)
                    preds_ens.extend(p_ens.cpu().numpy().tolist())
            
            # --- ARCHIVE TO SQL ---
            if not in_sql:
                # Extract patient ID (e.g. shhs1-200207 -> 200207? Or just use "shhs1")
                # Filename: shhs1-200207.parquet
                # Patient ID strategy: just use first part or the ID number.
                # In previous SQL: 'SC4072' is Patient ID for 'SC4072E.parquet'.
                # For 'shhs1-200207.parquet', patient_id is likely 'shhs1-200207'.
                # We can just use the core name.
                p_id = fname.replace(".parquet", "")
                save_to_sql(fname, preds_base, confs_base, baseline_ckpt_name, p_id)
                pass

            # --- REPORTING ---
            core_name = fname.replace(".parquet", "")
            xml_fname = f"{core_name}-profusion.xml"
            xml_path = os.path.join("parquet_files", "annotations-events-profusion", "shhs1", xml_fname)
            
            gt_labels = extract_gt_from_xml(xml_path)
            
            if len(gt_labels) > 0:
                output_base = os.path.join("png", f"{DATE_STAMP}_{core_name}")
                
                generate_comparative_report(
                    output_base, 
                    gt_labels, 
                    preds_ens, 
                    preds_base, 
                    "Baseline (ConvNext)", 
                    0.0, 0.0, "Calc",
                    lang='ES'
                )
                
                generate_comparative_report(
                    output_base, 
                    gt_labels, 
                    preds_ens, 
                    preds_base, 
                    "Baseline (ConvNext)", 
                    0.0, 0.0, "Calc",
                    lang='EN'
                )
                
                with open(done_marker, 'w') as f:
                    f.write(f"Processed on {datetime.now()}")
                    
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
