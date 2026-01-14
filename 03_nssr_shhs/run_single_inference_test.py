import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, accuracy_score

# Configuration
TEST_FILE = "shhs1-200001_processed.parquet"
MODEL_PATH = "ensemble_model_scripted.pt"
XML_PATH = "parquet_files/annotations-events-profusion/shhs1/shhs1-200001-profusion.xml"
BASE_OUTPUT_NAME = "manual_test_shhs1-200001"

# Reverse Map for parsing SQL
STR_TO_STAGE = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "Unknown": -1}

# Localization Dictionaries
TEXT_EN = {
    "suptitle": "Global Accuracy Comparison: Baseline ({:.2f}%) vs Ensemble ({:.2f}%)\nImprovement: +{:.2f}%",
    "baseline_label": "Baseline (ConvNext)",
    "ensemble_label": "Ensemble",
    "gt_label": "Ground Truth",
    "baseline_title": "Baseline: {} (Acc: {})",
    "ensemble_title": "New: Ensemble Model (Acc: {:.2f}%)",
    "no_baseline": "No Baseline Data Found",
    "cm_title": "Confusion Matrix ({})",
    "dist_title": "Class Distribution ({})",
    "y_label": "True Class",
    "x_label_base": "Predicted (Baseline)",
    "x_label_ens": "Predicted (Ensemble)",
    "epoch_label": "Epoch (30s)",
    "stages": ["Wake", "N1", "N2", "N3", "REM"],
    "stages_short": ["W", "N1", "N2", "N3", "R"]
}

TEXT_ES = {
    "suptitle": "Comparación Global de Precisión: Modelo Base ({:.2f}%) vs Ensemble ({:.2f}%)\nMejora: +{:.2f}%",
    "baseline_label": "Modelo Base",
    "ensemble_label": "Ensemble",
    "gt_label": "Ground Truth",
    "baseline_title": "Modelo Base: {} (Prec: {})",
    "ensemble_title": "Nuevo: Modelo Ensemble (Prec: {:.2f}%)",
    "no_baseline": "No se encontraron datos del modelo base",
    "cm_title": "Matriz de Confusión ({})",
    "dist_title": "Distribución de Clases ({})",
    "y_label": "Clase Real",
    "x_label_base": "Predicho (Modelo Base)",
    "x_label_ens": "Predicho (Ensemble)",
    "epoch_label": "Época (30s)",
    "stages": ["Vigilia", "N1", "N2", "N3", "REM"],
    "stages_short": ["W", "N1", "N2", "N3", "R"]
}

def parse_sql_history(sql_path, filename):
    print(f"Parsing SQL history from {sql_path} for {filename}...")
    prev_preds = {} # epoch -> stage_int
    model_name = "Unknown"
    
    try:
        with open(sql_path, "r") as f:
            for line in f:
                if filename in line and "INSERT INTO" not in line:
                    parts = line.split(",")
                    if len(parts) >= 6:
                        try:
                            epoch = int(parts[2].strip())
                            pred_str = parts[3].strip().replace("'", "")
                            m_name = parts[5].strip().replace("'", "")
                            
                            val = STR_TO_STAGE.get(pred_str, -1)
                            prev_preds[epoch] = val
                            model_name = m_name
                        except:
                            continue
                            
        if not prev_preds:
            return [], "None"
            
        max_epoch = max(prev_preds.keys())
        stages = [prev_preds.get(i, -1) for i in range(max_epoch + 1)]
        
        if "convnext" in model_name:
            short_name = "Baseline (ConvNext)"
        else:
            short_name = f"Baseline ({model_name[:10]}...)"
            
        print(f"Found {len(stages)} epochs from previous model: {short_name}")
        return stages, short_name
        
    except Exception as e:
        print(f"SQL Parse Error: {e}")
        return [], "None"

def extract_gt_from_xml(xml_path):
    print(f"Loading GT from {xml_path}...")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        stages_nodes = root.findall(".//{*}SleepStage")
        if not stages_nodes:
            stages_nodes = root.findall(".//SleepStage")
            
        print(f"Found {len(stages_nodes)} SleepStage nodes.")
        
        stages = []
        for stage in stages_nodes:
            try:
                val = int(stage.text)
                if val == 0: val = 0 # Wake
                elif val == 1: val = 1 # N1
                elif val == 2: val = 2 # N2
                elif val == 3: val = 3 # N3
                elif val == 4: val = 3 # N4 -> N3
                elif val == 5: val = 4 # REM
                else: val = -1 
                stages.append(val)
            except:
                stages.append(-1)
        return stages
    except Exception as e:
        print(f"XML Error: {e}")
        return []

def generate_comparative_report(gt_labels, predictions, prev_predictions, prev_model_name, acc_ensemble, acc_prev, acc_prev_str, lang='ES'):
    T = TEXT_ES if lang == 'ES' else TEXT_EN
    print(f"Generating {lang} report...")
    
    L = min(len(gt_labels), len(predictions))
    y_true_final = gt_labels[:L]
    y_pred_final = predictions[:L]
    
    valid_mask = [t != -1 for t in y_true_final]
    y_t_acc = [t for t, m in zip(y_true_final, valid_mask) if m]
    y_p_acc = [p for p, m in zip(y_pred_final, valid_mask) if m]
    
    y_pr_acc = []
    if len(prev_predictions) >= L:
         y_prev_final = prev_predictions[:L]
         y_pr_acc = [p for p, m in zip(y_prev_final, valid_mask) if m]

    fig = plt.figure(figsize=(16, 20)) 
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.2, 1, 1.2]) 
    
    # SUPER TITLE
    diff = (acc_ensemble - acc_prev) * 100
    fig.suptitle(T["suptitle"].format(acc_prev*100, acc_ensemble*100, diff), 
                 fontsize=20, fontweight='bold', y=0.96)
    
    # ROW 0: BASELINE HYPNOGRAM
    ax_prev_hyp = fig.add_subplot(gs[0, :])
    if len(prev_predictions) >= L:
        ax_prev_hyp.step(range(L), prev_predictions[:L], where='post', label=T["baseline_label"], color='gray', linewidth=2, alpha=0.7, zorder=1)
        y_gt_plot = [l if l != -1 else float('nan') for l in gt_labels[:L]]
        ax_prev_hyp.step(range(L), y_gt_plot, where='post', label=T["gt_label"], color='#4caf50', linestyle='--', linewidth=2, zorder=2)
        ax_prev_hyp.set_title(T["baseline_title"].format(prev_model_name, acc_prev_str), fontweight='bold', color='#666666')
    else:
        ax_prev_hyp.text(0.5, 0.5, T["no_baseline"], ha='center', va='center')
        
    ax_prev_hyp.set_yticks([0, 1, 2, 3, 4])
    ax_prev_hyp.set_yticklabels(T["stages"])
    ax_prev_hyp.invert_yaxis()
    ax_prev_hyp.legend(loc='upper right')
    ax_prev_hyp.grid(True, alpha=0.3)

    # ROW 1: BASELINE METRICS
    if len(prev_predictions) >= L:
        # CM
        ax_prev_cm = fig.add_subplot(gs[1, 0])
        cm_prev = confusion_matrix(y_t_acc, y_pr_acc, labels=[0,1,2,3,4])
        cax_p = ax_prev_cm.matshow(cm_prev, cmap='Oranges') 
        fig.colorbar(cax_p, ax=ax_prev_cm, fraction=0.046, pad=0.04)
        ax_prev_cm.set_title(T["cm_title"].format(prev_model_name), fontweight='bold', color='#666666')
        ax_prev_cm.set_xticklabels([''] + T["stages"])
        ax_prev_cm.set_yticklabels([''] + T["stages"])
        ax_prev_cm.set_ylabel(T["y_label"])
        ax_prev_cm.set_xlabel(T["x_label_base"])
        
        thresh_p = cm_prev.max() / 2.
        for i in range(cm_prev.shape[0]):
            for j in range(cm_prev.shape[1]):
                ax_prev_cm.text(j, i, str(cm_prev[i, j]), va='center', ha='center',
                                color="white" if cm_prev[i, j] > thresh_p else "black")
        
        # Dist
        ax_prev_dist = fig.add_subplot(gs[1, 1])
        t_counts = [y_t_acc.count(l) for l in [0,1,2,3,4]]
        p_counts = [y_pr_acc.count(l) for l in [0,1,2,3,4]]
        x = np.arange(5)
        w = 0.35
        ax_prev_dist.bar(x - w/2, t_counts, w, label=T["gt_label"], color='#4caf50')
        ax_prev_dist.bar(x + w/2, p_counts, w, label=T["baseline_label"], color='#ff9800') 
        ax_prev_dist.set_title(T["dist_title"].format(prev_model_name), fontweight='bold', color='#666666')
        ax_prev_dist.set_xticks(x)
        ax_prev_dist.set_xticklabels(T["stages_short"])
        ax_prev_dist.legend()
    
    # ROW 2: ENSEMBLE HYPNOGRAM
    ax_ens_hyp = fig.add_subplot(gs[2, :])
    ax_ens_hyp.step(range(L), predictions[:L], where='post', label=T["ensemble_label"], color='#2196f3', linewidth=2, alpha=0.7, zorder=1)
    ax_ens_hyp.step(range(L), y_gt_plot, where='post', label=T["gt_label"], color='#4caf50', linestyle='--', linewidth=2, zorder=2)
    
    ax_ens_hyp.set_yticks([0, 1, 2, 3, 4])
    ax_ens_hyp.set_yticklabels(T["stages"])
    ax_ens_hyp.set_title(T["ensemble_title"].format(acc_ensemble*100), fontweight='bold', color='#2196f3')
    ax_ens_hyp.set_xlabel(T["epoch_label"])
    ax_ens_hyp.invert_yaxis()
    ax_ens_hyp.legend(loc='upper right')
    ax_ens_hyp.grid(True, alpha=0.3)
    
    # ROW 3: ENSEMBLE METRICS
    # CM
    ax_ens_cm = fig.add_subplot(gs[3, 0])
    cm_ens = confusion_matrix(y_t_acc, y_p_acc, labels=[0,1,2,3,4])
    cax_e = ax_ens_cm.matshow(cm_ens, cmap='Blues')
    fig.colorbar(cax_e, ax=ax_ens_cm, fraction=0.046, pad=0.04)
    ax_ens_cm.set_title(T["cm_title"].format("Ensemble"), fontweight='bold', color='#2196f3')
    ax_ens_cm.set_xticklabels([''] + T["stages"])
    ax_ens_cm.set_yticklabels([''] + T["stages"])
    ax_ens_cm.set_ylabel(T["y_label"])
    ax_ens_cm.set_xlabel(T["x_label_ens"])
    
    thresh_e = cm_ens.max() / 2.
    for i in range(cm_ens.shape[0]):
        for j in range(cm_ens.shape[1]):
            ax_ens_cm.text(j, i, str(cm_ens[i, j]), va='center', ha='center',
                           color="white" if cm_ens[i, j] > thresh_e else "black")
    
    # Dist
    ax_ens_dist = fig.add_subplot(gs[3, 1])
    e_counts = [y_p_acc.count(l) for l in [0,1,2,3,4]]
    ax_ens_dist.bar(x - w/2, t_counts, w, label=T["gt_label"], color='#4caf50')
    ax_ens_dist.bar(x + w/2, e_counts, w, label=T["ensemble_label"], color='#2196f3')
    ax_ens_dist.set_title(T["dist_title"].format("Ensemble"), fontweight='bold', color='#2196f3')
    ax_ens_dist.set_xticks(x)
    ax_ens_dist.set_xticklabels(T["stages_short"])
    ax_ens_dist.legend()

    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    
    final_path = f"{BASE_OUTPUT_NAME}_FINAL_{lang}.png"
    plt.savefig(final_path, dpi=150)
    plt.close(fig)
    print(f"Saved {lang} REPORT to {final_path}")

    return final_path

def main():
    print("--- Starting Comparative Inference Test ---")
    
    # Load Data
    print(f"Loading {TEST_FILE}...")
    df = pd.read_parquet(TEST_FILE)
    
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
    
    # Inference
    print(f"Loading Model {MODEL_PATH}...")
    device = torch.device("cpu") 
    model = torch.jit.load(MODEL_PATH)
    model.to(device)
    model.eval()
    
    predictions = []
    chunk_size = 128
    with torch.no_grad():
        t_tensor = torch.from_numpy(spectrogram_2d).to(device)
        total = len(t_tensor)
        for i in range(0, total, chunk_size):
            batch = t_tensor[i:i+chunk_size]
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy().tolist())

    # Load SQL History
    prev_predictions, prev_model_name = parse_sql_history("predictions.sql", TEST_FILE)

    # Load GT
    gt_labels = extract_gt_from_xml(XML_PATH)
    
    if len(gt_labels) > 0 and len(predictions) > 0:
        # Metrics
        L = min(len(gt_labels), len(predictions))
        y_true_final = gt_labels[:L]
        y_pred_final = predictions[:L]
        
        valid_mask = [t != -1 for t in y_true_final]
        y_t_acc = [t for t, m in zip(y_true_final, valid_mask) if m]
        y_p_acc = [p for p, m in zip(y_pred_final, valid_mask) if m]
        acc_ensemble = accuracy_score(y_t_acc, y_p_acc)
        
        acc_prev = 0.0
        acc_prev_str = "N/A"
        if len(prev_predictions) >= L:
             y_prev_final = prev_predictions[:L]
             y_pr_acc = [p for p, m in zip(y_prev_final, valid_mask) if m]
             acc_prev = accuracy_score(y_t_acc, y_pr_acc)
             acc_prev_str = f"{acc_prev*100:.2f}%"
        
        # Args
        args = (gt_labels, predictions, prev_predictions, prev_model_name, acc_ensemble, acc_prev, acc_prev_str)
        
        # Generate Reports
        generate_comparative_report(*args, lang='ES')
        generate_comparative_report(*args, lang='EN')
        
    else:
        print("ERROR: No aligned GT data found. Cannot generate metrics.")

if __name__ == "__main__":
    main()
