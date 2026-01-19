import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add current dir to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import convert_edf_to_parquet, detect_architecture
from ensemble_logic import load_ensemble_models, predict_ensemble

import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

def generate_hecam_report(ensemble_preds, baseline_preds=None, filename="report.png"):
    """
    Generates a full comparative report mimicking the system standard,
    adapted for missing Ground Truth (GT).
    """
    STAGES = ["Wake", "N1", "N2", "N3", "REM"]
    STAGES_SHORT = ["W", "N1", "N2", "N3", "R"]
    
    fig = plt.figure(figsize=(16, 16))
    
    # Layout:
    # Row 0: Title
    # Row 1: Agreement Matrix & Class Distribution
    # Row 2: Baseline Hypnogram
    # Row 3: Ensemble Hypnogram
    # Row 4: Comparison Hypnogram (Overlay) - Optional, mimicking original
    # Original was: 0:Title, 1:Base(CM+Dist), 2:BaseHyp, 3:Ens(CM+Dist), 4:EnsHyp
    # Since we don't have GT, we can't do CM vs GT.
    # We will do:
    # 0: Title
    # 1: Agreement Matrix (Base vs Ens) | Class Distributions (Grouped Bar)
    # 2: Baseline Hypnogram
    # 3: Ensemble Hypnogram
    # 4: Overlay Hypnogram
    
    gs = gridspec.GridSpec(5, 2, height_ratios=[0.2, 1.2, 0.8, 0.8, 0.8])
    
    # --- ROW 0: TITLE ---
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = "Reporte Comparativo: HECAM\nBase (Old) vs Ensamble (New)"
    # Add agreement %
    if baseline_preds is not None:
        L = min(len(ensemble_preds), len(baseline_preds))
        agree = np.mean(ensemble_preds[:L] == baseline_preds[:L]) * 100
        title_text += f"\nAcuerdo entre modelos: {agree:.2f}%"
        
    ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=18, fontweight='bold')
    
    # --- ROW 1: AGREEMENT MATRIX ---
    ax_agree = fig.add_subplot(gs[1, 0])
    
    if baseline_preds is not None:
        L = min(len(ensemble_preds), len(baseline_preds))
        # Base (Rows) vs Ensemble (Cols)
        cm = confusion_matrix(baseline_preds[:L], ensemble_preds[:L], labels=[0,1,2,3,4])
        
        cax = ax_agree.matshow(cm, cmap='Purples')
        fig.colorbar(cax, ax=ax_agree, fraction=0.046, pad=0.04)
        
        ax_agree.set_title("Matriz de Acuerdo\n(Filas=Base, Cols=Ensamble)", fontweight='bold', color='#663399')
        ax_agree.set_xticks(np.arange(5))
        ax_agree.set_yticks(np.arange(5))
        ax_agree.set_xticklabels(STAGES)
        ax_agree.set_yticklabels(STAGES)
        ax_agree.set_ylabel("Modelo Base (Old)")
        ax_agree.set_xlabel("Modelo Ensamble (New)")
        
        # Annotate
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_agree.text(j, i, str(cm[i, j]), va='center', ha='center',
                            color="white" if cm[i, j] > thresh else "black")
    else:
        ax_agree.text(0.5, 0.5, "Sin Baseline para comparar", ha='center', va='center')
        ax_agree.axis('off')

    # --- ROW 1 (Right): DISTRIBUTION ---
    ax_dist = fig.add_subplot(gs[1, 1])
    x = np.arange(5)
    w = 0.35
    
    ens_counts = [list(ensemble_preds).count(i) for i in range(5)]
    
    if baseline_preds is not None:
        base_counts = [list(baseline_preds).count(i) for i in range(5)]
        ax_dist.bar(x - w/2, base_counts, w, label='Base', color='#ff9900')
        ax_dist.bar(x + w/2, ens_counts, w, label='Ensamble', color='#3366ff')
    else:
        ax_dist.bar(x, ens_counts, w, label='Ensamble', color='#3366ff')
        
    ax_dist.set_title("Distribución de Clases Predichas", fontweight='bold')
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(STAGES_SHORT)
    ax_dist.legend()
    ax_dist.grid(axis='y', alpha=0.3)

    # --- SHARED FOR HYPNOGRAMS ---
    time_axis = np.arange(len(ensemble_preds))
    
    # --- ROW 2: BASELINE HYPNOGRAM ---
    ax_base_hyp = fig.add_subplot(gs[2, :])
    if baseline_preds is not None:
        L = min(len(time_axis), len(baseline_preds))
        ax_base_hyp.step(time_axis[:L], baseline_preds[:L], where='post', color='#ff9900', linewidth=1.5)
        ax_base_hyp.set_title("Hipnograma: Modelo Base", fontweight='bold', color='#ff9900')
        ax_base_hyp.set_yticks(np.arange(5))
        ax_base_hyp.set_yticklabels(STAGES_SHORT)
        ax_base_hyp.invert_yaxis()
        ax_base_hyp.grid(True, alpha=0.3)
        ax_base_hyp.set_xlim(0, len(time_axis))
    else:
        ax_base_hyp.text(0.5, 0.5, "Sin Baseline", ha='center')
        ax_base_hyp.axis('off')
        
    # --- ROW 3: ENSEMBLE HYPNOGRAM ---
    ax_ens_hyp = fig.add_subplot(gs[3, :])
    ax_ens_hyp.step(time_axis, ensemble_preds, where='post', color='#3366ff', linewidth=1.5)
    ax_ens_hyp.set_title("Hipnograma: Ensamble (Actual)", fontweight='bold', color='#3366ff')
    ax_ens_hyp.set_yticks(np.arange(5))
    ax_ens_hyp.set_yticklabels(STAGES_SHORT)
    ax_ens_hyp.invert_yaxis()
    ax_ens_hyp.grid(True, alpha=0.3)
    ax_ens_hyp.set_xlim(0, len(time_axis))
    
    # --- ROW 4: OVERLAY (Differences) ---
    ax_diff = fig.add_subplot(gs[4, :])
    if baseline_preds is not None:
        L = min(len(ensemble_preds), len(baseline_preds))
        # Plot both thin
        ax_diff.step(time_axis[:L], baseline_preds[:L], where='post', color='#ff9900', alpha=0.5, linewidth=1, label='Base')
        ax_diff.step(time_axis[:L], ensemble_preds[:L], where='post', color='#3366ff', alpha=0.5, linewidth=1, label='Ensamble')
        
        # Highlight differences
        diffs = np.where(ensemble_preds[:L] != baseline_preds[:L])[0]
        if len(diffs) > 0:
            # Scatter dots on diffs
            ax_diff.scatter(diffs, ensemble_preds[diffs], color='red', s=10, marker='|', label='Diferencia')
            
        ax_diff.set_title("Superposición y Diferencias", fontweight='bold', color='#444444')
        ax_diff.set_yticks(np.arange(5))
        ax_diff.set_yticklabels(STAGES_SHORT)
        ax_diff.invert_yaxis()
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend(loc='upper right')
        ax_diff.set_xlabel("Época (30s)")
        ax_diff.set_xlim(0, len(time_axis))
    else:
        ax_diff.axis('off')
        
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close(fig)
    print(f"Saved report to: {filename}")
    return True

def main():
    # 1. Configuration
    bdf_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/Caso HECAM EEG BDF/Patient_M_36 aos/Registro de EEG-HECAM.bdf"
    csv_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/comparison_results_Registro de EEG-HECAM_processed.bdf.csv"
    
    if not os.path.exists(bdf_path):
        print(f"Error: BDF file not found at {bdf_path}")
        return

    work_dir = os.path.dirname(os.path.abspath(__file__))
    output_parquet = os.path.join(work_dir, "temp_hecam.parquet")
    
    # Checkpoints
    ckpt_dir = os.path.join(work_dir, "checkpoint_files")
    ckpt_paths = glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True)
    
    if not ckpt_paths:
        print("Error: No checkpoints found in checkpoint_files/")
        return
        
    selected_ckpts = ckpt_paths[:3] 
    print(f"Selected {len(selected_ckpts)} checkpoints for ensemble.")

    # 2. Convert BDF -> Parquet
    print(f"Converting {bdf_path} to Parquet...")
    success, msg = convert_edf_to_parquet(bdf_path, output_parquet)
    if not success:
        print(f"Conversion failed: {msg}")
        return
    print(f"Conversion successful: {output_parquet}")

    # 3. Load Parquet
    df = pd.read_parquet(output_parquet)
    print(f"Loaded {len(df)} epochs.")

    # 4. Load Models & Run Inference
    print("Loading models...")
    models = load_ensemble_models(selected_ckpts)
    
    print("Running Ensemble Inference...")
    predictions, _ = predict_ensemble(models, df)
    
    # 5. Load Baseline from CSV
    baseline_preds = None
    if os.path.exists(csv_path):
        try:
            print(f"Loading baseline from CSV: {csv_path}")
            df_csv = pd.read_csv(csv_path)
            if 'pred_old' in df_csv.columns:
                baseline_preds = df_csv['pred_old'].values
                print(f"Loaded {len(baseline_preds)} baseline predictions.")
            else:
                print("Warning: 'pred_old' column not found in CSV.")
        except Exception as e:
            print(f"Error loading CSV: {e}")
    else:
        print(f"Warning: CSV not found at {csv_path}")

    # 6. Generate Plot
    png_dir = os.path.join(work_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    
    import datetime
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    output_png = os.path.join(png_dir, f"{today}_Registro de EEG-HECAM_es.png")
    
    generate_hecam_report(
        predictions, 
        baseline_preds=baseline_preds, 
        filename=output_png
    )
    
    # Cleanup
    if os.path.exists(output_parquet):
        os.remove(output_parquet)
        print("Cleaned up temp parquet.")

if __name__ == "__main__":
    main()
