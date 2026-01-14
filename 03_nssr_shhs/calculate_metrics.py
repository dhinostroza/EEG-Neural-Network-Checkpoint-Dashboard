import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report, confusion_matrix

def main():
    csv_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/ensemble_results.csv"
    
    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter valid
    # Check if 'true_label' is -1
    valid_mask = df['true_label'] != -1
    df_valid = df[valid_mask]
    
    y_true = df_valid['true_label'].values
    y_pred = df_valid['pred_ensemble'].values
    
    print(f"Valid samples: {len(y_true)} / {len(df)}")
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print("\n" + "="*40)
    print("FINAL ENSEMBLE METRICS")
    print("="*40)
    print(f"Global Accuracy:   {acc:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")
    print(f"Global Accuracy:   {acc:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")
    print(f"F1-Score (Macro):  {f1_macro:.4f}")
    print(f"F1-Score (Weight): {f1_weighted:.4f}")
    
    stagenames = ['Wake', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    
    print("\n--- Confusion Matrix ---")
    print("Rows: True, Cols: Pred")
    print(stagenames)
    print(cm)
    
    print("\n--- Precision per Class ---")
    # Precision = TP / (TP + FP) = Diagonal / Col Sum
    col_sums = cm.sum(axis=0)
    precisions = np.divide(cm.diagonal(), col_sums, out=np.zeros_like(cm.diagonal(), dtype=float), where=col_sums!=0)
    for i, p in enumerate(precisions):
        print(f"Precision {stagenames[i]}: {p:.4f} (pred_count={col_sums[i]})")

    print("\n--- Per Class Recall (Sensitivity) ---")
    
    # Sensitivity = TP / (TP + FN) = Diagonal / Row Sum
    row_sums = cm.sum(axis=1)
    sensitivities = np.divide(cm.diagonal(), row_sums, out=np.zeros_like(cm.diagonal(), dtype=float), where=row_sums!=0)
    
    for i, s in enumerate(sensitivities):
        print(f"Recall {stagenames[i]}: {s:.4f} (n={row_sums[i]})")
        
    print("\n--- Per Class F1-Score ---")
    # f1_score with average=None returns array of shape (n_classes,)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0,1,2,3,4])
    for i, s in enumerate(f1_per_class):
        print(f"F1 {stagenames[i]}: {s:.4f}")

if __name__ == "__main__":
    main()
