import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

csv_path = "03_nssr_shhs/comparison_results_SC4001E.csv"
df = pd.read_csv(csv_path)

y_true = df['true_label']
y_old = df['pred_old_sept04']
y_new = df['pred_new_fused_jan10']

target_names = ["Wake", "N1", "N2", "N3", "REM"]

print("--- RESULTS ANALYSIS ---")

print(f"\nOld Model (Sept 04) Accuracy: {accuracy_score(y_true, y_old):.4f}")
print("Confusion Matrix Old:")
print(confusion_matrix(y_true, y_old))

print(f"\nNew Fused Model (Jan 10) Accuracy: {accuracy_score(y_true, y_new):.4f}")
print("Confusion Matrix New:")
print(confusion_matrix(y_true, y_new))

# Specific N1 Check
n1_mask = (y_true == 1)
if n1_mask.sum() > 0:
    n1_acc_old = accuracy_score(y_true[n1_mask], y_old[n1_mask])
    n1_acc_new = accuracy_score(y_true[n1_mask], y_new[n1_mask])
    print(f"\nN1 Recall (Sensitivity):")
    print(f"Old: {n1_acc_old:.4f}")
    print(f"New: {n1_acc_new:.4f}")
    
# Specific N2 Check
n2_mask = (y_true == 2)
if n2_mask.sum() > 0:
    n2_acc_old = accuracy_score(y_true[n2_mask], y_old[n2_mask])
    n2_acc_new = accuracy_score(y_true[n2_mask], y_new[n2_mask])
    print(f"\nN2 Recall (Sensitivity):")
    print(f"Old: {n2_acc_old:.4f}")
    print(f"New: {n2_acc_new:.4f}")
