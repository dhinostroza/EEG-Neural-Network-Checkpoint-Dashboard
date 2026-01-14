import pandas as pd
from sklearn.metrics import confusion_matrix
import os

base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs"
csv_path = os.path.join(base_dir, "ensemble_results.csv")

if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# Filter invalid GT
df_valid = df[df['true_label'] != -1]

y_true = df_valid['true_label']
y_pred = df_valid['pred_ensemble']

# labels: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
labels = [0, 1, 2, 3, 4]
class_names = ["Wake", "N1", "N2", "N3", "REM"]

cm = confusion_matrix(y_true, y_pred, labels=labels)

print("\nConfusion Matrix (Ensemble Solution):")
print("-" * 30)
print(cm)
print("-" * 30)
print("Labels: Wake, N1, N2, N3, REM")

# Optional: Print with headers for clarity
print("\nFormatted:")
print(f"{'':<6} {'Wake':<6} {'N1':<6} {'N2':<6} {'N3':<6} {'REM':<6}")
for i, row in enumerate(cm):
    print(f"{class_names[i]:<6} " + " ".join([f"{val:<6}" for val in row]))
