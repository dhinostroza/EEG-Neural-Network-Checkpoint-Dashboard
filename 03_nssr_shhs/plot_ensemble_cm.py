import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

# 1. Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "ensemble_results.csv")
output_path = os.path.join(base_dir, "ensemble_confusion_matrix.png")

# 2. Load Data
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# 3. Filter Valid Data
df_valid = df[df['true_label'] != -1]
y_true = df_valid['true_label']
y_pred = df_valid['pred_ensemble']

# 4. Compute Matrix
labels = [0, 1, 2, 3, 4]
class_names = ["Wake", "N1", "N2", "N3", "REM"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 5. Plot
plt.figure(figsize=(10, 8))
sns.set_style("white")

# Heatmap
# fmt='d' for integers
# cmap='Blues' for blue shades
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=class_names, yticklabels=class_names,
                 cbar=True, square=True, annot_kws={"size": 12})

# Labels
plt.title('Ensemble Model Confusion Matrix', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)

plt.tight_layout()

# 6. Save
plt.savefig(output_path, dpi=300)
print(f"Graph saved to: {output_path}")
