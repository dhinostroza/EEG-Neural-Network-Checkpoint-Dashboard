import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Data provided by user
# Rows: True Labels, Cols: Predicted Labels (Standard Convention)
# Stages: Wake, N1, N2, N3, REM
cm = np.array([
    [68715, 1370, 454, 58, 1614],
    [689, 1017, 604, 28, 466],
    [351, 550, 14874, 1254, 770],
    [44, 16, 1716, 3917, 10],
    [340, 749, 830, 5, 5793]
])

labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

# Plotting
plt.figure(figsize=(10, 8))
sns.set_style("white")

# Create heatmap
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=labels, yticklabels=labels,
                 cbar=True, square=True, annot_kws={"size": 12})

# labels
plt.title('Overall Confusion Matrix (Stage 2)', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)

# Tweaking layout
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), "overall_confusion_matrix.png")
plt.savefig(output_path, dpi=300)
print(f"Confusion matrix saved to: {output_path}")
