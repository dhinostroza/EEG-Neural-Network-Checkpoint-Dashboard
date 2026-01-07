
import json
import numpy as np

path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/python_results/results_8_subjects_python_v1_2.json"

with open(path, 'r') as f:
    data = json.load(f)

folds = data['folds']
print(f"Loaded {len(folds)} folds.")

all_cms = [np.array(f['confusion_matrix']) for f in folds]
cm_sum = np.sum(all_cms, axis=0)

print("v1.1 Matrix (Copy this to JS):")
print(cm_sum.tolist())
