import json
import os

def extract_code(notebook_path):
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    print(f"--- Code from {os.path.basename(notebook_path)} ---")
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if source.strip():
                print(f"\n### Cell {i} ###")
                print(source)

if __name__ == "__main__":
    nb_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/ConvNext_Tiny_500_files_GCS.ipynb"
    extract_code(nb_path)
