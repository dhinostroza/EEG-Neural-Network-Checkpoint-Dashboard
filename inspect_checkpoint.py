import torch
import sys
import os

def inspect(path):
    print(f"--- Inspecting {os.path.basename(path)} ---")
    try:
        # map_location='cpu' is important if saved on GPU
        data = torch.load(path, map_location='cpu')
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            
            if 'hyper_parameters' in data:
                print("Hyperparameters:")
                for k, v in data['hyper_parameters'].items():
                    print(f"  {k}: {v}")
            else:
                print("No 'hyper_parameters' key found.")
                
            if 'epoch' in data:
                print(f"Epoch: {data['epoch']}")
            
            if 'global_step' in data:
                print(f"Global Step: {data['global_step']}")
            
            if 'state_dict' in data:
                print(f"State dict keys count: {len(data['state_dict'])}")
                # Guess architecture from first conv layer or similar if possible
                keys = list(data['state_dict'].keys())
                print(f"First 5 keys: {keys[:5]}")
                # Check for specific keys that might indicate model size
                # e.g. 'model.stages.3...'
            
            if 'callbacks' in data:
                print(f"Callbacks keys: {list(data['callbacks'].keys())}")
                # Sometimes metrics are stored in ModelCheckpoint callback
                for cb_key, cb_val in data['callbacks'].items():
                    if 'ModelCheckpoint' in cb_key:
                        print(f"  {cb_key} data: {cb_val}")

    except Exception as e:
        print(f"Error loading {path}: {e}")
    print("\n")

if __name__ == "__main__":
    base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/"
    files = [
        "2025-08-10 best-model-epoch=10-val_loss=1.34.ckpt",
        "best-model-20250903_163743_convnext_base_2000files_lr2e-05_cwN1-8.0.ckpt"
    ]
    
    for f in files:
        full_path = os.path.join(base_dir, f)
        if os.path.exists(full_path):
            inspect(full_path)
        else:
            print(f"File not found: {full_path}")
