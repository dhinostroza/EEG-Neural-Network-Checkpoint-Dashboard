import torch
import os

def inspect_checkpoint(ckpt_path):
    print(f"Inspecting: {ckpt_path}")
    try:
        # Load on CPU
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        
        print("\n--- Top Level Keys ---")
        print(checkpoint.keys())
        
        print("\n--- Hyper Parameters ---")
        if 'hyper_parameters' in checkpoint:
            print(checkpoint['hyper_parameters'])
            
        print("\n--- Callbacks ---")
        if 'callbacks' in checkpoint:
            for key, val in checkpoint['callbacks'].items():
                print(f"Key: {key}")
                if isinstance(val, dict):
                    print(f"  Keys: {val.keys()}")
                    if 'best_model_score' in val:
                        print(f"  Best Score: {val['best_model_score']}")
                    if 'best_k_models' in val:
                         print(f"  Best K Models: {val['best_k_models']}")
        
        print("\n--- Early Stopping State ---")
        # Sometimes hidden in callbacks keys
        
        print("\n--- Loops ---")
        # Check for loop states if present
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    ckpt_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/2000 files/2025-09-19_04-34_convnext_base_consolidated_cwN1-6.5-epoch=3-val_loss=0.5971.ckpt"
    if os.path.exists(ckpt_path):
        inspect_checkpoint(ckpt_path)
    else:
        print(f"File not found: {ckpt_path}")
