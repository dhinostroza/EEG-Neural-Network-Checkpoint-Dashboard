import torch
import os

def average_checkpoints(ckpt_paths, output_path):
    print(f"Starting model averaging for {len(ckpt_paths)} models...")
    
    avg_state_dict = {}
    base_checkpoint = None
    
    # 1. Load and Sum Weights
    for i, path in enumerate(ckpt_paths):
        print(f"Loading: {path}")
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if 'state_dict' not in checkpoint:
                print(f"Error: No 'state_dict' in {path}")
                return
            
            # Use the last checkpoint as the "base" for metadata (hyperparams, etc.)
            if i == len(ckpt_paths) - 1:
                base_checkpoint = checkpoint
                
            state_dict = checkpoint['state_dict']
            
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    if key not in avg_state_dict:
                        avg_state_dict[key] = tensor.clone().float()
                    else:
                        avg_state_dict[key] += tensor.float()
                else:
                    pass
                    
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return

    # 2. Divide by N (Average)
    print("Averaging weights...")
    for key in avg_state_dict:
        if avg_state_dict[key].is_floating_point():
             avg_state_dict[key] /= len(ckpt_paths)
        else:
             avg_state_dict[key] = avg_state_dict[key] // len(ckpt_paths)

    # 3. Create Output Checkpoint
    print("Creating fused checkpoint...")
    if base_checkpoint:
        # Replace state_dict with averaged one
        base_checkpoint['state_dict'] = avg_state_dict
        
        # Save
        print(f"Saving to: {output_path}")
        torch.save(base_checkpoint, output_path)
        print("Success! Fused model saved.")
    else:
        print("Error: Could not establish base checkpoint.")

if __name__ == "__main__":
    # Base directory
    BASE_DIR = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/2000 files"
    
    # Input files
    FILES = [
        "2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt",
        "2025-09-09_15-43_convnext_base_2000files_Augmented_cwN1-6.5.ckpt",
        "2025-09-19_04-34_convnext_base_consolidated_cwN1-6.5-epoch=3-val_loss=0.5971.ckpt"
    ]
    
    FULL_PATHS = [os.path.join(BASE_DIR, f) for f in FILES]
    
    # Output file
    OUTPUT_FILE = os.path.join(BASE_DIR, "2026-01-10_Modelo_Definitivo_Fusionado.ckpt")
    
    average_checkpoints(FULL_PATHS, OUTPUT_FILE)
