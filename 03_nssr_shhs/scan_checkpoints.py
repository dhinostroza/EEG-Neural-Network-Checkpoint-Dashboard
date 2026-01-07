import torch
import os
import glob

def scan_checkpoints(base_dir):
    files = sorted(glob.glob(os.path.join(base_dir, "*.ckpt")))
    bad_files = []
    suspicious_files = []
    
    print(f"Scanning {len(files)} files in {base_dir}...\n")
    
    for f in files:
        filename = os.path.basename(f)
        size = os.path.getsize(f)
        
        # Check for empty or near-empty files
        if size < 1024: # Less than 1KB
            print(f"[EMPTY] {filename} (Size: {size} bytes)")
            bad_files.append(filename)
            continue
            
        try:
            # Try loading
            # map_location='cpu' is faster and doesn't require GPU
            ckpt = torch.load(f, map_location='cpu')
            
            if not isinstance(ckpt, dict):
                print(f"[INVALID FORMAT] {filename} (Loaded content is not a dict)")
                bad_files.append(filename)
                continue
                
            if 'state_dict' not in ckpt:
                print(f"[NO STATE_DICT] {filename}")
                bad_files.append(filename)
                continue
                
            # Check for suspicious values (e.g. very high loss indicating divergence)
            # This is subjective, but user asked for "non-usable"
            callbacks = ckpt.get('callbacks', {})
            val_loss = None
            for k, v in callbacks.items():
                if isinstance(v, dict) and 'best_model_score' in v:
                    val_loss = v['best_model_score']
            
            if val_loss is not None:
                # Handle tensor or float
                if hasattr(val_loss, 'item'):
                    val = val_loss.item()
                else:
                    val = float(val_loss)
                
                if val > 1000: # Arbitrary threshold for "exploded" loss
                    print(f"[SUSPICIOUS] {filename} (Val Loss: {val})")
                    suspicious_files.append(filename)

        except Exception as e:
            print(f"[CORRUPT] {filename} (Error: {str(e)})")
            bad_files.append(filename)
    
    print("\n--- Summary ---")
    if not bad_files and not suspicious_files:
        print("All files appear technically valid and usable.")
    else:
        # Combine lists
        to_rename = bad_files + suspicious_files
        print(f"Renaming {len(to_rename)} unusable/suspicious files with '_delete' suffix...")
        
        for filename in to_rename:
            full_path = os.path.join(base_dir, filename)
            new_path = full_path + "_delete"
            try:
                os.rename(full_path, new_path)
                print(f"  Renamed: {filename} -> {filename}_delete")
            except Exception as e:
                print(f"  Error renaming {filename}: {e}")

if __name__ == "__main__":
    base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/"
    scan_checkpoints(base_dir)
