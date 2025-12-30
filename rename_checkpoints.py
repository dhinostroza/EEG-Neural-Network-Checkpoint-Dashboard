import os
import glob
import datetime

def rename_files(base_dir):
    # grab all files including .ckpt and .ckpt_delete (and others?) 
    # User said "the files", so probably just the checkpoints we are working with.
    # I'll look for everything that ends with .ckpt or .ckpt_delete
    files = glob.glob(os.path.join(base_dir, "*"))
    
    print(f"Renaming files in {base_dir} to start with modification time...\n")
    
    count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Skip directories and scripts/reports
        if filename.endswith(".py") or filename.endswith(".md") or filename.endswith(".ipynb") or os.path.isdir(file_path):
            continue
            
        # Get modification time
        mtime = os.path.getmtime(file_path)
        dt_obj = datetime.datetime.fromtimestamp(mtime)
        timestamp_str = dt_obj.strftime("%Y-%m-%d_%H-%M")
        
        # Avoid double prefixing if I ran this before (check if starts with date pattern)
        # But user wants me to do it based on screenshot dates.
        # Check if filename already starts with this exact timestamp to avoid idempotency issues
        if filename.startswith(timestamp_str):
            print(f"Skipping {filename} (already has timestamp prefix)")
            continue
            
        new_filename = f"{timestamp_str}_{filename}"
        new_path = os.path.join(base_dir, new_filename)
        
        try:
            os.rename(file_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            count += 1
        except Exception as e:
            print(f"Error renaming {filename}: {e}")

    print(f"\nTotal files renamed: {count}")

if __name__ == "__main__":
    base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/"
    rename_files(base_dir)
