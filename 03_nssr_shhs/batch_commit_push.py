import os
import subprocess
import glob
import time

def run_git_cmd(cmd_list, show_output=False, retries=3):
    for attempt in range(retries):
        try:
            result = subprocess.run(cmd_list, check=True, text=True, capture_output=True)
            if show_output:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            if "index.lock" in e.stderr:
                print(f"Lock file detected. Retrying {attempt + 1}/{retries} in 2s...")
                time.sleep(2)
                continue
                
            print(f"Error executing: {' '.join(cmd_list)}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False
            
    return False

def main():
    png_files = sorted(glob.glob("png/*.png"))
    total_files = len(png_files)
    batch_size = 50
    start_index = 850 
    
    print(f"Found {total_files} PNG files in directory.")
    print(f"Starting from index {start_index} based on last success.")

    for i in range(start_index, total_files, batch_size):
        batch = png_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}: files {i} to {i + len(batch)}")
        
        # Add files
        if not run_git_cmd(["git", "add"] + batch):
            print("Git add failed. Stopping.")
            break
            
        # Commit
        msg = f"chore: Batch upload PNGs {i} to {i + len(batch)}"
        if not run_git_cmd(["git", "commit", "-m", msg], show_output=True):
             # Check if it was because nothing to commit (already added in previous run?)
             # Just warn and continue to push
             print("Commit failed (possibly empty). Attempting push anyway...")

        # Push 
        print("Pushing batch...")
        if not run_git_cmd(["git", "push", "origin", "main"], show_output=True):
             print("Push failed. Retrying in 10s...")
             time.sleep(10)
             if not run_git_cmd(["git", "push", "origin", "main"], show_output=True):
                 print("Push failed twice. Stopping.")
                 break
        
        print(f"Batch {i // batch_size + 1} complete.")

if __name__ == "__main__":
    main()
