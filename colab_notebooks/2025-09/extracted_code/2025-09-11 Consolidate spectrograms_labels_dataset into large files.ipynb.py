# ==============================================================================
# SCRIPT TO CONSOLIDATE THE ENTIRE DATASET INTO A FEW LARGE FILES
# This is the final pre-processing step to eliminate all I/O bottlenecks.
# Run this script ONCE on a CPU runtime with high RAM. It is resumable.
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import time
from google.colab import auth, drive

# --- 1. SETUP THE COLAB ENVIRONMENT ---
print("--- Step 1: Preparing the main Colab environment ---")
try:
    auth.authenticate_user()
    print("✅ Authentication successful.")
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted successfully.")
except Exception as e:
    sys.exit(f"❌ FATAL ERROR: Could not set up environment. Details: {e}")

# --- 2. CONFIGURATION ---
GCS_MANIFEST_PATH = "gs://shhs-sleepedfx-data-bucket/metadata/shhs_dataset_manifest.csv"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/shhs_consolidated_data/"
# We will save progress here to make the script resumable
PROGRESS_FILE_PATH = os.path.join(DRIVE_OUTPUT_DIR, "_progress.txt")
FINAL_DATA_PATH = os.path.join(DRIVE_OUTPUT_DIR, "all_spectrograms.npy")
FINAL_LABELS_PATH = os.path.join(DRIVE_OUTPUT_DIR, "all_labels.npy")

print("\n" + "="*80)
print("--- CONFIGURATION ---")
print(f"  -> Source Manifest: {GCS_MANIFEST_PATH}")
print(f"  -> Output Directory: {DRIVE_OUTPUT_DIR}")
print("="*80 + "\n")

# --- 3. PREPARE ENVIRONMENT ---
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
!pip install --upgrade -q "pandas>=2.0" "pyarrow>=15.0" "fsspec>=2023.6.0" gcsfs

# --- 4. LOAD MANIFEST AND CHECK PROGRESS ---
print("--- Step 2: Loading manifest and checking for saved progress ---")
try:
    manifest_df = pd.read_csv(GCS_MANIFEST_PATH)
    all_file_paths = manifest_df['file_path'].tolist()
    print(f"✅ Manifest loaded. Found {len(all_file_paths)} files to process.")
except Exception as e:
    sys.exit(f"❌ FATAL ERROR: Could not load manifest file. Details: {e}")

processed_files = set()
if os.path.exists(PROGRESS_FILE_PATH):
    with open(PROGRESS_FILE_PATH, 'r') as f:
        processed_files = set(f.read().splitlines())
    print(f"✅ Found progress file. Resuming. {len(processed_files)} files already processed.")

# --- 5. PROCESS FILES AND CONSOLIDATE DATA ---
files_to_process = [fp for fp in all_file_paths if fp not in processed_files]
print(f"\n--- Step 3: Consolidating data from {len(files_to_process)} remaining files ---")
print("This is a very long process. Progress will be updated periodically.")
time.sleep(2)

# Load existing data if it exists, otherwise initialize empty lists
if os.path.exists(FINAL_DATA_PATH) and os.path.exists(FINAL_LABELS_PATH) and processed_files:
    print("  -> Loading previously consolidated data...")
    all_spectrograms = np.load(FINAL_DATA_PATH, mmap_mode='r+').tolist()
    all_labels = np.load(FINAL_LABELS_PATH, mmap_mode='r+').tolist()
    print("     ...done.")
else:
    all_spectrograms = []
    all_labels = []

for i, f_path in enumerate(files_to_process):
    try:
        if (i + 1) % 10 == 0 or i == len(files_to_process) - 1 or i == 0:
            print(f"\r  -> Progress: [{i+1}/{len(files_to_process)}] | Overall: [{len(processed_files) + i + 1}/{len(all_file_paths)}] | File: {os.path.basename(f_path)}", end="")

        df = pd.read_parquet(f_path)
        df_filtered = df[df['label'].isin([0, 1, 2, 3, 4])]

        labels = df_filtered['label'].values.astype(np.int64)
        spectrograms_flat = df_filtered.drop('label', axis=1).values.astype(np.float32)

        all_labels.extend(labels)
        all_spectrograms.extend(spectrograms_flat)

        # Update progress
        processed_files.add(f_path)

        # Periodically save checkpoint
        if (i > 0 and (i + 1) % 100 == 0):
            print(f"\n     -> 💾 CHECKPOINT: Saving progress ({len(all_labels)} total epochs)...")
            np.save(FINAL_DATA_PATH, np.array(all_spectrograms, dtype=np.float32))
            np.save(FINAL_LABELS_PATH, np.array(all_labels, dtype=np.int64))
            with open(PROGRESS_FILE_PATH, 'w') as f:
                f.write("\n".join(sorted(list(processed_files))))
            print("        ...✅ Progress saved.")

    except Exception as e:
        print(f"\n     -> ⚠️ WARNING: Could not process {os.path.basename(f_path)}. Skipping. Error: {e}")

# --- 6. FINAL SAVE ---
print("\n\n--- Step 4: Saving final consolidated dataset ---")
try:
    final_spectrograms = np.array(all_spectrograms, dtype=np.float32)
    final_labels = np.array(all_labels, dtype=np.int64)

    print(f"  -> Final data shape: {final_spectrograms.shape}")
    print(f"  -> Final labels shape: {final_labels.shape}")

    np.save(FINAL_DATA_PATH, final_spectrograms)
    np.save(FINAL_LABELS_PATH, final_labels)
    with open(PROGRESS_FILE_PATH, 'w') as f:
        f.write("\n".join(sorted(list(processed_files))))

    print(f"\n✅ Final dataset saved successfully to {DRIVE_OUTPUT_DIR}")
    print("\n" + "="*80)
    print("🎉 SCRIPT COMPLETE 🎉")
    print("You can now use the final, simplified training script on a GPU runtime.")
    print("="*80)
except Exception as e:
    print(f"❌ FATAL ERROR: Could not save the final dataset. Details: {e}")

# ==================== NEW CELL ====================

