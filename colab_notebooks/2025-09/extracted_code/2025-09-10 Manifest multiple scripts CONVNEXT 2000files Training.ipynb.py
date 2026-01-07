# ==============================================================================
# SCRIPT TO PRE-PROCESS AND CREATE A DATASET MANIFEST (DEFINITIVE SOLUTION)
# This script creates a completely isolated virtual environment to bypass all
# dependency conflicts in the Google Colab runtime.
# Run this script ONCE on a CPU runtime. It is resumable.
# ==============================================================================

import os
import sys
from google.colab import auth

# --- 1. AUTHENTICATE IN THE MAIN COLAB ENVIRONMENT ---
# This is the most critical step. We authenticate here to create the
# credential file that our isolated environment can use.
print("--- Step 1: Authenticating to Google Cloud in the main environment ---")
try:
    auth.authenticate_user()
    print("✅ Authentication successful. Credentials are now available for other processes.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not authenticate. The script cannot continue. Details: {e}")
    sys.exit() # Stop the script if authentication fails.

# --- 2. INSTALL THE VIRTUALENV CREATION TOOL ---
print("\n--- Step 2: Installing the robust 'virtualenv' package ---")
!pip install --upgrade -q virtualenv
print("✅ `virtualenv` installed.")

# --- 3. CREATE A COMPLETELY ISOLATED VIRTUAL ENVIRONMENT ---
print("\n--- Step 3: Creating a clean, isolated Python virtual environment ---")
# This creates a self-contained "sandbox" that does NOT inherit conflicting system packages.
!virtualenv manifest_env
print("✅ Virtual environment 'manifest_env' created successfully.")


# --- 4. INSTALL A KNOWN-GOOD SET OF PACKAGES INTO THE VIRTUAL ENVIRONMENT ---
print("\n--- Step 4: Installing a compatible set of dependencies into the clean environment ---")
# This is our known-good "toolchain". google-auth allows the ADC mechanism to work.
!manifest_env/bin/pip install --upgrade -q pip "pandas==2.2.2" "pyarrow==15.0.2" "fsspec==2023.6.0" gcsfs google-auth
print("✅ All dependencies installed successfully into 'manifest_env'.")


# --- 5. CREATE AND RUN THE PYTHON LOGIC SCRIPT ---
print("\n--- Step 5: Preparing and executing the manifest creation logic ---")

# This is the Python code that will be run inside the clean environment.
python_script_logic = r'''
import pandas as pd
import os
import time
import subprocess
import sys

# This script runs inside the clean environment.
# It relies on the Application Default Credentials (ADC) created by the main notebook.

# --- Configuration ---
GCS_BUCKET_BASE = "gs://shhs-sleepedfx-data-bucket"
GCS_SHHS1_PATH = f"{GCS_BUCKET_BASE}/shhs1_processed"
GCS_SHHS2_PATH = f"{GCS_BUCKET_BASE}/shhs2_processed"
OUTPUT_GCS_PATH = f"{GCS_BUCKET_BASE}/metadata/shhs_dataset_manifest.csv"
CHECKPOINT_INTERVAL = 200 # How often to save progress

print("\n" + "="*80)
print("--- MANIFEST CREATION SCRIPT (RUNNING IN ISOLATED ENV) ---")
print(f"  -> Target Manifest Path: {OUTPUT_GCS_PATH}")
print("="*80 + "\n")

# --- Gather File Paths using subprocess for robustness ---
def get_gcs_files(path):
    try:
        # The 'gsutil' command will automatically use the ADC file for auth.
        result = subprocess.run(['gsutil', 'ls', f'{path}/*.parquet'], capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n')
        return [f for f in files if f.startswith('gs://')]
    except Exception as e:
        print(f"     ❌ ERROR running gsutil for {path}. Details: {e}")
        return []

print("--- Listing all .parquet files in GCS buckets ---")
all_file_paths = get_gcs_files(GCS_SHHS1_PATH) + get_gcs_files(GCS_SHHS2_PATH)

if not all_file_paths:
    print("\n❌ ERROR: Failed to find any .parquet files. Exiting.")
    sys.exit()

print(f"✅ Success! Found a total of {len(all_file_paths)} files to process.")

# --- Robustly check for and initialize the manifest state file ---
processed_files = set()
manifest_data = []
try:
    print(f"\n--- 🔎 Step A: Searching for existing manifest to resume progress... ---")
    print(f"  -> Checking path: {OUTPUT_GCS_PATH}")
    partial_df = pd.read_csv(OUTPUT_GCS_PATH)
    if 'file_path' in partial_df.columns:
        processed_files = set(partial_df['file_path'])
        manifest_data = partial_df.to_dict('records')
    print(f"  -> ✅ Found and successfully loaded existing manifest with {len(processed_files)} entries. Resuming session.")
except FileNotFoundError:
    print("  -> No existing manifest found. Attempting to create one now to initialize state.")
    try:
        empty_df = pd.DataFrame(columns=['file_path', 'epoch_count'])
        empty_df.to_csv(OUTPUT_GCS_PATH, index=False)
        print("  -> ✅ Successfully created and saved an empty manifest file to GCS. The process is now resumable.")
    except Exception as e:
        print(f"  -> ❌ FATAL ERROR: Could not create the initial manifest file at {OUTPUT_GCS_PATH}.")
        print(f"     Please check your GCS permissions. Details: {e}")
        sys.exit()
except Exception as e:
    print(f"  -> ❌ FATAL ERROR: Could not read the existing manifest file. It may be corrupted. Details: {e}")
    sys.exit()

# --- Process Files ---
print("\n--- Step B: Processing all files not already in the manifest ---")
files_to_process = [fp for fp in all_file_paths if fp not in processed_files]
print(f"  -> {len(processed_files)} files already processed.")
print(f"  -> {len(files_to_process)} files remaining in this session.")
time.sleep(2)

if not files_to_process:
    print("\n✅ All files have already been processed. Manifest is up to date.")
else:
    for i, f_path in enumerate(files_to_process):
        if (i + 1) % 50 == 0 or i == len(files_to_process) - 1 or i == 0:
            print(f"  -> Progress: [{i+1}/{len(files_to_process)}] | Overall: [{len(processed_files) + i + 1}/{len(all_file_paths)}] | Processing: {os.path.basename(f_path)}")
        try:
            df_labels = pd.read_parquet(f_path, columns=['label'])
            num_valid = df_labels['label'].isin([0, 1, 2, 3, 4]).sum()
            if num_valid > 0:
                manifest_data.append({'file_path': f_path, 'epoch_count': num_valid})
        except Exception as e:
            print(f"     -> ⚠️ WARNING: Could not process {os.path.basename(f_path)}. Skipping. Error: {e}")

        if (i > 0 and (i + 1) % CHECKPOINT_INTERVAL == 0) and manifest_data:
            print(f"     -> 💾 CHECKPOINT: Saving progress ({len(manifest_data)} total entries) to GCS...")
            try:
                temp_df = pd.DataFrame(manifest_data)
                temp_df.to_csv(OUTPUT_GCS_PATH, index=False)
                print("        ...✅ Progress successfully saved.")
            except Exception as e:
                print(f"        ...❌ WARNING: Checkpoint save failed. Will retry later. Error: {e}")

    print("\n✅ Epoch counting complete for this session.")

# --- Final Save and Summary ---
if manifest_data:
    print("\n--- Step C: Creating and saving the final manifest file ---")
    final_manifest_df = pd.DataFrame(manifest_data)
    print(f"  -> Attempting to save final manifest with {len(final_manifest_df)} entries...")
    final_manifest_df.to_csv(OUTPUT_GCS_PATH, index=False)
    print(f"  -> ✅ Final manifest successfully written to GCS.")

    print("\n--- Final Manifest Data Preview ---")
    print("  -> First 5 rows:")
    print(final_manifest_df.head().to_string())

    print("\n" + "="*80)
    print("🎉 SCRIPT COMPLETE 🎉")
    print(f"The dataset manifest is now available at: {OUTPUT_GCS_PATH}")
    print("You can now switch to a GPU runtime and use the manifest-powered training script.")
    print("="*80)
else:
    print("\n✅ No new files to process or all files resulted in errors/no epochs.")
'''

# Write the script to a file
with open("run_manifest_creation.py", "w") as f:
    f.write(python_script_logic)

# Execute the script using the virtual environment's python
!manifest_env/bin/python run_manifest_creation.py

print("\n--- Script execution finished. ---")

# ==============================================================================
# SCRIPT TO PRE-PROCESS AND CREATE A DATASET MANIFEST (DEFINITIVE SOLUTION)
# This script creates a completely isolated virtual environment to bypass all
# dependency conflicts in the Google Colab runtime.
# Run this script ONCE on a CPU runtime. It is resumable.
# ==============================================================================

import os
import sys
from google.colab import auth

# --- 1. AUTHENTICATE IN THE MAIN COLAB ENVIRONMENT ---
# This is the most critical step. We authenticate here to create the
# credential file that our isolated environment can use.
print("--- Step 1: Authenticating to Google Cloud in the main environment ---")
try:
    auth.authenticate_user()
    print("✅ Authentication successful. Credentials are now available for other processes.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not authenticate. The script cannot continue. Details: {e}")
    sys.exit() # Stop the script if authentication fails.

# --- 2. INSTALL THE VIRTUALENV CREATION TOOL ---
print("\n--- Step 2: Installing the robust 'virtualenv' package ---")
!pip install --upgrade -q virtualenv
print("✅ `virtualenv` installed.")

# --- 3. CREATE A COMPLETELY ISOLATED VIRTUAL ENVIRONMENT ---
print("\n--- Step 3: Creating a clean, isolated Python virtual environment ---")
# This creates a self-contained "sandbox" that does NOT inherit conflicting system packages.
!virtualenv manifest_env
print("✅ Virtual environment 'manifest_env' created successfully.")


# --- 4. INSTALL A KNOWN-GOOD SET OF PACKAGES INTO THE VIRTUAL ENVIRONMENT ---
print("\n--- Step 4: Installing a compatible set of dependencies into the clean environment ---")
# This is our known-good "toolchain". google-auth allows the ADC mechanism to work.
!manifest_env/bin/pip install --upgrade -q pip "pandas==2.2.2" "pyarrow==15.0.2" "fsspec==2023.6.0" gcsfs google-auth
print("✅ All dependencies installed successfully into 'manifest_env'.")


# --- 5. CREATE AND RUN THE PYTHON LOGIC SCRIPT ---
print("\n--- Step 5: Preparing and executing the manifest creation logic ---")

# This is the Python code that will be run inside the clean environment.
python_script_logic = r'''
import pandas as pd
import os
import time
import subprocess
import sys

# This script runs inside the clean environment.
# It relies on the Application Default Credentials (ADC) created by the main notebook.

# --- Configuration ---
GCS_BUCKET_BASE = "gs://shhs-sleepedfx-data-bucket"
GCS_SHHS1_PATH = f"{GCS_BUCKET_BASE}/shhs1_processed"
GCS_SHHS2_PATH = f"{GCS_BUCKET_BASE}/shhs2_processed"
OUTPUT_GCS_PATH = f"{GCS_BUCKET_BASE}/metadata/shhs_dataset_manifest.csv"
CHECKPOINT_INTERVAL = 200 # How often to save progress

print("\n" + "="*80)
print("--- MANIFEST CREATION SCRIPT (RUNNING IN ISOLATED ENV) ---")
print(f"  -> Target Manifest Path: {OUTPUT_GCS_PATH}")
print("="*80 + "\n")

# --- Gather File Paths using subprocess for robustness ---
def get_gcs_files(path):
    try:
        # The 'gsutil' command will automatically use the ADC file for auth.
        result = subprocess.run(['gsutil', 'ls', f'{path}/*.parquet'], capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n')
        return [f for f in files if f.startswith('gs://')]
    except Exception as e:
        print(f"     ❌ ERROR running gsutil for {path}. Details: {e}")
        return []

print("--- Listing all .parquet files in GCS buckets ---")
all_file_paths = get_gcs_files(GCS_SHHS1_PATH) + get_gcs_files(GCS_SHHS2_PATH)

if not all_file_paths:
    print("\n❌ ERROR: Failed to find any .parquet files. Exiting.")
    sys.exit()

print(f"✅ Success! Found a total of {len(all_file_paths)} files to process.")

# --- Robustly check for and initialize the manifest state file ---
processed_files = set()
manifest_data = []
try:
    print(f"\n--- 🔎 Step A: Searching for existing manifest to resume progress... ---")
    print(f"  -> Checking path: {OUTPUT_GCS_PATH}")
    partial_df = pd.read_csv(OUTPUT_GCS_PATH)
    if 'file_path' in partial_df.columns:
        processed_files = set(partial_df['file_path'])
        manifest_data = partial_df.to_dict('records')
    print(f"  -> ✅ Found and successfully loaded existing manifest with {len(processed_files)} entries. Resuming session.")
except FileNotFoundError:
    print("  -> No existing manifest found. Attempting to create one now to initialize state.")
    try:
        empty_df = pd.DataFrame(columns=['file_path', 'epoch_count'])
        empty_df.to_csv(OUTPUT_GCS_PATH, index=False)
        print("  -> ✅ Successfully created and saved an empty manifest file to GCS. The process is now resumable.")
    except Exception as e:
        print(f"  -> ❌ FATAL ERROR: Could not create the initial manifest file at {OUTPUT_GCS_PATH}.")
        print(f"     Please check your GCS permissions. Details: {e}")
        sys.exit()
except Exception as e:
    print(f"  -> ❌ FATAL ERROR: Could not read the existing manifest file. It may be corrupted. Details: {e}")
    sys.exit()

# --- Process Files ---
print("\n--- Step B: Processing all files not already in the manifest ---")
files_to_process = [fp for fp in all_file_paths if fp not in processed_files]
print(f"  -> {len(processed_files)} files already processed.")
print(f"  -> {len(files_to_process)} files remaining in this session.")
time.sleep(2)

if not files_to_process:
    print("\n✅ All files have already been processed. Manifest is up to date.")
else:
    for i, f_path in enumerate(files_to_process):
        if (i + 1) % 50 == 0 or i == len(files_to_process) - 1 or i == 0:
            print(f"  -> Progress: [{i+1}/{len(files_to_process)}] | Overall: [{len(processed_files) + i + 1}/{len(all_file_paths)}] | Processing: {os.path.basename(f_path)}")
        try:
            df_labels = pd.read_parquet(f_path, columns=['label'])
            num_valid = df_labels['label'].isin([0, 1, 2, 3, 4]).sum()
            if num_valid > 0:
                manifest_data.append({'file_path': f_path, 'epoch_count': num_valid})
        except Exception as e:
            print(f"     -> ⚠️ WARNING: Could not process {os.path.basename(f_path)}. Skipping. Error: {e}")

        if (i > 0 and (i + 1) % CHECKPOINT_INTERVAL == 0) and manifest_data:
            print(f"     -> 💾 CHECKPOINT: Saving progress ({len(manifest_data)} total entries) to GCS...")
            try:
                temp_df = pd.DataFrame(manifest_data)
                temp_df.to_csv(OUTPUT_GCS_PATH, index=False)
                print("        ...✅ Progress successfully saved.")
            except Exception as e:
                print(f"        ...❌ WARNING: Checkpoint save failed. Will retry later. Error: {e}")

    print("\n✅ Epoch counting complete for this session.")

# --- Final Save and Summary ---
if manifest_data:
    print("\n--- Step C: Creating and saving the final manifest file ---")
    final_manifest_df = pd.DataFrame(manifest_data)
    print(f"  -> Attempting to save final manifest with {len(final_manifest_df)} entries...")
    final_manifest_df.to_csv(OUTPUT_GCS_PATH, index=False)
    print(f"  -> ✅ Final manifest successfully written to GCS.")

    print("\n--- Final Manifest Data Preview ---")
    print("  -> First 5 rows:")
    print(final_manifest_df.head().to_string())

    print("\n" + "="*80)
    print("🎉 SCRIPT COMPLETE 🎉")
    print(f"The dataset manifest is now available at: {OUTPUT_GCS_PATH}")
    print("You can now switch to a GPU runtime and use the manifest-powered training script.")
    print("="*80)
else:
    print("\n✅ No new files to process or all files resulted in errors/no epochs.")
'''

# Write the script to a file
with open("run_manifest_creation.py", "w") as f:
    f.write(python_script_logic)

# Execute the script using the virtual environment's python
!manifest_env/bin/python run_manifest_creation.py

print("\n--- Script execution finished. ---")

# ==================== NEW CELL ====================

# ==============================================================================
# SCRIPT TO TRAIN THE MODEL (DEFINITIVE, STABLE & RESUMABLE SOLUTION)
# This script uses an isolated environment and a single, cached data worker
# to guarantee stability and reliable checkpointing for long training runs.
# ==============================================================================

import os
import sys
from google.colab import auth
from google.colab import drive

# --- 1. SETUP THE COLAB ENVIRONMENT ---
print("--- Step 1: Preparing the main Colab environment ---")
try:
    auth.authenticate_user()
    print("✅ Authentication successful. Credentials are now available for other processes.")
except Exception as e:
    sys.exit(f"❌ FATAL ERROR: Could not authenticate. Details: {e}")

try:
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted successfully.")
except Exception as e:
    sys.exit(f"❌ FATAL ERROR: Could not mount Google Drive. Details: {e}")

# --- 2. CREATE AND PROVISION THE ISOLATED TRAINING ENVIRONMENT ---
print("\n--- Step 2: Creating and provisioning the isolated training environment ---")
!pip install --upgrade -q virtualenv
print("  -> `virtualenv` installed.")
!virtualenv train_env
print("  -> Virtual environment 'train_env' created successfully.")
!train_env/bin/pip install --upgrade -q pip "pytorch-lightning" "timm" "pandas>=2.0" "pyarrow>=15.0" "fsspec>=2023.6.0" gcsfs google-auth matplotlib seaborn scikit-learn
print("  -> All dependencies installed successfully into 'train_env'.")


# --- 3. CREATE AND RUN THE FULL TRAINING SCRIPT ---
print("\n--- Step 3: Preparing and executing the training logic in the isolated environment ---")

python_script_logic = r'''
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from collections import Counter
from datetime import datetime
import os
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

torch.set_float32_matmul_precision('medium')
print("✅ Libraries imported inside virtual environment.")

# --- All class and function definitions are here for clarity ---

def get_model(model_name='convnext_base', num_classes=5, pretrained=True):
    if model_name == 'convnext_base':
        model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained)
        original_conv = model.stem[0]
        new_first_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding, bias=(original_conv.bias is not None))
        with torch.no_grad():
            if original_conv.weight.shape[1] == 3:
                new_first_conv.weight[:, :] = original_conv.weight.clone().mean(dim=1, keepdim=True)
        model.stem[0] = new_first_conv
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    return model

class SleepStageClassifierLightning(pl.LightningModule):
    def __init__(self, model_name, learning_rate, class_weights, epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name=self.hparams.model_name)
        self.train_accuracy = MulticlassAccuracy(num_classes=5)
        self.val_accuracy = MulticlassAccuracy(num_classes=5)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float) if class_weights else None)

    def forward(self, x):
        return self.model(x)

    def normalize_on_gpu(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        return (x - mean) / (std + 1e-6)

    def spec_augment(self, x, time_mask_param=10, freq_mask_param=10):
        _, _, num_freq_bins, num_time_steps = x.shape
        f_mask_width = int(np.random.uniform(0.0, freq_mask_param))
        f_mask_start = int(np.random.uniform(0.0, num_freq_bins - f_mask_width))
        x[:, :, f_mask_start:f_mask_start + f_mask_width, :] = 0
        t_mask_width = int(np.random.uniform(0.0, time_mask_param))
        t_mask_start = int(np.random.uniform(0.0, num_time_steps - t_mask_width))
        x[:, :, :, t_mask_start:t_mask_start + t_mask_width] = 0
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        x_normalized = self.normalize_on_gpu(x)
        x_augmented = self.spec_augment(x_normalized)
        y_pred_logits = self(x_augmented)
        loss = self.loss_fn(y_pred_logits, y_true)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy(y_pred_logits, y_true), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        x_normalized = self.normalize_on_gpu(x)
        y_pred_logits = self(x_normalized)
        loss = self.loss_fn(y_pred_logits, y_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy(y_pred_logits, y_true), on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=1e-7)
        return [optimizer], [scheduler]

class CombinedDataset(Dataset):
    def __init__(self, manifest_path, num_files=None):
        manifest_df = pd.read_csv(manifest_path)
        if num_files:
            manifest_df = manifest_df.head(num_files)
        self.file_paths = manifest_df['file_path'].tolist()
        self.cumulative_epochs = np.cumsum(manifest_df['epoch_count'].values)
        self.total_epochs = self.cumulative_epochs[-1]
        self._cache = {} # The cache is safe with a single worker
        print(f"✅ Dataset initialized from manifest. Found {self.total_epochs} epochs across {len(self.file_paths)} files.")

    def __len__(self):
        return self.total_epochs

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_epochs, idx, side='right')
        local_idx = idx - (self.cumulative_epochs[file_idx - 1] if file_idx > 0 else 0)
        file_path = self.file_paths[file_idx]
        if file_path not in self._cache:
            # This is slow on the first epoch, but fast on all subsequent epochs
            self._cache[file_path] = pd.read_parquet(file_path)[lambda df: df['label'].isin([0, 1, 2, 3, 4])].reset_index(drop=True)

        row = self._cache[file_path].iloc[local_idx]
        label = np.int64(row['label'])
        spectrogram_flat = row.drop('label').values.astype(np.float32)
        spectrogram_2d = spectrogram_flat.reshape(1, 76, 60)
        return torch.from_numpy(spectrogram_2d), torch.tensor(label)

def generate_performance_report(ckpt_path, dataloader, device, save_dir, exp_name):
    # ... placeholder
    pass

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    GCS_MANIFEST_PATH = "gs://shhs-sleepedfx-data-bucket/metadata/shhs_dataset_manifest.csv"
    NUM_FILES_TO_USE = 2000
    CLASS_WEIGHTS = [0.7, 6.5, 0.5, 1.5, 1.2]
    EPOCHS = 40
    BATCH_SIZE = 256
    LEARNING_RATE = 2e-5
    # --- MODIFICATION: Revert to a single worker for stability and to enable caching ---
    NUM_WORKERS = 0
    DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/final_model_checkpoint/"

    # --- Data Loading ---
    full_dataset = CombinedDataset(GCS_MANIFEST_PATH, num_files=NUM_FILES_TO_USE)
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Using num_workers=0 makes persistent_workers irrelevant
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Experiment Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_convnext_base_{NUM_FILES_TO_USE}files_Augmented_cwN1-{CLASS_WEIGHTS[1]}"

    model = SleepStageClassifierLightning('convnext_base', LEARNING_RATE, CLASS_WEIGHTS, EPOCHS)

    drive_log_dir = "/content/drive/MyDrive/sleep_logs/"
    os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

    csv_logger = CSVLogger(drive_log_dir, name=experiment_name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=DRIVE_CHECKPOINT_DIR, filename=f"best-model-{experiment_name}", save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')

    # --- Find latest checkpoint to resume from ---
    print(f"\n--- Searching for latest checkpoint in {DRIVE_CHECKPOINT_DIR} ---")
    checkpoint_files = glob.glob(os.path.join(DRIVE_CHECKPOINT_DIR, "*.ckpt"))
    latest_checkpoint = None
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"✅ Found checkpoint. Resuming training from: {os.path.basename(latest_checkpoint)}")
    else:
        print("  -> No checkpoint found. Starting a new training run.")

    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator="gpu", devices=1, logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="bf16-mixed", gradient_clip_val=1.0
    )

    print(f"\n🚀🚀🚀 Starting/Resuming augmented training for experiment: {experiment_name} 🚀🚀🚀")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=latest_checkpoint)
    print(f"\n✅ Training complete!")

    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        # generate_performance_report is now a local function, so we call it directly
        # generate_performance_report(checkpoint_callback.best_model_path, val_loader, model.device, DRIVE_CHECKPOINT_DIR, experiment_name)
        print("Performance report generation placeholder")
    else:
        print("  -> No checkpoint was saved. Skipping final report.")
'''

# Write the script to a file
with open("run_training.py", "w") as f:
    f.write(python_script_logic)

# Execute the script using the virtual environment's python, forcing the correct backend
!MPLBACKEND=Agg train_env/bin/python run_training.py

print("\n--- Script execution finished. ---")

# ==================== NEW CELL ====================

# ==============================================================================
# SCRIPT TO ARCHIVE OLD MODEL CHECKPOINTS
# This script mounts Google Drive and moves all existing .ckpt files
# into a dedicated archive folder to ensure the next training run starts fresh.
# ==============================================================================

import os
import glob
from google.colab import drive

print("--- Step 1: Mounting Google Drive ---")
try:
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not mount Google Drive. Halting script. Details: {e}")
    # We stop the script if we can't access the drive
    exit()

# Define the paths for the checkpoint and the new archive directory
checkpoint_dir = "/content/drive/MyDrive/final_model_checkpoint/"
archive_dir = os.path.join(checkpoint_dir, "_archive")

print(f"\n--- Step 2: Creating archive directory ---")
print(f"  -> Ensuring directory exists: {archive_dir}")
# The 'exist_ok=True' flag prevents any errors if the folder already exists
os.makedirs(archive_dir, exist_ok=True)
print("✅ Archive directory is ready.")


print(f"\n--- Step 3: Finding and moving checkpoint files ---")
# Find all files ending with .ckpt in the main checkpoint directory
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

if not checkpoint_files:
    print("  -> No checkpoint files found in the main directory. Nothing to move.")
else:
    print(f"  -> Found {len(checkpoint_files)} checkpoint file(s) to archive.")
    # Use the 'mv' command to move all found .ckpt files into the archive.
    # The -v flag makes the command verbose, listing each file as it's moved.
    !mv -v /content/drive/MyDrive/final_model_checkpoint/*.ckpt /content/drive/MyDrive/final_model_checkpoint/_archive/
    print("\n  -> ✅ All checkpoint files have been successfully moved to the archive.")

print("\n" + "="*80)
print("🎉 SCRIPT COMPLETE 🎉")
print("Your checkpoint directory is now clean.")
print("You can now switch back to a GPU runtime and run the main training script.")
print("="*80)

# ==================== NEW CELL ====================

