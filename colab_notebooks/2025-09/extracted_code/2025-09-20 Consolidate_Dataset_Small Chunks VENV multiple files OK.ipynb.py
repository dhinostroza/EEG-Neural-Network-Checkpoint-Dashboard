!rm -f /content/drive/MyDrive/shhs_consolidated_data/*

# ==================== NEW CELL ====================

# ==============================================================================
# SCRIPT TO CONSOLIDATE THE DATASET (DEFINITIVE, MEMORY-SAFE)
# This script uses a small chunk size to prevent memory crashes, ensuring a
# stable and resumable pre-processing run.
# Run this script ONCE on a CPU runtime with high RAM.
# ==============================================================================

import os
import sys
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

# --- 2. CREATE AND PROVISION THE ISOLATED ENVIRONMENT ---
print("\n--- Step 2: Creating and provisioning the isolated consolidation environment ---")
!pip install --upgrade -q virtualenv
print("  -> `virtualenv` installed.")
!virtualenv consolidate_env
print("  -> Virtual environment 'consolidate_env' created successfully.")
!consolidate_env/bin/pip install --upgrade -q pip "pandas==2.2.2" "pyarrow==15.0.2" "fsspec>=2023.6.0" gcsfs google-auth
print("  -> All dependencies installed successfully into 'consolidate_env'.")


# --- 3. CREATE AND RUN THE CONSOLIDATION LOGIC SCRIPT ---
print("\n--- Step 3: Preparing and executing the consolidation logic in the isolated environment ---")

python_script_logic = r'''
import pandas as pd
import numpy as np
import os
import sys
import time
import glob
import subprocess

# --- CONFIGURATION ---
GCS_MANIFEST_PATH = "gs://shhs-sleepedfx-data-bucket/metadata/shhs_dataset_manifest.csv"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/shhs_consolidated_data/"
PROGRESS_FILE_PATH = os.path.join(DRIVE_OUTPUT_DIR, "_progress.txt")
# --- MODIFICATION: Reduced chunk size to a safe value to prevent memory crashes ---
CHUNK_SIZE = 50

print("\n" + "="*80)
print("--- DATA CONSOLIDATION SCRIPT (RUNNING IN ISOLATED ENV) ---")
print(f"  -> Output Directory: {DRIVE_OUTPUT_DIR}")
print(f"  -> Chunk Size: {CHUNK_SIZE} files per chunk")
print("="*80 + "\n")

# --- Prepare Environment ---
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)

# --- Load Manifest ---
print("--- Step A: Loading manifest... ---")
try:
    manifest_df = pd.read_csv(GCS_MANIFEST_PATH)
    all_file_paths = manifest_df['file_path'].tolist()
    print(f"✅ Manifest loaded. Found {len(all_file_paths)} files to process.")
except Exception as e:
    sys.exit(f"❌ FATAL ERROR: Could not load manifest file from GCS. Details: {e}")

# --- Check for Progress ---
processed_files = set()
if os.path.exists(PROGRESS_FILE_PATH):
    with open(PROGRESS_FILE_PATH, 'r') as f:
        processed_files = set(f.read().splitlines())
    print(f"✅ Found progress file. Resuming. {len(processed_files)} files already processed.")

# --- Process Files in Chunks ---
files_to_process = [fp for fp in all_file_paths if fp not in processed_files]
print(f"\n--- Step B: Consolidating data from {len(files_to_process)} remaining files ---")
print("This is a very long process. Progress will be updated periodically.")
time.sleep(2)

# Determine starting chunk number based on existing files
chunk_num = len(glob.glob(os.path.join(DRIVE_OUTPUT_DIR, "spectrograms_*.npy")))
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

        all_labels.extend(labels.tolist())
        all_spectrograms.extend(spectrograms_flat.tolist())
        processed_files.add(f_path)

        if (i > 0 and (i + 1) % CHUNK_SIZE == 0) or ((i + 1) == len(files_to_process) and len(all_labels) > 0):
            print(f"\n     -> 💾 CHUNK COMPLETE: Saving chunk #{chunk_num} with {len(all_labels)} epochs...")

            chunk_data_path = os.path.join(DRIVE_OUTPUT_DIR, f"spectrograms_{chunk_num}.npy")
            chunk_labels_path = os.path.join(DRIVE_OUTPUT_DIR, f"labels_{chunk_num}.npy")

            np.save(chunk_data_path, np.array(all_spectrograms, dtype=np.float32))
            np.save(chunk_labels_path, np.array(all_labels, dtype=np.int64))

            with open(PROGRESS_FILE_PATH, 'w') as f:
                f.write("\n".join(sorted(list(processed_files))))

            print(f"        ...✅ Chunk #{chunk_num} saved successfully.")

            all_spectrograms.clear()
            all_labels.clear()
            chunk_num += 1

    except Exception as e:
        print(f"\\n     -> ⚠️ WARNING: Could not process {os.path.basename(f_path)}. Skipping. Error: {e}")

print("\n\n--- Step C: Final verification ---")
final_chunks = glob.glob(os.path.join(DRIVE_OUTPUT_DIR, "spectrograms_*.npy"))
print(f"  -> Found {len(final_chunks)} saved data chunks in the output directory.")

print("\\n" + "="*80)
print("🎉 SCRIPT COMPLETE 🎉")
print("Your dataset is now consolidated into manageable chunks in your Google Drive.")
print("You can now use the final, simplified training script on a GPU runtime.")
print("="*80)
'''

# Write the script to a file
with open("run_consolidation.py", "w") as f:
    f.write(python_script_logic)

# Execute the script using the virtual environment's python
!consolidate_env/bin/python run_consolidation.py

print("\n--- Script execution finished. ---")

# ==================== NEW CELL ====================

# ==============================================================================
# SCRIPT TO CONSOLIDATE THE DATASET (DEFINITIVE, VIRTUAL ENV & MEMORY-SAFE)
# This script creates an isolated environment AND processes data in chunks
# to ensure a stable, memory-safe, and resumable run.
# Run this script ONCE on a CPU runtime with high RAM.
# ==============================================================================

import os
import sys
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

# --- 2. CREATE AND PROVISION THE ISOLATED ENVIRONMENT ---
print("\n--- Step 2: Creating and provisioning the isolated consolidation environment ---")
!pip install --upgrade -q virtualenv
print("  -> `virtualenv` installed.")
!virtualenv consolidate_env
print("  -> Virtual environment 'consolidate_env' created successfully.")
# Install our known-good, conflict-free packages into the clean room
!consolidate_env/bin/pip install --upgrade -q pip "pandas==2.2.2" "pyarrow==15.0.2" "fsspec>=2023.6.0" gcsfs google-auth
print("  -> All dependencies installed successfully into 'consolidate_env'.")


# --- 3. CREATE AND RUN THE CONSOLIDATION LOGIC SCRIPT ---
print("\n--- Step 3: Preparing and executing the consolidation logic in the isolated environment ---")

python_script_logic = r'''
import pandas as pd
import numpy as np
import os
import sys
import time
import glob
import subprocess

# --- Configuration ---
GCS_MANIFEST_PATH = "gs://shhs-sleepedfx-data-bucket/metadata/shhs_dataset_manifest.csv"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/shhs_consolidated_data/"
PROGRESS_FILE_PATH = os.path.join(DRIVE_OUTPUT_DIR, "_progress.txt")
CHUNK_SIZE = 50 # Process 50 files at a time

print("\n" + "="*80)
print("--- DATA CONSOLIDATION SCRIPT (RUNNING IN ISOLATED ENV) ---")
print(f"  -> Output Directory: {DRIVE_OUTPUT_DIR}")
print("="*80 + "\n")

# --- Prepare Environment ---
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)

# --- Load Manifest ---
print("--- Step A: Loading manifest... ---")
try:
    manifest_df = pd.read_csv(GCS_MANIFEST_PATH)
    all_file_paths = manifest_df['file_path'].tolist()
    print(f"✅ Manifest loaded. Found {len(all_file_paths)} files to process.")
except Exception as e:
    sys.exit(f"❌ FATAL ERROR: Could not load manifest file from GCS. Details: {e}")

# --- Check for Progress ---
processed_files = set()
if os.path.exists(PROGRESS_FILE_PATH):
    with open(PROGRESS_FILE_PATH, 'r') as f:
        processed_files = set(f.read().splitlines())
    print(f"✅ Found progress file. Resuming. {len(processed_files)} files already processed.")

# --- Process Files in Chunks ---
files_to_process = [fp for fp in all_file_paths if fp not in processed_files]
print(f"\n--- Step B: Consolidating data from {len(files_to_process)} remaining files ---")
print("This is a very long process. Progress will be updated periodically.")
time.sleep(2)

# Determine starting chunk number based on existing files
chunk_num = len(glob.glob(os.path.join(DRIVE_OUTPUT_DIR, "spectrograms_*.npy")))
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

        all_labels.extend(labels.tolist())
        all_spectrograms.extend(spectrograms_flat.tolist())
        processed_files.add(f_path)

        if (i > 0 and (i + 1) % CHUNK_SIZE == 0) or ((i + 1) == len(files_to_process) and len(all_labels) > 0):
            print(f"\n     -> 💾 CHUNK COMPLETE: Saving chunk #{chunk_num} with {len(all_labels)} epochs...")

            chunk_data_path = os.path.join(DRIVE_OUTPUT_DIR, f"spectrograms_{chunk_num}.npy")
            chunk_labels_path = os.path.join(DRIVE_OUTPUT_DIR, f"labels_{chunk_num}.npy")

            np.save(chunk_data_path, np.array(all_spectrograms, dtype=np.float32))
            np.save(chunk_labels_path, np.array(all_labels, dtype=np.int64))

            with open(PROGRESS_FILE_PATH, 'w') as f:
                f.write("\\n".join(sorted(list(processed_files))))

            print(f"        ...✅ Chunk #{chunk_num} saved successfully.")

            all_spectrograms.clear()
            all_labels.clear()
            chunk_num += 1

    except Exception as e:
        print(f"\\n     -> ⚠️ WARNING: Could not process {os.path.basename(f_path)}. Skipping. Error: {e}")

print("\n\n--- Step C: Final verification ---")
final_chunks = glob.glob(os.path.join(DRIVE_OUTPUT_DIR, "spectrograms_*.npy"))
print(f"  -> Found {len(final_chunks)} saved data chunks in the output directory.")

print("\\n" + "="*80)
print("🎉 SCRIPT COMPLETE 🎉")
print("Your dataset is now consolidated into manageable chunks in your Google Drive.")
print("You can now use the final, simplified training script on a GPU runtime.")
print("="*80)
'''

# Write the script to a file
with open("run_consolidation.py", "w") as f:
    f.write(python_script_logic)

# Execute the script using the virtual environment's python, forcing the correct backend
!MPLBACKEND=Agg consolidate_env/bin/python run_consolidation.py

print("\n--- Script execution finished. ---")

# ==================== NEW CELL ====================

# ==============================================================================
# FINAL, HIGH-PERFORMANCE TRAINING SCRIPT (DEFINITIVE VERSION)
# This script uses a chunked, iterable dataset from Google Drive to guarantee
# a stable, memory-safe, and high-performance training run.
# ==============================================================================

import os
import sys
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

# --- 2. CREATE AND PROVISION THE ISOLATED TRAINING ENVIRONMENT ---
print("\n--- Step 2: Creating and provisioning the isolated training environment ---")
!pip install --upgrade -q virtualenv
!virtualenv train_env
!train_env/bin/pip install --upgrade -q pip "pytorch-lightning" "timm" "pandas>=2.0" "pyarrow>=15.0" "fsspec>=2023.6.0" gcsfs google-auth matplotlib seaborn scikit-learn
print("✅ All dependencies installed successfully into 'train_env'.")


# --- 3. CREATE AND RUN THE FULL TRAINING SCRIPT ---
print("\n--- Step 3: Preparing and executing the training logic in the isolated environment ---")

python_script_logic = r'''
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import timm
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import os
import sys
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

torch.set_float32_matmul_precision('medium')
print("✅ Libraries imported inside virtual environment.")

# --- All class and function definitions are here ---

def get_model(model_name='convnext_base', pretrained=True):
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained)
    original_conv = model.stem[0]
    new_first_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding, bias=(original_conv.bias is not None))
    with torch.no_grad():
        if original_conv.weight.shape[1] == 3:
            new_first_conv.weight[:, :] = original_conv.weight.clone().mean(dim=1, keepdim=True)
    model.stem[0] = new_first_conv
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, 5)
    return model

class SleepStageClassifierLightning(pl.LightningModule):
    def __init__(self, model_name, learning_rate, class_weights, epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name=self.hparams.model_name)
        self.train_accuracy = MulticlassAccuracy(num_classes=5)
        self.val_accuracy = MulticlassAccuracy(num_classes=5)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float) if class_weights else None)
    def forward(self, x): return self.model(x)
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

class ChunkedIterableDataset(IterableDataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        spectrogram_chunks = sorted(glob.glob(os.path.join(data_dir, "spectrograms_*.npy")))
        label_chunks = sorted(glob.glob(os.path.join(data_dir, "labels_*.npy")))

        split_idx = int(0.8 * len(spectrogram_chunks))
        if self.is_train:
            self.spectrogram_chunks = spectrogram_chunks[:split_idx]
            self.label_chunks = label_chunks[:split_idx]
        else:
            self.spectrogram_chunks = spectrogram_chunks[split_idx:]
            self.label_chunks = label_chunks[split_idx:]

        print(f"✅ {'Training' if is_train else 'Validation'} dataset initialized with {len(self.spectrogram_chunks)} chunks.")

    def __iter__(self):
        chunk_indices = list(range(len(self.spectrogram_chunks)))
        if self.is_train:
            random.shuffle(chunk_indices)

        for chunk_idx in chunk_indices:
            X_chunk = np.load(self.spectrogram_chunks[chunk_idx])
            y_chunk = np.load(self.label_chunks[chunk_idx])

            if self.is_train:
                indices = np.random.permutation(len(y_chunk))
                X_chunk = X_chunk[indices]
                y_chunk = y_chunk[indices]

            for i in range(len(y_chunk)):
                spectrogram_flat = X_chunk[i]
                label = y_chunk[i]
                spectrogram_2d = spectrogram_flat.reshape(1, 76, 60)
                yield torch.from_numpy(spectrogram_2d), torch.tensor(label, dtype=torch.long)

def generate_performance_report(ckpt_path, dataloader, device, save_dir, exp_name):
    # ... placeholder for brevity ...
    pass

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    CONSOLIDATED_DATA_DIR = "/content/drive/MyDrive/shhs_consolidated_data/"
    CLASS_WEIGHTS = [0.7, 6.5, 0.5, 1.5, 1.2]
    EPOCHS = 40
    BATCH_SIZE = 512
    LEARNING_RATE = 2e-5
    NUM_WORKERS = 2 # Safe to use multiple workers now
    DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/final_model_checkpoint/"

    train_dataset = ChunkedIterableDataset(CONSOLIDATED_DATA_DIR, is_train=True)
    val_dataset = ChunkedIterableDataset(CONSOLIDATED_DATA_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_convnext_base_consolidated_cwN1-{CLASS_WEIGHTS[1]}"

    model = SleepStageClassifierLightning('convnext_base', LEARNING_RATE, CLASS_WEIGHTS, EPOCHS)

    drive_log_dir = "/content/drive/MyDrive/sleep_logs/"
    os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

    csv_logger = CSVLogger(drive_log_dir, name=experiment_name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=DRIVE_CHECKPOINT_DIR, filename=f"best-model-{experiment_name}", save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')

    checkpoint_files = glob.glob(os.path.join(DRIVE_CHECKPOINT_DIR, "*.ckpt"))
    latest_checkpoint = None
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"\\n✅ Found checkpoint. Resuming from: {os.path.basename(latest_checkpoint)}")
    else:
        print("\\n  -> No checkpoint found. Starting a new training run.")

    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator="gpu", devices=1, logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="bf16-mixed", gradient_clip_val=1.0
    )

    print(f"\\n🚀🚀🚀 Starting/Resuming training for experiment: {experiment_name} 🚀🚀🚀")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=latest_checkpoint)
    print(f"\\n✅ Training complete!")

    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        print("Performance report generation placeholder")
    else:
        print("  -> No checkpoint was saved. Skipping final report.")
'''

# Write the script to a file
with open("run_training.py", "w") as f:
    f.write(python_script_logic)

# Execute the script
!MPLBACKEND=Agg train_env/bin/python run_training.py

print("\n--- Script execution finished. ---")

# ==================== NEW CELL ====================

# ==============================================================================
# 2025-09-18 FINAL, HIGH-PERFORMANCE TRAINING SCRIPT (DEFINITIVE VERSION)
# This script uses a chunked, iterable dataset from Google Drive to guarantee
# a stable, memory-safe, and high-performance training run.
# ==============================================================================

import os
import sys
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

# --- 2. CREATE AND PROVISION THE ISOLATED TRAINING ENVIRONMENT ---
print("\n--- Step 2: Creating and provisioning the isolated training environment ---")
!pip install --upgrade -q virtualenv
!virtualenv train_env
!train_env/bin/pip install --upgrade -q pip "pytorch-lightning" "timm" "pandas>=2.0" "pyarrow>=15.0" "fsspec>=2023.6.0" gcsfs google-auth matplotlib seaborn scikit-learn
print("✅ All dependencies installed successfully into 'train_env'.")


# --- 3. CREATE AND RUN THE FULL TRAINING SCRIPT ---
print("\n--- Step 3: Preparing and executing the training logic in the isolated environment ---")

python_script_logic = r'''
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import timm
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import os
import sys
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

torch.set_float32_matmul_precision('medium')
print("✅ Libraries imported inside virtual environment.")

# --- All class and function definitions are here ---

def get_model(model_name='convnext_base', pretrained=True):
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained)
    original_conv = model.stem[0]
    new_first_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding, bias=(original_conv.bias is not None))
    with torch.no_grad():
        if original_conv.weight.shape[1] == 3:
            new_first_conv.weight[:, :] = original_conv.weight.clone().mean(dim=1, keepdim=True)
    model.stem[0] = new_first_conv
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, 5)
    return model

class SleepStageClassifierLightning(pl.LightningModule):
    def __init__(self, model_name, learning_rate, class_weights, epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name=self.hparams.model_name)
        self.train_accuracy = MulticlassAccuracy(num_classes=5)
        self.val_accuracy = MulticlassAccuracy(num_classes=5)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float) if class_weights else None)
    def forward(self, x): return self.model(x)
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

class ChunkedIterableDataset(IterableDataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        spectrogram_chunks = sorted(glob.glob(os.path.join(data_dir, "spectrograms_*.npy")))
        label_chunks = sorted(glob.glob(os.path.join(data_dir, "labels_*.npy")))

        split_idx = int(0.8 * len(spectrogram_chunks))
        if self.is_train:
            self.spectrogram_chunks = spectrogram_chunks[:split_idx]
            self.label_chunks = label_chunks[:split_idx]
        else:
            self.spectrogram_chunks = spectrogram_chunks[split_idx:]
            self.label_chunks = label_chunks[split_idx:]

        print(f"✅ {'Training' if is_train else 'Validation'} dataset initialized with {len(self.spectrogram_chunks)} chunks.")

    def __iter__(self):
        chunk_indices = list(range(len(self.spectrogram_chunks)))
        if self.is_train:
            random.shuffle(chunk_indices)

        for chunk_idx in chunk_indices:
            X_chunk = np.load(self.spectrogram_chunks[chunk_idx])
            y_chunk = np.load(self.label_chunks[chunk_idx])

            if self.is_train:
                indices = np.random.permutation(len(y_chunk))
                X_chunk = X_chunk[indices]
                y_chunk = y_chunk[indices]

            for i in range(len(y_chunk)):
                spectrogram_flat = X_chunk[i]
                label = y_chunk[i]
                spectrogram_2d = spectrogram_flat.reshape(1, 76, 60)
                yield torch.from_numpy(spectrogram_2d), torch.tensor(label, dtype=torch.long)

def generate_performance_report(ckpt_path, dataloader, device, save_dir, exp_name):
    # ... placeholder for brevity ...
    pass

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    CONSOLIDATED_DATA_DIR = "/content/drive/MyDrive/shhs_consolidated_data/"
    CLASS_WEIGHTS = [0.7, 6.5, 0.5, 1.5, 1.2]
    EPOCHS = 40
    BATCH_SIZE = 512
    LEARNING_RATE = 2e-5
    NUM_WORKERS = 2 # Safe to use multiple workers now
    DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/final_model_checkpoint/"

    train_dataset = ChunkedIterableDataset(CONSOLIDATED_DATA_DIR, is_train=True)
    val_dataset = ChunkedIterableDataset(CONSOLIDATED_DATA_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_convnext_base_consolidated_cwN1-{CLASS_WEIGHTS[1]}"

    model = SleepStageClassifierLightning('convnext_base', LEARNING_RATE, CLASS_WEIGHTS, EPOCHS)

    drive_log_dir = "/content/drive/MyDrive/sleep_logs/"
    os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

    csv_logger = CSVLogger(drive_log_dir, name=experiment_name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=DRIVE_CHECKPOINT_DIR, filename=f"best-model-{experiment_name}", save_top_k=1, mode='min')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')

    checkpoint_files = glob.glob(os.path.join(DRIVE_CHECKPOINT_DIR, "*.ckpt"))
    latest_checkpoint = None
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"\\n✅ Found checkpoint. Resuming from: {os.path.basename(latest_checkpoint)}")
    else:
        print("\\n  -> No checkpoint found. Starting a new training run.")

    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator="gpu", devices=1, logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="bf16-mixed", gradient_clip_val=1.0
    )

    print(f"\\n🚀🚀🚀 Starting/Resuming training for experiment: {experiment_name} 🚀🚀🚀")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=latest_checkpoint)
    print(f"\\n✅ Training complete!")

    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        print("Performance report generation placeholder")
    else:
        print("  -> No checkpoint was saved. Skipping final report.")
'''

# Write the script to a file
with open("run_training.py", "w") as f:
    f.write(python_script_logic)

# Execute the script
!MPLBACKEND=Agg train_env/bin/python run_training.py

print("\n--- Script execution finished. ---")

# ==================== NEW CELL ====================

