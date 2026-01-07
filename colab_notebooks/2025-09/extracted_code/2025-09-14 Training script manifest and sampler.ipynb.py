# ==============================================================================
# SCRIPT TO TRAIN THE MODEL (DEFINITIVE, WITH WEIGHTED SAMPLER)
# This script uses an isolated environment, a single cached worker, and a
# WeightedRandomSampler to provide the most stable and balanced training.
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
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
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

class CombinedDataset(Dataset):
    def __init__(self, manifest_path, num_files=None):
        manifest_df = pd.read_csv(manifest_path)
        if num_files:
            manifest_df = manifest_df.head(num_files)
        self.file_paths = manifest_df['file_path'].tolist()
        self.cumulative_epochs = np.cumsum(manifest_df['epoch_count'].values)
        self.total_epochs = self.cumulative_epochs[-1]
        self._cache = {}
        print(f"✅ Dataset initialized from manifest. Found {self.total_epochs} epochs across {len(self.file_paths)} files.")
    def __len__(self): return self.total_epochs

    def get_labels_for_sampler(self):
        """A new, truly efficient method to get all labels for the sampler."""
        all_labels = []
        print("  -> Efficiently gathering all labels for sampler weighting...")
        for i, file_path in enumerate(self.file_paths):
            if (i + 1) % 200 == 0 or i == len(self.file_paths) - 1:
                 print(f"\r     ...processing file {i+1}/{len(self.file_paths)}", end="")
            # --- MODIFICATION: Only read the 'label' column ---
            df_labels = pd.read_parquet(file_path, columns=['label'])
            labels = df_labels['label'][df_labels['label'].isin([0, 1, 2, 3, 4])].tolist()
            all_labels.extend(labels)
        print("\n     ...done.")
        return all_labels

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_epochs, idx, side='right')
        local_idx = idx - (self.cumulative_epochs[file_idx - 1] if file_idx > 0 else 0)
        file_path = self.file_paths[file_idx]
        if file_path not in self._cache:
            self._cache[file_path] = pd.read_parquet(file_path)[lambda df: df['label'].isin([0, 1, 2, 3, 4])].reset_index(drop=True)
        row = self._cache[file_path].iloc[local_idx]
        label = np.int64(row['label'])
        spectrogram_flat = row.drop('label').values.astype(np.float32)
        spectrogram_2d = spectrogram_flat.reshape(1, 76, 60)
        return torch.from_numpy(spectrogram_2d), torch.tensor(label)

def generate_performance_report(ckpt_path, dataloader, device, save_dir, exp_name):
    pass # Placeholder

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    GCS_MANIFEST_PATH = "gs://shhs-sleepedfx-data-bucket/metadata/shhs_dataset_manifest.csv"
    NUM_FILES_TO_USE = 2000
    CLASS_WEIGHTS = [0.7, 6.5, 0.5, 1.5, 1.2]
    EPOCHS = 40
    BATCH_SIZE = 256
    LEARNING_RATE = 2e-5
    NUM_WORKERS = 0
    DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/final_model_checkpoint/"

    full_dataset = CombinedDataset(GCS_MANIFEST_PATH, num_files=NUM_FILES_TO_USE)
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset_subset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print("\n--- Creating a WeightedRandomSampler to address class imbalance ---")

    # Efficiently get all labels from the full dataset, then filter for the training subset
    all_labels_in_dataset = full_dataset.get_labels_for_sampler()
    train_subset_labels = [all_labels_in_dataset[i] for i in train_dataset_subset.indices]

    class_counts = Counter(train_subset_labels)
    print(f"  -> Training class distribution: {class_counts}")
    class_weights_for_sampler = {i: 1.0 / count for i, count in class_counts.items()}
    sample_weights = [class_weights_for_sampler[label] for label in train_subset_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("✅ Sampler created successfully.")

    # Use the sampler in the training DataLoader. shuffle MUST be False when using a sampler.
    train_loader = DataLoader(train_dataset_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_convnext_base_{NUM_FILES_TO_USE}files_Sampler_cwN1-{CLASS_WEIGHTS[1]}"

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
        print(f"✅ Found checkpoint. Resuming from: {os.path.basename(latest_checkpoint)}")
    else:
        print("  -> No checkpoint found. Starting a new training run.")

    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator="gpu", devices=1, logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="bf16-mixed", gradient_clip_val=1.0
    )

    print(f"\n🚀🚀🚀 Starting/Resuming training with sampler for experiment: {experiment_name} 🚀🚀🚀")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=latest_checkpoint)
    print(f"\n✅ Training complete!")

    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        print("Performance report generation placeholder")
    else:
        print("  -> No checkpoint was saved.")
'''

# Write the script to a file
with open("run_training.py", "w") as f:
    f.write(python_script_logic)

# Execute the script
!MPLBACKEND=Agg train_env/bin/python run_training.py

print("\n--- Script execution finished. ---")

# ==================== NEW CELL ====================

