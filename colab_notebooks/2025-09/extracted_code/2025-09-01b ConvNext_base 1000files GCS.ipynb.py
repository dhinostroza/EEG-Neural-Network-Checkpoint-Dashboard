# Original code fubared by tiny
# ==============================================================================
# 1. SETUP: AUTHENTICATION AND DRIVE MOUNT
# ==============================================================================
from google.colab import auth
from google.colab import drive
import os

print("Authenticating to Google Cloud...")
auth.authenticate_user()
print("✅ Authentication successful.")

print("\nMounting Google Drive...")
drive.mount('/content/drive', force_remount=True)
print("✅ Google Drive mounted.")


# ==============================================================================
# 2. DEPENDENCY INSTALLATION
# ==============================================================================
print("\nEnsuring PyTorch Lightning and other libraries are installed...")
# Pinned fsspec to a compatible version to resolve the dependency conflict
!pip install --upgrade -q pytorch-lightning timm "pandas==2.2.2" "pyarrow==19.0.0" gcsfs "fsspec==2025.3.0"
print("✅ Installation check complete.")

# ==============================================================================
# 3. IMPORTS AND INITIAL CONFIGURATION
# ==============================================================================
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

torch.set_float32_matmul_precision('medium')
print("✅ Libraries imported and configuration set.")

# ==============================================================================
# 4. MODEL ARCHITECTURE DEFINITION (SINGLE-MODEL)
# ==============================================================================
def get_model(model_name='convnext_tiny', num_classes=5, pretrained=True):
    """Creates a ConvNeXT Tiny model adapted for sleep stage classification."""
    if model_name == 'convnext_tiny':
        model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=pretrained)
        original_conv = model.stem[0]
        new_first_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding, bias=(original_conv.bias is not None))
        with torch.no_grad():
            if original_conv.weight.shape[1] == 3:
                new_first_conv.weight[:, :] = original_conv.weight.clone().mean(dim=1, keepdim=True)
        model.stem[0] = new_first_conv
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, num_classes)
        print(f"✅ ConvNeXT Tiny model created.")
    else:
        raise ValueError(f"Model '{model_name}' not supported in this script.")
    return model

print("✅ `get_model` function defined for ConvNeXT Tiny.")


# ==============================================================================
# 5. PYTORCH LIGHTNING MODULE
# ==============================================================================
class SleepStageClassifierLightning(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name=self.hparams.model_name, num_classes=5, pretrained=True)
        self.train_accuracy = MulticlassAccuracy(num_classes=5)
        self.val_accuracy = MulticlassAccuracy(num_classes=5)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred_logits = self(x)
        loss = self.loss_fn(y_pred_logits, y_true)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy(y_pred_logits, y_true), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred_logits = self(x)
        loss = self.loss_fn(y_pred_logits, y_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_accuracy(y_pred_logits, y_true), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3), 'monitor': 'val_loss'}
        return [optimizer], [scheduler]

print("✅ `SleepStageClassifierLightning` module defined.")


# ==============================================================================
# 6. CUSTOM DATASET DEFINITION (MORE VERBOSE)
# ==============================================================================
class CombinedDataset(Dataset):
    def __init__(self, file_paths_chunk):
        print(f"Initializing dataset with {len(file_paths_chunk)} files from GCS...")
        self.file_paths = file_paths_chunk
        self.epochs_per_file = []

        total_files = len(self.file_paths)
        for i, f_path in enumerate(self.file_paths):
            if (i + 1) % 50 == 0 or i == total_files - 1:
                print(f"  -> [{i+1}/{total_files}] Reading header from: {os.path.basename(f_path)}")
            try:
                df_labels = pd.read_parquet(f_path, columns=['label'])
                num_valid = df_labels['label'].isin([0, 1, 2, 3, 4]).sum()
                self.epochs_per_file.append(num_valid)
            except Exception as e:
                print(f"  -> WARNING: Could not process {os.path.basename(f_path)}. Skipping. Error: {e}")
                self.epochs_per_file.append(0)

        self.cumulative_epochs = np.cumsum(self.epochs_per_file)
        self.total_epochs = self.cumulative_epochs[-1] if self.cumulative_epochs.size > 0 else 0
        self._cache = {}
        print(f"✅ Dataset initialized. Found a total of {self.total_epochs} valid epochs.")

    def __len__(self):
        return self.total_epochs

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_epochs, idx, side='right')
        local_idx = idx - (self.cumulative_epochs[file_idx - 1] if file_idx > 0 else 0)
        file_path = self.file_paths[file_idx]
        if file_path not in self._cache:
            df = pd.read_parquet(file_path)
            self._cache[file_path] = df[df['label'].isin([0, 1, 2, 3, 4])].reset_index(drop=True)
        row = self._cache[file_path].iloc[local_idx]
        label = np.int64(row['label'])
        spectrogram_flat = row.drop('label').values.astype(np.float32)
        mean, std = spectrogram_flat.mean(), spectrogram_flat.std()
        spectrogram_normalized = (spectrogram_flat - mean) / (std + 1e-6)
        spectrogram_2d = spectrogram_normalized.reshape(1, 76, 60)
        return torch.from_numpy(spectrogram_2d), torch.tensor(label)

print("✅ `CombinedDataset` class defined.")


# ==============================================================================
# 7. PERFORMANCE REPORTING FUNCTION
# ==============================================================================
def generate_performance_report(model_checkpoint_path, dataloader, device):
    """Loads the best model and generates a detailed classification report."""
    # --- MODIFIED: Removed the redundant mount command ---

    print("\n" + "="*80)
    print("Generating Final Performance Metrics on the Validation Set...")
    model = SleepStageClassifierLightning.load_from_checkpoint(model_checkpoint_path)
    model.to(device)
    model.eval()

    print("  -> Predicting on validation data...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x.to(device))
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    print("  -> Prediction complete.")

    num_classes = 5
    metrics = {
        "Precision": MulticlassPrecision(num_classes=num_classes, average=None),
        "Recall": MulticlassRecall(num_classes=num_classes, average=None),
        "F1-Score": MulticlassF1Score(num_classes=num_classes, average=None)
    }
    results = {name: metric(all_preds, all_labels) for name, metric in metrics.items()}
    accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')(all_preds, all_labels)
    support = torch.bincount(all_labels, minlength=num_classes)

    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    print("\n--- Sleep Stage Classification Report (Best Model) ---")
    print(f"{'Stage':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 65)
    for i in range(num_classes):
        print(f"{stage_map[i]:<10} | {results['Precision'][i]:<10.4f} | {results['Recall'][i]:<10.4f} | {results['F1-Score'][i]:<10.4f} | {support[i]:<10}")
    print("-" * 65)
    print(f"\nOverall Accuracy: {accuracy.item():.4f}")

    print("\n--- Confusion Matrix ---")
    conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
    matrix = conf_matrix(all_preds, all_labels)
    print(matrix)
    print("="*80 + "\n")

print("✅ `generate_performance_report` function defined.")


# ==============================================================================
# 8. TRAINING EXECUTION
# ==============================================================================
print("\n--- Starting Model Training ---")

# --- ⚙️ USER CONFIGURATION ⚙️ ---
GCS_SHHS1_PATH = "gs://shhs-sleepedfx-data-bucket/shhs1_processed"
GCS_SHHS2_PATH = "gs://shhs-sleepedfx-data-bucket/shhs2_processed"
NUM_FILES_PER_SET = 500

MODEL_TO_TEST = 'convnext_base'
EPOCHS = 40
BATCH_SIZE = 256
NUM_WORKERS = 0
CLASS_WEIGHTS = [0.7, 5.0, 0.5, 1.5, 1.2]
LEARNING_RATE = 5e-5

# --- Get file paths from each specified GCS folder ---
print(f"Listing {NUM_FILES_PER_SET} files from {GCS_SHHS1_PATH}...")
shhs1_files_str = !gsutil ls {GCS_SHHS1_PATH}/*.parquet | head -n {NUM_FILES_PER_SET}
shhs1_file_paths = shhs1_files_str.nlstr.split()

print(f"Listing {NUM_FILES_PER_SET} files from {GCS_SHHS2_PATH}...")
shhs2_files_str = !gsutil ls {GCS_SHHS2_PATH}/*.parquet | head -n {NUM_FILES_PER_SET}
shhs2_file_paths = shhs2_files_str.nlstr.split()

raw_file_paths = shhs1_file_paths + shhs2_file_paths
specific_shhs_file_paths = [path for path in raw_file_paths if path.startswith("gs://")]
print(f"✅ Found {len(specific_shhs_file_paths)} valid GCS file paths.")


# --- Main Experiment ---
if not specific_shhs_file_paths:
     print("\nERROR: No valid .parquet files found. Check GCS paths and permissions. Aborting.")
else:
    full_dataset = CombinedDataset(specific_shhs_file_paths)

    if len(full_dataset) > 1:
        print("\nSplitting the dataset into training and validation sets...")
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"✅ Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

        print("\nCreating DataLoaders...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, persistent_workers=False)
        print("✅ DataLoaders created.")

        print(f"\n{'='*20} CONFIGURING EXPERIMENT FOR MODEL: {MODEL_TO_TEST.upper()} {'='*20}")
        model = SleepStageClassifierLightning(MODEL_TO_TEST, LEARNING_RATE, CLASS_WEIGHTS)

        drive_log_dir = "/content/drive/MyDrive/sleep_logs/"
        drive_checkpoint_dir = "/content/drive/MyDrive/final_model_checkpoint/"
        experiment_name = f"{MODEL_TO_TEST}_gcs_500_file_test_tuned"

        # --- NEW: Verify that the output directories exist on Google Drive ---
        print("\nVerifying output directories on Google Drive...")
        os.makedirs(drive_log_dir, exist_ok=True)
        os.makedirs(drive_checkpoint_dir, exist_ok=True)
        print(f"  -> Log directory is ready: {drive_log_dir}")
        print(f"  -> Checkpoint directory is ready: {drive_checkpoint_dir}")

        print(f"  -> Logger: Saving CSV logs to {drive_log_dir}{experiment_name}")
        csv_logger = CSVLogger(drive_log_dir, name=experiment_name)

        print(f"  -> Checkpoint: Saving best model to {drive_checkpoint_dir}")
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=drive_checkpoint_dir, filename=f"sleep-stage-{experiment_name}-{{epoch:02d}}-{{val_loss:.4f}}", save_top_k=1, mode='min')

        print("  -> Early Stopping: Patience set to 7 epochs monitoring 'val_loss'")
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=7, verbose=True, mode='min')

        print("\nConfiguring PyTorch Lightning Trainer...")
        trainer = pl.Trainer(
            max_epochs=EPOCHS, accelerator="gpu", devices=1, logger=csv_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            precision="bf16-mixed", gradient_clip_val=1.0
        )
        print("✅ Trainer configured.")

        print(f"\n🚀🚀🚀 Starting model training for {MODEL_TO_TEST.upper()}... 🚀🚀🚀")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"\n✅ Training complete for {MODEL_TO_TEST.upper()}!")

        if checkpoint_callback.best_model_path:
            print(f"  -> Best model saved at: {checkpoint_callback.best_model_path}")
            generate_performance_report(checkpoint_callback.best_model_path, val_loader, model.device)
        else:
            print("  -> No checkpoint was saved. Skipping performance report.")
        print(f"{'='*20} FINISHED EXPERIMENT FOR MODEL: {MODEL_TO_TEST.upper()} {'='*20}")
    else:
        print("Dataset is too small to split. Aborting.")

print("\n--- Model Training Complete ---")

# ==================== NEW CELL ====================

# ==============================================================================
# 1. SETUP: AUTHENTICATION AND DRIVE MOUNT
# ==============================================================================
from google.colab import auth
from google.colab import drive
import os

print("Authenticating to Google Cloud...")
auth.authenticate_user()
print("✅ Authentication successful.")

print("\nMounting Google Drive...")
drive.mount('/content/drive', force_remount=True)
print("✅ Google Drive mounted.")


# ==============================================================================
# 2. DEPENDENCY INSTALLATION
# ==============================================================================
print("\nEnsuring PyTorch Lightning and other libraries are installed...")
# Pinned fsspec to a compatible version to resolve potential dependency conflicts
!pip install --upgrade -q pytorch-lightning timm "pandas==2.2.2" "pyarrow==19.0.0" gcsfs "fsspec==2023.6.0"
print("✅ Installation check complete.")


# ==============================================================================
# 3. IMPORTS AND INITIAL CONFIGURATION
# ==============================================================================
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

# Use 'medium' precision for matmul operations to leverage Tensor Cores on supported GPUs
torch.set_float32_matmul_precision('medium')
print("✅ Libraries imported and configuration set.")


# ==============================================================================
# 4. MODEL ARCHITECTURE DEFINITION (MODIFIED FOR CONVNEXT-V2-BASE)
# ==============================================================================
def get_model(model_name='convnext_base', num_classes=5, pretrained=True):
    """
    Creates a ConvNeXT v2 Base model adapted for single-channel input (spectrograms)
    for sleep stage classification.
    """
    if model_name == 'convnext_base':
        model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained)

        # Adapt the first convolutional layer from 3 channels (RGB) to 1 channel (grayscale spectrogram)
        original_conv = model.stem[0]
        new_first_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )

        # Initialize new layer's weights by averaging the original weights across the channel dimension
        with torch.no_grad():
            if original_conv.weight.shape[1] == 3: # Check if original input channels is 3
                new_first_conv.weight[:, :] = original_conv.weight.clone().mean(dim=1, keepdim=True)

        model.stem[0] = new_first_conv

        # Replace the final classification layer to match the number of sleep stages
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, num_classes)
        print(f"✅ ConvNeXT v2 Base model created and adapted for 1-channel input.")
    else:
        raise ValueError(f"Model '{model_name}' not supported in this script. Only 'convnext_base' is configured.")
    return model

print("✅ `get_model` function defined for ConvNeXT v2 Base.")


# ==============================================================================
# 5. PYTORCH LIGHTNING MODULE
# ==============================================================================
class SleepStageClassifierLightning(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-5, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name=self.hparams.model_name, num_classes=5, pretrained=True)
        self.train_accuracy = MulticlassAccuracy(num_classes=5)
        self.val_accuracy = MulticlassAccuracy(num_classes=5)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred_logits = self(x)
        loss = self.loss_fn(y_pred_logits, y_true)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy(y_pred_logits, y_true), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred_logits = self(x)
        loss = self.loss_fn(y_pred_logits, y_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_accuracy(y_pred_logits, y_true), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

print("✅ `SleepStageClassifierLightning` module defined.")


# ==============================================================================
# 6. CUSTOM DATASET DEFINITION
# ==============================================================================
class CombinedDataset(Dataset):
    def __init__(self, file_paths_chunk):
        print(f"Initializing dataset with {len(file_paths_chunk)} files from GCS...")
        self.file_paths = file_paths_chunk
        self.epochs_per_file = []

        total_files = len(self.file_paths)
        for i, f_path in enumerate(self.file_paths):
            if (i + 1) % 50 == 0 or i == total_files - 1:
                print(f"  -> [{i+1}/{total_files}] Reading header from: {os.path.basename(f_path)}")
            try:
                df_labels = pd.read_parquet(f_path, columns=['label'])
                # Count only epochs with valid sleep stage labels (0-4)
                num_valid = df_labels['label'].isin([0, 1, 2, 3, 4]).sum()
                self.epochs_per_file.append(num_valid)
            except Exception as e:
                print(f"  -> WARNING: Could not process {os.path.basename(f_path)}. Skipping. Error: {e}")
                self.epochs_per_file.append(0)

        self.cumulative_epochs = np.cumsum(self.epochs_per_file)
        self.total_epochs = self.cumulative_epochs[-1] if self.cumulative_epochs.size > 0 else 0
        self._cache = {}
        print(f"✅ Dataset initialized. Found a total of {self.total_epochs} valid epochs.")

    def __len__(self):
        return self.total_epochs

    def __getitem__(self, idx):
        # Find which file this index belongs to
        file_idx = np.searchsorted(self.cumulative_epochs, idx, side='right')
        # Find the local index within that file
        local_idx = idx - (self.cumulative_epochs[file_idx - 1] if file_idx > 0 else 0)

        file_path = self.file_paths[file_idx]

        # Load and cache file if not already in memory
        if file_path not in self._cache:
            df = pd.read_parquet(file_path)
            # Filter for valid labels and reset index for consistent local indexing
            self._cache[file_path] = df[df['label'].isin([0, 1, 2, 3, 4])].reset_index(drop=True)

        row = self._cache[file_path].iloc[local_idx]

        label = np.int64(row['label'])
        spectrogram_flat = row.drop('label').values.astype(np.float32)

        # Normalize each spectrogram individually
        mean, std = spectrogram_flat.mean(), spectrogram_flat.std()
        spectrogram_normalized = (spectrogram_flat - mean) / (std + 1e-6) # Add epsilon to avoid division by zero

        # Reshape to [channels, height, width] format
        spectrogram_2d = spectrogram_normalized.reshape(1, 76, 60)

        return torch.from_numpy(spectrogram_2d), torch.tensor(label)

print("✅ `CombinedDataset` class defined.")


# ==============================================================================
# 7. PERFORMANCE REPORTING FUNCTION
# ==============================================================================
def generate_performance_report(model_checkpoint_path, dataloader, device):
    """Loads the best model and generates a detailed classification report."""
    print("\n" + "="*80)
    print("Generating Final Performance Metrics on the Validation Set...")
    model = SleepStageClassifierLightning.load_from_checkpoint(model_checkpoint_path)
    model.to(device)
    model.eval()

    print("  -> Predicting on validation data...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x.to(device))
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    print("  -> Prediction complete.")

    num_classes = 5
    metrics = {
        "Precision": MulticlassPrecision(num_classes=num_classes, average=None),
        "Recall": MulticlassRecall(num_classes=num_classes, average=None),
        "F1-Score": MulticlassF1Score(num_classes=num_classes, average=None)
    }
    results = {name: metric(all_preds, all_labels) for name, metric in metrics.items()}
    accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')(all_preds, all_labels)
    support = torch.bincount(all_labels, minlength=num_classes)

    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    print("\n--- Sleep Stage Classification Report (Best Model) ---")
    print(f"{'Stage':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    print("-" * 65)
    for i in range(num_classes):
        print(f"{stage_map[i]:<10} | {results['Precision'][i]:<10.4f} | {results['Recall'][i]:<10.4f} | {results['F1-Score'][i]:<10.4f} | {support[i]:<10}")
    print("-" * 65)
    print(f"\nOverall Accuracy: {accuracy.item():.4f}")

    print("\n--- Confusion Matrix ---")
    conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
    matrix = conf_matrix(all_preds, all_labels)
    print(matrix.cpu().numpy()) # Print as numpy array for better formatting
    print("="*80 + "\n")

print("✅ `generate_performance_report` function defined.")


# ==============================================================================
# 8. TRAINING EXECUTION
# ==============================================================================
print("\n--- Starting Model Training ---")

# --- ⚙️ USER CONFIGURATION ⚙️ ---
GCS_SHHS1_PATH = "gs://shhs-sleepedfx-data-bucket/shhs1_processed"
GCS_SHHS2_PATH = "gs://shhs-sleepedfx-data-bucket/shhs2_processed"
NUM_FILES_PER_SET = 500

MODEL_TO_TEST = 'convnext_base'
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 2e-5

# --- MODIFICATION: Set NUM_WORKERS to 0 to prevent RAM exhaustion ---
# This forces data loading to happen in the main process, avoiding memory duplication.
NUM_WORKERS = 0
CLASS_WEIGHTS = [0.7, 5.0, 0.5, 1.5, 1.2]


# --- Get file paths from each specified GCS folder ---
print(f"Listing {NUM_FILES_PER_SET} files from {GCS_SHHS1_PATH}...")
shhs1_files_str = !gsutil ls {GCS_SHHS1_PATH}/*.parquet | head -n {NUM_FILES_PER_SET}
shhs1_file_paths = shhs1_files_str.nlstr.split()

print(f"Listing {NUM_FILES_PER_SET} files from {GCS_SHHS2_PATH}...")
shhs2_files_str = !gsutil ls {GCS_SHHS2_PATH}/*.parquet | head -n {NUM_FILES_PER_SET}
shhs2_file_paths = shhs2_files_str.nlstr.split()

raw_file_paths = shhs1_file_paths + shhs2_file_paths
specific_shhs_file_paths = [path for path in raw_file_paths if path.startswith("gs://")]
print(f"✅ Found {len(specific_shhs_file_paths)} valid GCS file paths.")


# --- Main Experiment ---
if not specific_shhs_file_paths:
    print("\nERROR: No valid .parquet files found. Check GCS paths and permissions. Aborting.")
else:
    full_dataset = CombinedDataset(specific_shhs_file_paths)

    if len(full_dataset) > 1:
        print("\nSplitting the dataset into training and validation sets (80/20)...")
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"✅ Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

        print("\nCreating DataLoaders...")
        # Note: persistent_workers is irrelevant if num_workers is 0
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print("✅ DataLoaders created.")

        print(f"\n{'='*20} CONFIGURING EXPERIMENT FOR MODEL: {MODEL_TO_TEST.upper()} {'='*20}")
        model = SleepStageClassifierLightning(MODEL_TO_TEST, LEARNING_RATE, CLASS_WEIGHTS)

        drive_log_dir = "/content/drive/MyDrive/sleep_logs/"
        drive_checkpoint_dir = "/content/drive/MyDrive/final_model_checkpoint/"
        experiment_name = f"{MODEL_TO_TEST}_gcs_{NUM_FILES_PER_SET*2}_files_tuned_lr_{LEARNING_RATE}"

        print("\nVerifying output directories on Google Drive...")
        os.makedirs(drive_log_dir, exist_ok=True)
        os.makedirs(drive_checkpoint_dir, exist_ok=True)
        print(f"  -> Log directory is ready: {drive_log_dir}")
        print(f"  -> Checkpoint directory is ready: {drive_checkpoint_dir}")

        print(f"  -> Logger: Saving CSV logs to {drive_log_dir}{experiment_name}")
        csv_logger = CSVLogger(drive_log_dir, name=experiment_name)

        print(f"  -> Checkpoint: Saving best model to {drive_checkpoint_dir}")
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=drive_checkpoint_dir,
            filename=f"sleep-stage-{experiment_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=1,
            mode='min'
        )

        print("  -> Early Stopping: Patience set to 7 epochs monitoring 'val_loss'")
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=7, verbose=True, mode='min')

        print("\nConfiguring PyTorch Lightning Trainer...")
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=1,
            logger=csv_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            precision="bf16-mixed",
            gradient_clip_val=1.0
        )
        print("✅ Trainer configured.")

        print(f"\n🚀🚀🚀 Starting model training for {MODEL_TO_TEST.upper()}... 🚀🚀🚀")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print(f"\n✅ Training complete for {MODEL_TO_TEST.upper()}!")

        if checkpoint_callback.best_model_path:
            print(f"  -> Best model saved at: {checkpoint_callback.best_model_path}")
            generate_performance_report(checkpoint_callback.best_model_path, val_loader, model.device)
        else:
            print("  -> No checkpoint was saved. Skipping performance report.")

        print(f"{'='*20} FINISHED EXPERIMENT FOR MODEL: {MODEL_TO_TEST.upper()} {'='*20}")
    else:
        print("Dataset is too small to split. Aborting.")

print("\n--- End of Script ---")

# ==================== NEW CELL ====================

