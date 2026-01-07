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
!pip install --upgrade -q pytorch-lightning timm "pandas==2.2.2" "pyarrow==19.0.0" gcsfs "fsspec==2023.6.0" matplotlib seaborn scikit-learn
print("✅ Installation check complete.")


# ==============================================================================
# 3. IMPORTS AND INITIAL CONFIGURATION
# ==============================================================================
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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
import glob

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

torch.set_float32_matmul_precision('medium')
print("✅ Libraries imported and configuration set.")


# ==============================================================================
# 4. MODEL ARCHITECTURE DEFINITION
# ==============================================================================
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
        print(f"✅ ConvNeXT v2 Base model created and adapted for 1-channel input.")
    else:
        raise ValueError(f"Model '{model_name}' not supported in this script.")
    return model

print("✅ `get_model` function defined for ConvNeXT v2 Base.")


# ==============================================================================
# 5. PYTORCH LIGHTNING MODULE
# ==============================================================================
class SleepStageClassifierLightning(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-5, class_weights=None, epochs=40):
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
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy(y_pred_logits, y_true), on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=1e-7)
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
            if (i + 1) % 50 == 0 or i == total_files - 1 or i == 0:
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
# 7. VISUALIZATION AND REPORTING FUNCTIONS
# ==============================================================================
def plot_training_metrics(csv_path, save_dir, experiment_name):
    try:
        metrics_df = pd.read_csv(csv_path)
        epoch_metrics = metrics_df.dropna(subset=['epoch', 'train_loss_epoch', 'val_loss'])

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(epoch_metrics['epoch'], epoch_metrics['train_acc_epoch'], 'o-', label='Training Accuracy')
        ax1.plot(epoch_metrics['epoch'], epoch_metrics['val_acc'], 'o-', label='Validation Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Training & Validation Accuracy\n({experiment_name})')
        ax1.legend()

        ax2.plot(epoch_metrics['epoch'], epoch_metrics['train_loss_epoch'], 'o-', label='Training Loss')
        ax2.plot(epoch_metrics['epoch'], epoch_metrics['val_loss'], 'o-', label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'Training & Validation Loss\n({experiment_name})')
        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{experiment_name}_metrics_plot.png")
        plt.savefig(save_path, dpi=300)
        print(f"\n✅ Training metrics plot saved to: {save_path}")
        plt.show()
    except Exception as e:
        print(f"\nCould not generate training plot. Error: {e}")

def generate_performance_report(model_checkpoint_path, dataloader, device, save_dir, experiment_name):
    print("\n" + "="*80)
    print("Generating Final Performance Metrics and Visualizations...")
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
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    print("  -> Prediction complete.")

    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    target_names = [stage_map[i] for i in range(5)]

    print("\n--- Detailed Classification Report (Best Model) ---")
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(report)

    print("\n--- Generating Confusion Matrix Heatmap ---")
    conf_matrix_metric = MulticlassConfusionMatrix(num_classes=5)
    matrix = conf_matrix_metric(torch.tensor(all_preds), torch.tensor(all_labels)).numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix\n({experiment_name})', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    save_path = os.path.join(save_dir, f"{experiment_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix plot saved to: {save_path}")
    plt.show()
    print("="*80 + "\n")

print("✅ Visualization and reporting functions defined.")


# ==============================================================================
# 8. TRAINING EXECUTION
# ==============================================================================
print("\n--- Starting Model Training ---")

# --- ⚙️ USER CONFIGURATION ⚙️ ---
GCS_SHHS1_PATH = "gs://shhs-sleepedfx-data-bucket/shhs1_processed"
GCS_SHHS2_PATH = "gs://shhs-sleepedfx-data-bucket/shhs2_processed"
NUM_FILES_PER_SET = 1000

MODEL_TO_TEST = 'convnext_base'
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 2e-5
# --- MODIFICATION: Increase NUM_WORKERS to leverage more system RAM and speed up data loading ---
NUM_WORKERS = 2
CLASS_WEIGHTS = [0.7, 8.0, 0.5, 1.5, 1.2]

# --- Get file paths ---
print(f"Listing {NUM_FILES_PER_SET} files from {GCS_SHHS1_PATH}...")
shhs1_file_paths = !gsutil ls {GCS_SHHS1_PATH}/*.parquet | head -n {NUM_FILES_PER_SET}
shhs1_file_paths = shhs1_file_paths.nlstr.split()
print(f"Listing {NUM_FILES_PER_SET} files from {GCS_SHHS2_PATH}...")
shhs2_file_paths = !gsutil ls {GCS_SHHS2_PATH}/*.parquet | head -n {NUM_FILES_PER_SET}
shhs2_file_paths = shhs2_file_paths.nlstr.split()

specific_shhs_file_paths = [path for path in (shhs1_file_paths + shhs2_file_paths) if path.startswith("gs://")]
print(f"✅ Found {len(specific_shhs_file_paths)} valid GCS file paths.")

# --- Main Experiment ---
if not specific_shhs_file_paths:
    print("\nERROR: No valid .parquet files found.")
else:
    full_dataset = CombinedDataset(specific_shhs_file_paths)

    if len(full_dataset) > 1:
        print("\nSplitting dataset into training and validation (80/20)...")
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"✅ Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation.")

        print("\n--- Starting Pre-Training Data Distribution Analysis... ---")
        stage_map_display = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
        train_counts = Counter(full_dataset[i][1].item() for i in train_dataset.indices)
        val_counts = Counter(full_dataset[i][1].item() for i in val_dataset.indices)

        print("✅ Analysis Complete.")
        print(f"{'Set':<12} | {'Stage':<6} | {'Count':>10} | {'Percentage':>12}")
        print("-" * 50)
        train_total = len(train_dataset)
        for i in range(5):
            count = train_counts.get(i, 0)
            print(f"{'Training':<12} | {stage_map_display[i]:<6} | {count:>10} | {(count/train_total*100):>11.2f}%")
        print("-" * 50)
        val_total = len(val_dataset)
        for i in range(5):
            count = val_counts.get(i, 0)
            print(f"{'Validation':<12} | {stage_map_display[i]:<6} | {count:>10} | {(count/val_total*100):>11.2f}%")
        print("-" * 50)

        print("\nCreating DataLoaders...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print("✅ DataLoaders created.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = (
            f"{timestamp}_{MODEL_TO_TEST}_{len(specific_shhs_file_paths)}files_"
            f"lr{LEARNING_RATE}_cwN1-{CLASS_WEIGHTS[1]}_workers{NUM_WORKERS}"
        )

        print(f"\n{'='*20} CONFIGURING EXPERIMENT: {experiment_name} {'='*20}")
        model = SleepStageClassifierLightning(MODEL_TO_TEST, LEARNING_RATE, CLASS_WEIGHTS, epochs=EPOCHS)

        drive_log_dir = "/content/drive/MyDrive/sleep_logs/"
        drive_checkpoint_dir = "/content/drive/MyDrive/final_model_checkpoint/"

        os.makedirs(drive_log_dir, exist_ok=True)
        os.makedirs(drive_checkpoint_dir, exist_ok=True)
        print(f"  -> Log directory: {drive_log_dir}")
        print(f"  -> Checkpoint & Plot directory: {drive_checkpoint_dir}")

        csv_logger = CSVLogger(drive_log_dir, name=experiment_name)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=drive_checkpoint_dir, filename=f"best-model-{experiment_name}", save_top_k=1, mode='min')
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')

        print("\n--- Checking for existing checkpoints to resume training ---")
        checkpoint_files = glob.glob(os.path.join(drive_checkpoint_dir, "*.ckpt"))
        latest_checkpoint = None
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            print(f"✅ Found checkpoint. Resuming training from: {os.path.basename(latest_checkpoint)}")
        else:
            print("  -> No checkpoint found. Starting a new training run.")

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

        print(f"\n🚀🚀🚀 Starting training... 🚀🚀🚀")
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=latest_checkpoint
        )
        print(f"\n✅ Training complete!")

        if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
            print(f"  -> Best model saved at: {checkpoint_callback.best_model_path}")
            generate_performance_report(checkpoint_callback.best_model_path, val_loader, model.device, drive_checkpoint_dir, experiment_name)

            log_dir = csv_logger.log_dir
            metrics_file = os.path.join(log_dir, 'metrics.csv')
            if os.path.exists(metrics_file):
                plot_training_metrics(metrics_file, drive_checkpoint_dir, experiment_name)
        else:
            print("  -> No checkpoint was saved or found. Skipping final report.")

        print(f"{'='*20} FINISHED EXPERIMENT: {experiment_name} {'='*20}")
    else:
        print("Dataset is too small to split. Aborting.")

print("\n--- End of Script ---")

# ==================== NEW CELL ====================

