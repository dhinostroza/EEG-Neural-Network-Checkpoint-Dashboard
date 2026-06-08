import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import time
import os
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_recall_fscore_support

from models import EEGSNet
from data_loader import get_data_splits, SleepDataset

# Configuration
DATA_FILE = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/SleepEDFX8_processed_parallel.mat"
OUTPUT_DIR = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/python_results" # Saving alongside matlab
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32 # Reduced for sequence memory
SEQ_LEN = 5 # Li 2022 Context Window
EPOCHS = 100 
LR = 0.001
PATIENCE = 10 

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if alpha > 0:
        index = torch.randperm(batch_size).to(x.device)
        
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_fold(fold_idx, train_loader, val_loader, device, previous_results, output_dir, total_folds, class_weights=None):
    model = EEGSNet(lstm_hidden=128).to(device)
    # Improvement for Li 2022: Added Weight Decay 1e-4 for regularization
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Apply Class Weights if provided
    # Improvement for Li 2022: Added Label Smoothing 0.1 to prevent overconfidence
    # Apply Class Weights if provided
    # Improvement for Li 2022: Added Label Smoothing 0.1 to prevent overconfidence
    
    # v1.3: We use Sampling, so we DO NOT use Class Weights in Loss (Avoid Double Penalty)
    # But we keep Label Smoothing.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    train_losses_step = []
    train_accs_step = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"--- Fold {fold_idx+1} Training ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            # X shape: (B, Seq, C, F, T) -> Correctly handled by models.py
            # y shape: (B, Seq)
            
            optimizer.zero_grad()
            
            # mixup_data mixes along dim 0 (Batch). This preserves sequence structure within a sample.
            # y is (B, Seq), so y_a and y_b will be (B, Seq).
            inputs, targets_a, targets_b, lam = mixup_data(X, y, alpha=0.3, use_cuda=device.type=='cuda' or device.type=='mps')
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            
            outputs, aux_outputs = model(inputs)
            # outputs: (B, 5) -> Main Class (Last Step)
            # aux_outputs: (B, Seq, 5) -> Aux Class (All Steps)
            
            # Loss Calculation with Mixup
            
            # 1. Main Loss (Last Step)
            y_main_a = targets_a[:, -1]
            y_main_b = targets_b[:, -1]
            
            # Define original ground truth for metrics (Comparison against true label)
            y_main = y[:, -1]
            
            loss_main = mixup_criterion(criterion, outputs, y_main_a, y_main_b, lam)
            
            # 2. Aux Loss (All Steps)
            # Reshape for dense supervision
            y_aux_a = targets_a.reshape(-1)
            y_aux_b = targets_b.reshape(-1)
            aux_out_flat = aux_outputs.reshape(-1, 5)
            
            loss_aux = mixup_criterion(criterion, aux_out_flat, y_aux_a, y_aux_b, lam)
            
            loss = loss_main + 0.5 * loss_aux
            
            loss.backward()
            optimizer.step()
            
            # Per-Step Metrics
            batch_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_total = y_main.size(0)
            batch_correct = (predicted == y_main).sum().item()
            batch_acc = 100 * batch_correct / batch_total
            
            # Log every step
            train_losses_step.append(batch_loss)
            train_accs_step.append(batch_acc)
            
            running_loss += loss.item()
            total += y_main.size(0)
            correct += (predicted == y_main).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100*correct/total})
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        all_preds_epoch = []
        all_labels_epoch = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs, _ = model(X) # Ignore Aux for Validation
                _, predicted = torch.max(outputs.data, 1)
                
                y_main = y[:, -1]
                
                val_total += y_main.size(0)
                val_correct += (predicted == y_main).sum().item()
                
                all_preds_epoch.extend(predicted.cpu().numpy())
                all_labels_epoch.extend(y_main.cpu().numpy())
        
        # --- APPLY BIOLOGICAL RULES (v1.1) ---
        from sleep_logic import apply_transition_rules
        import numpy as np
        
        # Convert to numpy for processing
        all_preds_np = np.array(all_preds_epoch)
        all_labels_np = np.array(all_labels_epoch)
        
        # Apply Logic
        all_preds_corrected = apply_transition_rules(all_preds_np)
        
        # Update lists for downstream metrics
        all_preds_epoch = all_preds_corrected.tolist()
        
        # Recalculate Accuracy
        val_acc = 100 * (all_preds_corrected == all_labels_np).sum() / len(all_labels_np)
        # -------------------------------------
        
        # Calculate Advanced Metrics for this Epoch
        cm_epoch = confusion_matrix(all_labels_epoch, all_preds_epoch)
        kappa_epoch = cohen_kappa_score(all_labels_epoch, all_preds_epoch)
        precision_epoch, recall_epoch, f1_epoch, _ = precision_recall_fscore_support(all_labels_epoch, all_preds_epoch, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}% | Kappa: {kappa_epoch:.4f}")

        # --- REAL-TIME SAVE ---
        current_res = {
            'fold': fold_idx + 1,
            'train_loss': train_losses_step,
            'train_acc': train_accs_step,
            'val_acc': val_acc,
            'kappa': kappa_epoch, 
            'f1': f1_epoch, 
            'precision': precision_epoch, 
            'recall': recall_epoch,
            'confusion_matrix': cm_epoch.tolist()
        }
        
        all_res = previous_results + [current_res]
        
        avg_acc = np.mean([r['val_acc'] for r in previous_results]) if previous_results else val_acc
        
        final_output = {
            'folds': all_res,
            'average_acc': avg_acc, # Approximate
            'type': 'python_mps_eegsnet',
            'progress': f"Fold {fold_idx+1}/{total_folds} - Epoch {epoch+1}/{EPOCHS}",
            'kappa': kappa_epoch, 
            'f1': f1_epoch, 
            'precision': precision_epoch, 
            'recall': recall_epoch
        }
        
        try:
            with open(os.path.join(output_dir, 'results_8_subjects_python_v1_3.json'), 'w') as f:
                json.dump(final_output, f)
        except Exception as e:
            print(f"Warning: Failed to save intermediate result: {e}")
        # ----------------------
        
        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at Epoch {epoch+1}")
            break

    # Final Evaluation (Rest of function...)
    model.eval() # Redundant but kept for safety
    
    # Return last calculated metrics (or re-calculate if needed, but we have them)
    return {
        'fold': fold_idx + 1,
        'train_loss': train_losses_step,
        'train_acc': train_accs_step,
        'val_acc': val_acc,
        'confusion_matrix': cm_epoch.tolist(),
        'kappa': kappa_epoch,
        'precision': precision_epoch,
        'recall': recall_epoch,
        'f1': f1_epoch
    }

def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    # Load Data
    spectrograms, labels, splits = get_data_splits(DATA_FILE)
    
    results = []
    start_fold = 0
    
    # Checkpoint Loading
    checkpoint_path = os.path.join(OUTPUT_DIR, 'results_8_subjects_python_v1_3.json')
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                if 'folds' in data and isinstance(data['folds'], list):
                    loaded_results = data['folds']
                    if len(loaded_results) > 0:
                        last_result = loaded_results[-1]
                        # Check if last fold is complete (EPOCHS)
                        # We assume 'train_loss' stores per-step or per-epoch? 
                        # In my code: train_losses_step stores per-STEP. 
                        # A better check is needed. Maybe just check if the last saved dict has 'val_acc' (which implies epoch completion).
                        # BUT, my real-time save updates the LAST element *during* the fold.
                        # So if we crash mid-fold, the last element exists but is incomplete.
                        # How to distinguish?
                        # Simple Heuristic: If we are restarting, assume the last fold is incomplete/partial and Discard it to restart that fold cleanly.
                        # Unless `progress` says "Completed Fold X".
                        # But `progress` is in the outer dict.
                        
                        # Let's trust 'folds' count. If I have N folds in list, I have N *attempts*.
                        # If the last one was partial, I should remove it.
                        print(f"Found existing checkpoint with {len(loaded_results)} entries.")
                        print("Resuming... (Discarding last entry to ensure clean restart of that fold)")
                        
                        # Discard last one to be safe (re-run that fold from scratch)
                        # If it WAS complete, we re-run it. Better safe than handling partial state complexity.
                        if len(loaded_results) > 0:
                             loaded_results.pop()
                        
                        results = loaded_results
                        start_fold = len(results)
                        print(f"Resuming from Fold {start_fold + 1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    print(f"Starting LOSO CV on {len(splits)} subjects. Resuming from Fold {start_fold + 1}")
    
    for i in range(start_fold, len(splits)):
        print(f"Running Fold {i+1}...")
        
        test_start, test_end = splits[i]
        X_test = spectrograms[test_start:test_end]
        y_test = labels[test_start:test_end]
        
        train_indices = []
        for j, (start, end) in enumerate(splits):
            if i != j:
                train_indices.extend(list(range(start, end)))
                
        X_train = spectrograms[train_indices]
        y_train = labels[train_indices]
        
        # Calculate class counts for Sampling
        classes, counts = np.unique(y_train, return_counts=True)
        
        # v1.3: Weighted Random Sampling (Corrected for Sequence Length)
        # 1. Compute weight for each sample
        # Note: SleepDataset truncates the last (seq_len - 1) samples because they can't form a full sequence.
        # We must align weights to the valid indices.
        
        valid_length = len(y_train) - SEQ_LEN + 1 
        
        # Calculate weights for ALL labels first
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights_map = {cls: weight for cls, weight in zip(classes, class_weights)}
        
        # 2. Construct sample_weights for VALID indices only
        # Dataset index i -> Sequence y[i : i+SEQ_LEN]. The label for the sequence is y[i+SEQ_LEN-1] (the last one).
        
        valid_indices = np.arange(valid_length)
        target_label_indices = valid_indices + SEQ_LEN - 1
        target_labels = y_train[target_label_indices]
        
        sample_weights = np.array([class_weights_map[t] for t in target_labels])
        sample_weights = torch.from_numpy(sample_weights).double()
        
        # 3. Create Sampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=valid_length, replacement=True)
        print(f"Using WeightedRandomSampler for Fold {i+1}")
        print(f"  > Valid Dataset Length: {valid_length}")
        print(f"  > Sampler Weights Shape: {sample_weights.shape}")

        train_dset = SleepDataset(X_train, y_train, seq_len=SEQ_LEN)
        val_dset = SleepDataset(X_test, y_test, seq_len=SEQ_LEN)
        
        # Verify lengths match
        assert len(train_dset) == len(sample_weights), f"Dataset len {len(train_dset)} != Sampler len {len(sample_weights)}"
        
        # Shuffle must be False when using sampler
        train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler)
        val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Pass results and dirs for saving
        # Pass None for class_weights since we handle it via Sampling
        res = train_one_fold(i, train_loader, val_loader, device, results, OUTPUT_DIR, len(splits), class_weights=None)
        results.append(res)
        
        # Save Completed Fold Results
        final_output = {
            'folds': results,
            'average_acc': np.mean([r['val_acc'] for r in results]),
            'average_kappa': np.mean([r['kappa'] for r in results]),
            'average_f1': np.mean([r['f1'] for r in results]),
            'average_precision': np.mean([r['precision'] for r in results]),
            'average_recall': np.mean([r['recall'] for r in results]),
            'type': 'python_mps_eegsnet',
            'progress': f"Completed {i+1}/{len(splits)} folds"
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(final_output, f)
        print(f"Saved completed results for Fold {i+1}")
        
    print("Results saved.")

if __name__ == "__main__":
    main()
