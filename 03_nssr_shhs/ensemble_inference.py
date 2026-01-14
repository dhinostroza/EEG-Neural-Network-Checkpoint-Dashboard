import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# Add path to import inference util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import get_model, load_checkpoint_weights, preprocess_spectrogram, detect_architecture
from compare_models import SpectrogramDataset # Reuse dataset with XML logic

def load_ensemble_models(ckpt_paths):
    models = []
    for path in ckpt_paths:
        print(f"Loading ensemble member: {os.path.basename(path)}")
        model_name = detect_architecture(path)
        model = get_model(model_name=model_name, num_classes=5)
        model, _ = load_checkpoint_weights(model, path)
        model.eval()
        models.append(model)
    return models

def predict_ensemble(models, df, source_path=None, batch_size=256):
    dataset = SpectrogramDataset(df, source_path=source_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    avg_probs_list = []
    truths = []
    
    with torch.no_grad():
        for batch, label_batch in loader:
            # batch: (B, 1, 76, 60)
            
            # 1. Get logits from all models
            batch_probs_sum = None
            
            for model in models:
                logits = model(batch)
                probs = F.softmax(logits, dim=1) # Convert to probabilities
                
                if batch_probs_sum is None:
                    batch_probs_sum = probs
                else:
                    batch_probs_sum += probs
            
            # 2. Average
            avg_probs = batch_probs_sum / len(models)
            
            # 3. Store
            avg_probs_list.append(avg_probs.numpy())
            truths.extend(label_batch.numpy())
            
    # Concatenate
    all_probs = np.concatenate(avg_probs_list, axis=0) # (N, 5)
    predictions = np.argmax(all_probs, axis=1) # (N,)
    
    return predictions, np.array(truths)

def main():
    base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs"
    
    # 1. Select Checkpoints for Ensemble
    ckpt_dir = os.path.join(base_dir, "checkpoint_files/2000 files")
    ckpts = [
        os.path.join(ckpt_dir, "2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt"),
        os.path.join(ckpt_dir, "2025-09-09_15-43_convnext_base_2000files_Augmented_cwN1-6.5.ckpt"),
        os.path.join(ckpt_dir, "2025-09-19_04-34_convnext_base_consolidated_cwN1-6.5-epoch=3-val_loss=0.5971.ckpt")
    ]
    
    # 2. Files to test
    files_to_test = [
        os.path.join(base_dir, "SC4001E.parquet"),
        os.path.join(base_dir, "shhs1-200001_processed.parquet"),
        os.path.join(base_dir, "shhs1-200002_processed.parquet")
    ]
    
    # 3. Load Models
    print("Initializing Ensemble...")
    models = load_ensemble_models(ckpts)
    
    # 4. Run Benchmark
    print("\n--- ENSEMBLE BENCHMARK ---")
    
    total_preds = []
    total_truth = []
    
    for f in files_to_test:
        if not os.path.exists(f): continue
        
        try:
            print(f"\nProcessing {os.path.basename(f)}...")
            df = pd.read_parquet(f)
            preds, gt = predict_ensemble(models, df, source_path=f)
            
            # Score
            valid_mask = (gt != -1)
            if valid_mask.sum() > 0:
                acc = accuracy_score(gt[valid_mask], preds[valid_mask])
                print(f"  Accuracy: {acc:.4f} (on {valid_mask.sum()} samples)")
            else:
                print("  No valid GT.")
                
            total_preds.append(preds)
            total_truth.append(gt)
            
        except Exception as e:
            print(f"Error on {f}: {e}")
            
    # 5. Review Global Metrics
    all_preds = np.concatenate(total_preds)
    all_truth = np.concatenate(total_truth)
    
    valid_mask = (all_truth != -1)
    if valid_mask.sum() > 0:
        final_acc = accuracy_score(all_truth[valid_mask], all_preds[valid_mask])
        print("\n" + "="*30)
        print(f"ENSEMBLE GLOBAL ACCURACY: {final_acc:.4f}")
        print("="*30)
        
        # N1 Recall
        n1_mask = (all_truth == 1)
        if n1_mask.sum() > 0:
            n1_rec = accuracy_score(all_truth[n1_mask], all_preds[n1_mask])
            print(f"Ensemble N1 Recall: {n1_rec:.4f}")
            
        # N2 Recall
        n2_mask = (all_truth == 2)
        if n2_mask.sum() > 0:
            n2_rec = accuracy_score(all_truth[n2_mask], all_preds[n2_mask])
            print(f"Ensemble N2 Recall: {n2_rec:.4f}")

    # Save CSV
    df_out = pd.DataFrame({
        'true_label': all_truth,
        'pred_ensemble': all_preds
    })
    df_out.to_csv(os.path.join(base_dir, "ensemble_results.csv"), index=False)
    print("Saved ensemble_results.csv")

if __name__ == "__main__":
    main()
