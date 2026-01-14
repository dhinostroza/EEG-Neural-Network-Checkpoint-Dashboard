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

    return models

def predict_ensemble(models, df, source_path=None, batch_size=256):
    # Determine Device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Ensemble Inference Device: {device}")

    # Move models to device
    for model in models:
        model.to(device)

    dataset = SpectrogramDataset(df, source_path=source_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    avg_probs_list = []
    truths = []
    
    with torch.no_grad():
        for batch, label_batch in loader:
            # batch: (B, 1, 76, 60)
            batch = batch.to(device)
            
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
            
            # 3. Store (Move back to CPU)
            avg_probs_list.append(avg_probs.cpu().numpy())
            truths.extend(label_batch.numpy())
            
    # Concatenate
    all_probs = np.concatenate(avg_probs_list, axis=0) # (N, 5)
    predictions = np.argmax(all_probs, axis=1) # (N,)
    
    return predictions, np.array(truths)

# Logic module - no main execution

