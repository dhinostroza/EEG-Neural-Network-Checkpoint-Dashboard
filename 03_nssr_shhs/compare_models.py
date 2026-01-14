import torch
import pandas as pd
import numpy as np
import os
import sys
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# Add path to import inference util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import get_model, load_checkpoint_weights, preprocess_spectrogram, detect_architecture

# Helper to parse SHHS XML
def parse_shhs_xml(xml_path):
    import xml.etree.ElementTree as ET
    try:
        # Check if file exists
        if not os.path.exists(xml_path):
            return None
        
        # Read raw content to find SleepStages list
        # We can implement a simple parser if it's just a flat list of tags
        # but let's try standard ET for robustness, though SHHS XML is sometimes weird (CMPStudyConfig root)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Stages are usually in SleepStages > SleepStage
        stages = []
        
        # Try finding SleepStages tag directly
        sleep_stages_node = root.find('.//SleepStages')
        if sleep_stages_node is not None:
             for node in sleep_stages_node.findall('SleepStage'):
                 stages.append(int(node.text))
        else:
             # Fallback: Maybe just root has them?
             for node in root.findall('.//SleepStage'):
                 stages.append(int(node.text))
                 
        if not stages:
            return None
            
        print(f"    Extracted {len(stages)} labels from XML.")
        return np.array(stages)
        
    except Exception as e:
        print(f"    XML parsing failed: {e}")
        return None

class SpectrogramDataset(Dataset):
    def __init__(self, df, source_path=None):
        cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage', 'true_label'] if c in df.columns]
        self.data_matrix = df.drop(columns=cols_to_drop).values.astype(np.float32)
        
        # Labels - try to find truth
        self.labels = np.full(len(self.data_matrix), -1)
        
        # 1. Embedded labels
        if 'label' in df.columns:
            self.labels = df['label'].values
        elif 'true_label' in df.columns:
            self.labels = df['true_label'].values
            
        # 2. External XML (SHHS)
        # If labels are basically empty/invalid, try finding XML
        # Validation: If most labels are -1, try XML
        valid_mask = (self.labels != -1)
        if valid_mask.sum() < 100 and source_path and "shhs" in source_path:
            # Construct expected XML path
            # parquet: .../shhs1-200001_processed.parquet
            # expected xml: .../parquet_files/annotations-events-profusion/shhs1/shhs1-200001-profusion.xml
            
            base_name = os.path.basename(source_path) # shhs1-200001_processed.parquet
            # Extract ID: shhs1-200001
            # Assuming format shhs1-XXXXXX
            parts = base_name.split('_')[0] 
            
            # Search logic
            xml_dir = os.path.join(os.path.dirname(source_path), "parquet_files/annotations-events-profusion/shhs1")
            xml_name = f"{parts}-profusion.xml"
            xml_path = os.path.join(xml_dir, xml_name)
            
            # Simple fallback if dir structure is different
            if not os.path.exists(xml_path):
                 # Try finding via glob in general area
                 root_search = os.path.dirname(source_path)
                 candidates = glob.glob(os.path.join(root_search, "**", xml_name), recursive=True)
                 if candidates:
                     xml_path = candidates[0]
            
            if os.path.exists(xml_path):
                print(f"    Found external GT: {xml_name}")
                xml_labels = parse_shhs_xml(xml_path)
                
                if xml_labels is not None:
                    # Align lengths
                    # SHHS XML labels are usually 30s epochs. 
                    # Parquet should be same length.
                    n_parquet = len(self.labels)
                    n_xml = len(xml_labels)
                    
                    if n_parquet == n_xml:
                        self.labels = xml_labels
                    elif abs(n_parquet - n_xml) < 5:
                         # Minor mismatch, crop to min
                         length = min(n_parquet, n_xml)
                         self.labels[:length] = xml_labels[:length]
                    else:
                        print(f"    WARNING: Length mismatch Parquet ({n_parquet}) vs XML ({n_xml}). Ignoring XML.")

    def __len__(self):
        return len(self.data_matrix)
    
    def __getitem__(self, idx):
        flat_data = self.data_matrix[idx]
        tensor = preprocess_spectrogram(flat_data)
        return tensor, self.labels[idx]

def predict_dataset(model, df, source_path=None, batch_size=256):
    dataset = SpectrogramDataset(df, source_path=source_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    preds = []
    truths = []
    
    with torch.no_grad():
        for batch, label_batch in loader:
            logits = model(batch)
            batch_preds = torch.argmax(logits, dim=1).numpy()
            preds.extend(batch_preds)
            truths.extend(label_batch.numpy())
            
    return np.array(preds), np.array(truths)

def evaluate_model_on_files(model, files):
    total_preds = []
    total_truth = []
    
    for f in files:
        if not os.path.exists(f): 
            continue
        try:
            df = pd.read_parquet(f)
            p, t = predict_dataset(model, df, source_path=f)
            total_preds.append(p)
            total_truth.append(t)
            print(f"  Scored {os.path.basename(f)}: {len(p)} samples. Valid GT: {(t!=-1).sum()}")
        except Exception as e:
            print(f"  Error reading {f}: {e}")
            
    if not total_preds:
        return np.array([]), np.array([])
        
    return np.concatenate(total_preds), np.concatenate(total_truth)

def main():
    base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs"
    
    # Files to test
    # Mix of Sleep-EDF and SHHS
    files_to_test = [
        os.path.join(base_dir, "SC4001E.parquet"),
        os.path.join(base_dir, "shhs1-200001_processed.parquet"),
        os.path.join(base_dir, "shhs1-200002_processed.parquet")
    ]
    
    # Models
    # A: Old (Sept 04)
    ckpt_old = os.path.join(base_dir, "checkpoint_files/2000 files/2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt")
    # B: New Recalibrated (Jan 11)
    ckpt_new = os.path.join(base_dir, "checkpoint_files/2000 files/2026-01-11_Modelo_Fusionado_Recalibrated.ckpt")
    
    print("--- COMPARATIVE BENCHMARK ---")
    
    # 1. Load Model A
    print("\nLoading Model A (Old)...")
    model_a_name = detect_architecture(ckpt_old)
    model_a = get_model(model_name=model_a_name, num_classes=5)
    model_a, _ = load_checkpoint_weights(model_a, ckpt_old)
    model_a.eval()
    
    # 2. Evaluate A
    preds_a, truth_a = evaluate_model_on_files(model_a, files_to_test)
    
    # 3. Load Model B
    print("\nLoading Model B (Recalibrated)...")
    model_b_name = detect_architecture(ckpt_new)
    model_b = get_model(model_name=model_b_name, num_classes=5)
    model_b, _ = load_checkpoint_weights(model_b, ckpt_new)
    model_b.eval()
    
    # 4. Evaluate B
    preds_b, truth_b = evaluate_model_on_files(model_b, files_to_test)
    
    # 5. Report
    print("\n" + "="*40)
    print("FINAL RESULTS (Sleep-EDF + SHHS)")
    print("="*40)
    
    # Filter valid labels != -1
    mask_a = (truth_a != -1)
    if mask_a.sum() > 0:
        acc_a = accuracy_score(truth_a[mask_a], preds_a[mask_a])
        print(f"\nModel A (Old) Overall Acc: {acc_a:.4f}")
        
        # N1 Recall
        mask_n1 = (truth_a == 1)
        if mask_n1.sum() > 0:
            rec_n1 = accuracy_score(truth_a[mask_n1], preds_a[mask_n1])
            print(f"Model A N1 Recall: {rec_n1:.4f}")

    mask_b = (truth_b != -1)
    if mask_b.sum() > 0:
        acc_b = accuracy_score(truth_b[mask_b], preds_b[mask_b])
        print(f"\nModel B (New) Overall Acc: {acc_b:.4f}")
        
        # N1 Recall
        mask_n1_b = (truth_b == 1)
        if mask_n1_b.sum() > 0:
            rec_n1_b = accuracy_score(truth_b[mask_n1_b], preds_b[mask_n1_b])
            print(f"Model B N1 Recall: {rec_n1_b:.4f}")

    # Save to CSV
    df_res = pd.DataFrame({
        'true_label': truth_b,
        'pred_old': preds_a,
        'pred_new': preds_b
    })
    df_res.to_csv(os.path.join(base_dir, "full_comparison_results.csv"), index=False)
    print("\nSaved full results to full_comparison_results.csv")

if __name__ == "__main__":
    main()
