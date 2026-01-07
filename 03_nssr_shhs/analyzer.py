import os
import glob
import json
import torch
import re
import datetime
import pandas as pd


CACHE_FILE = "analysis_cache_v3.json" # Bumped version to force re-scan

class CheckpointAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.cache_path = os.path.join(base_dir, CACHE_FILE)
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def extract_metadata(self, file_path):
        filename = os.path.basename(file_path)
        
        # Default values
        meta = {
            "filename": filename,
            "filepath": file_path,
            "size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "mtime": os.path.getmtime(file_path),
            "date_modified": datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M'),
            "model_architecture": "Unknown",
            "val_loss": None,
            "epoch": -1,
            "params_m": 0,
            "pl_version": "Unknown",
            "status": "Valid",
            "error": None,
            # Initialize these to None/0 so we prefer the filename extracted ones
            "date": "N/A", 
            "time": "N/A",
            "workers": 0,
            "trained_on_files": 0,
            "lr": "N/A",
            "weighted_sampler": False
        }

        try:
            # We must map to cpu to avoid GPU errors
            checkpoint = torch.load(file_path, map_location='cpu')
            
            # --- Architecture & Params ---
            hparams = checkpoint.get('hyper_parameters', {})
            meta["model_architecture"] = hparams.get('model_name', 'Unknown')
            meta["pl_version"] = checkpoint.get('pytorch-lightning_version', 'N/A')
            meta["epoch"] = checkpoint.get('epoch', -1)
            
            # --- EXTRACT FROM FILENAME (User Request) ---
            # e.g. 2025-09-04_05-36_best-model_convnext_base_2000files_lr2e-05_cwn1-8.0_workers2.ckpt
            try:
                # Date: YYYY-MM-DD
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
                if date_match:
                    meta["date"] = date_match.group(1)
                
                # Time: HH-MM (optionally followed by _)
                time_match = re.search(r'_(\d{2}-\d{2})', filename)
                if time_match:
                    meta["time"] = time_match.group(1).replace("-", ":")

                # Workers
                workers_match = re.search(r'workers(\d+)', filename, re.IGNORECASE)
                if workers_match:
                    meta["workers"] = int(workers_match.group(1))

                # Files
                files_match = re.search(r'(\d+)files', filename, re.IGNORECASE)
                if files_match:
                    meta["trained_on_files"] = int(files_match.group(1))

                # LR
                lr_match = re.search(r'lr([\deE\-\.]+)', filename, re.IGNORECASE)
                if lr_match:
                     meta["lr"] = lr_match.group(1)

                # CW (Class Weights / Weighted Sampler)
                if "cwn" in filename.lower() or "wrs" in filename.lower():
                    meta["weighted_sampler"] = True
                
                # Architecture from filename fallback
                if meta["model_architecture"] == "Unknown":
                    # Specific variants
                    for arch in ["convnext_base", "convnext_small", "convnext_tiny", "resnet18", "resnet50", "swin_base"]:
                        if arch in filename.lower():
                            meta["model_architecture"] = arch
                            break
                    
                    # Generic fallback (User clarified 'convnext' in these files matches 'convnext_tiny')
                    if meta["model_architecture"] == "Unknown" and "convnext" in filename.lower():
                         meta["model_architecture"] = "convnext_tiny"
            except Exception as parse_e:
                print(f"Warning: Filename parsing error for {filename}: {parse_e}")


            # Count params
            state_dict = checkpoint.get('state_dict', {})
            total_params = 0
            for tensor in state_dict.values():
                if hasattr(tensor, 'numel'):
                    total_params += tensor.numel()
            meta["params_m"] = round(total_params / 1_000_000, 2)
            
            # --- Validation Loss ---
            # Strategy 1: Look in Callbacks
            best_score = None
            callbacks = checkpoint.get('callbacks', {})
            for key, val in callbacks.items():
                if 'ModelCheckpoint' in key and isinstance(val, dict):
                        try:
                            s = val['best_model_score']
                            if isinstance(s, (int, float)):
                                best_score = float(s)
                            elif hasattr(s, 'item'):
                                best_score = s.item()
                            elif s is not None:
                                best_score = float(s)
                        except:
                            pass
            
            # Strategy 2: Look in Filename (regex)
            if best_score is None:
                match = re.search(r'val_loss=([\d\.]+)', filename)
                if match:
                    best_score = float(match.group(1))
            
            meta["val_loss"] = best_score
            
            # Identify suspicious/failed runs
            if meta["val_loss"] and meta["val_loss"] > 1000:
                meta["status"] = "Suspicious (High Loss)"
            
        except Exception as e:
            meta["status"] = "Error"
            meta["error"] = str(e)
            
        return meta

    def scan(self):
        """
        Scans directory recursively, processes new/modified files, updates cache.
        Returns a DataFrame of all data.
        """
        # all_files = sorted(glob.glob(os.path.join(self.base_dir, "*.ckpt")))
        all_files = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    all_files.append(os.path.join(root, file))
        
        updated = False
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            # Ignore deleted/marked files
            if "delete" in filename:
                continue
                
            mtime = os.path.getmtime(file_path)
            
            # Extract Group (Parent Folder)
            # relative path from base_dir
            rel_dir = os.path.relpath(os.path.dirname(file_path), self.base_dir)
            if rel_dir == ".":
                group = "Uncategorized"
            else:
                group = rel_dir
            
            # Check if in cache and valid
            if filename in self.cache:
                cached_entry = self.cache[filename]
                # If file hasn't changed, skip re-processing
                if cached_entry.get('mtime') == mtime:
                    # Update group if moved
                    if cached_entry.get('group') != group:
                        cached_entry['group'] = group
                        updated = True
                    
                    # CRITICAL FIX: Always update filepath in case it was moved
                    if cached_entry.get('filepath') != file_path:
                        cached_entry['filepath'] = file_path
                        updated = True
                        
                    continue
            
            # Process new or modified file
            print(f"Analyzing {filename}...")
            meta = self.extract_metadata(file_path)
            meta['group'] = group # Add group
            self.cache[filename] = meta
            updated = True
            
        if updated:
            self._save_cache()
            
        # Convert cache values to list
        data = list(self.cache.values())
        # Filter out any files that might have been deleted from disk but exist in cache
        # (Optional cleanup step)
        existing_filenames = {os.path.basename(f) for f in all_files}
        data = [d for d in data if d['filename'] in existing_filenames]
        
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Test run
    analyzer = CheckpointAnalyzer("/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/")
    df = analyzer.scan()
    print(df.head())
    print(f"Total rows: {len(df)}")
