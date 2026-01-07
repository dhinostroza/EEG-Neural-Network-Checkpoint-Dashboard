import os
import glob
import pandas as pd
import shutil

# --- CONFIG ---
BASE_DIR = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files"
SQL_PATH = "predictions.sql"
BACKUP_PATH = "predictions_backup.sql"

def migrate_database():
    print("Starting SQL Migration (Adding Ground Truth)...")
    
    if not os.path.exists(SQL_PATH):
        print("No predictions.sql found to migrate.")
        return

    # 1. Backup
    shutil.copy(SQL_PATH, BACKUP_PATH)
    print(f"Backed up existing SQL to {BACKUP_PATH}")
    
    # 2. Read All Predictions
    # We parse the file manually because it's a series of INSERTs
    old_data = [] # List of dicts
    
    with open(SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    print(f"Read {len(lines)} lines from SQL.")
    
    # 3. Parse and buffer
    # Format: ('patient', 'filename', idx, 'stage', conf, 'model')
    count = 0
    
    new_inserts = []
    
    # We will group by filename to minimize parquet reads
    file_groups = {} 
    
    for line in lines:
        if "INSERT INTO" in line:
            continue
        if "CREATE TABLE" in line:
            continue
        if line.startswith("--"):
            continue
        if ";" in line: # End of block
            pass
            
        # Extract values between parens
        # This is a simple parser, assuming no complex nested parens
        # Raw: ('p', 'f', 0, 'W', 0.9, 'm'), ...
        # We start by splitting by "), ("
        
        # Actually, the file structure usually has one large INSERT or multiple blocks.
        # My app writes:
        # INSERT ... VALUES
        # (val),
        # (val);
        
        clean_line = line.strip()
        if not clean_line.startswith("("):
            continue
            
        # Remove trailing comma or semicolon
        clean_line = clean_line.rstrip(";,")
        
        # Split (naive)
        parts = clean_line.split(",")
        if len(parts) >= 6:
            # part 1: filename (index 1)
            # ('ID', 'FILENAME', ...
            try:
                fname = parts[1].strip().replace("'", "")
                epoch_idx = int(parts[2].strip())
                
                if fname not in file_groups:
                    file_groups[fname] = []
                
                file_groups[fname].append({
                    "line": clean_line,
                    "parts": parts,
                    "idx": epoch_idx
                })
            except:
                continue

    print(f"Identified {len(file_groups)} files in SQL history.")
    
    # 4. Migrate
    rebuilt_sql_lines = []
    rebuilt_sql_lines.append("CREATE TABLE IF NOT EXISTS sleep_predictions (patient_id TEXT, filename TEXT, epoch_index INT, predicted_stage TEXT, confidence FLOAT, model_used TEXT, true_stage TEXT);\n")
    
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    
    for fname, rows in file_groups.items():
        print(f"Migrating {fname} ({len(rows)} epochs)...")
        
        rebuilt_sql_lines.append(f"-- Data for {fname}\n")
        rebuilt_sql_lines.append("INSERT INTO sleep_predictions (patient_id, filename, epoch_index, predicted_stage, confidence, model_used, true_stage) VALUES\n")
        
        # Load Parquet for GT
        # Try both exact name and _processed
        gt_map = {} # idx -> label_str
        
        pq_path = os.path.join(BASE_DIR, "**", fname)
        # Using glob to find recursively
        candidates = glob.glob(pq_path, recursive=True)
        if not candidates:
             # Try _processed
             if not fname.endswith("_processed.parquet"):
                 fname_proc = fname.replace(".parquet", "_processed.parquet")
                 pq_path = os.path.join(BASE_DIR, "**", fname_proc)
                 candidates = glob.glob(pq_path, recursive=True)
        
        if candidates:
            try:
                df = pd.read_parquet(candidates[0])
                # Find GT col
                gt_col = None
                for c in ['label', 'stage', 'sleep_stage', 'true_label']:
                    if c in df.columns:
                        gt_col = c
                        break
                
                if gt_col:
                    vals = df[gt_col].tolist()
                    for i, v in enumerate(vals):
                        gt_map[i] = stage_map.get(v, "Unknown")
            except Exception as e:
                print(f"  Failed to load GT from {candidates[0]}: {e}")
        else:
            print(f"  Warning: Parquet file not found via glob: {fname}")

        # Build new Values
        new_values = []
        rows.sort(key=lambda x: x['idx'])
        
        for row in rows:
            parts = row['parts']
            idx = row['idx']
            
            # Reconstruct first 6 parts
            # ('p', 'f', i, 's', c, 'm')
            # Careful not to double quote if already quoted, but logic below just appends
            
            # The parts list might contain split strings within quotes if commas exist... 
            # But our data is simple.
            
            base_parts = ",".join(parts[:6]) 
            # Check if 7th exists (already migrated?)
            if len(parts) >= 7:
                 # Already has GT?
                 val_str = ",".join(parts) # Keep as is
            else:
                # Add GT
                if idx in gt_map:
                    gt_str = f"'{gt_map[idx]}'"
                else:
                    gt_str = "NULL"
                
                val_str = f"{base_parts}, {gt_str})" 
                # Note: base_parts included the closing ')' of the 6th element?
                # parts[5] is "'Model')" likely.
                # Use string manipulation to be safer.
                
                original_content = row['line'] # ('a', 'b', ... 'm')
                original_content = original_content.rstrip(")") # ('a', ... 'm'
                val_str = f"{original_content}, {gt_str})"
            
            new_values.append(val_str)
            
        rebuilt_sql_lines.append(",\n".join(new_values))
        rebuilt_sql_lines.append(";\n")
        
    # 5. Write
    with open(SQL_PATH, 'w') as f:
        f.writelines(rebuilt_sql_lines)
        
    print("Migration Complete.")

if __name__ == "__main__":
    migrate_database()
