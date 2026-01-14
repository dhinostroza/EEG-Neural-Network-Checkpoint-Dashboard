import pandas as pd
import os
import re
import sys
from run_batch_inference_reports import generate_comparative_report, extract_gt_from_xml

SQL_PATH = "predictions.sql"
STAGE_MAP = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

def parse_sql():
    data = {} # filename -> {epoch: {'base': -1, 'ens': -1, 'true': -1}}
    print(f"Parsing {SQL_PATH}...")
    
    current_filename = None
    
    with open(SQL_PATH, 'r', encoding='latin1', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Optimization: Check for filename comment
            if line.startswith("-- Data for"):
                # -- Data for SC4072E.parquet
                parts = line.split(" ")
                if len(parts) >= 4:
                    current_filename = parts[3]
                continue
            
            if not line.startswith("("): continue

            # Simple split by comma, but be careful with strings
            # Expected format: ('id', 'fname', idx, 'stage', conf, 'model', 'true')
            # remove tailing ), or );
            content = line.strip().rstrip(";,")
            content = content.lstrip("(")
            content = content.rstrip(")")
            
            # naive split by ", "
            parts = content.split(", ")
            if len(parts) < 6: continue
            
            # Clean quotes
            parts = [p.strip("'") for p in parts]
            
            fname = parts[1]
            try:
                epoch = int(parts[2])
            except:
                continue
                
            pred_stage_str = parts[3]
            pred_stage = STAGE_MAP.get(pred_stage_str, -1)
            
            model = parts[5]
            
            true_stage = -1
            if len(parts) >= 7:
                t_str = parts[6]
                if t_str != 'NULL':
                    true_stage = STAGE_MAP.get(t_str, -1)
            
            if fname not in data: data[fname] = {}
            if epoch not in data[fname]: data[fname][epoch] = {'base': -1, 'ens': -1, 'true': -1}
            
            if "Ensemble" in model:
                data[fname][epoch]['ens'] = pred_stage
            else:
                data[fname][epoch]['base'] = pred_stage
                
            if true_stage != -1:
                data[fname][epoch]['true'] = true_stage

    return data

def main():
    print("Starting restoration...")
    data = parse_sql()
    print(f"Loaded data for {len(data)} files from SQL.")
    
    os.makedirs("png", exist_ok=True)
    count = 0
    
    # Filter for SHHS only if desired, or all
    # User asked for "shhs files"
    target_files = [f for f in data.keys() if "shhs" in f.lower()]
    print(f"Found {len(target_files)} SHHS files to restore.")
    
    for fname in target_files:
        epochs = data[fname]
        if not epochs: continue
        
        sorted_idxs = sorted(epochs.keys())
        max_idx = sorted_idxs[-1]
        
        # Reconstruct arrays
        # Fill missing epochs with -1
        y_true = []
        y_ens = []
        y_base = []
        
        valid_gt_in_sql = False
        
        for i in range(max_idx + 1):
            ep_data = epochs.get(i, {'base': -1, 'ens': -1, 'true': -1})
            y_ens.append(ep_data['ens'])
            y_base.append(ep_data['base'])
            t = ep_data['true']
            y_true.append(t)
            if t != -1: valid_gt_in_sql = True
            
        # If GT missing in SQL, try XML
        if not valid_gt_in_sql:
            # infer core name
            # shhs1-200001_processed_processed.parquet -> shhs1-200001 ?
            # shhs1-200001.parquet -> shhs1-200001
            # cleaning:
            core = fname.replace(".parquet", "")
            # handle repeated _processed suffix if present
            if "_processed" in core:
                core = core.replace("_processed", "")
            
            xml_fname = f"{core}-profusion.xml"
            xml_path = os.path.join("parquet_files", "annotations-events-profusion", "shhs1", xml_fname)
            
            gt_xml = extract_gt_from_xml(xml_path)
            if gt_xml:
                y_true = gt_xml
                # Truncate or Pad predictions to match GT length
                L = min(len(y_true), len(y_ens))
                y_true = y_true[:L]
                y_ens = y_ens[:L]
                y_base = y_base[:L]
                valid_gt_in_sql = True
                
        if not valid_gt_in_sql:
            # Skip if no GT (cannot generate comparative report without GT)
            # print(f"Skipping {fname} (No Ground Truth found)")
            continue
            
        # Generate PNG
        # naming: [DATE]_shhs1-....png
        # Use fixed date or current? Previous used '2026-01-11'
        DATE_STAMP = "2026-01-11" # match existing convention
        core_name = fname.replace(".parquet", "")
        output_base = os.path.join("png", f"{DATE_STAMP}_{core_name}")
        
        # Only generating if output doesn't exist? Or overwrite? Overwrite to be sure.
        try:
            generate_comparative_report(
                output_base, 
                y_true, 
                y_ens, 
                y_base, 
                "Baseline (ConvNext)", 
                0.0, 0.0, "Calc",
                lang='ES'
            )
            generate_comparative_report(
                output_base, 
                y_true, 
                y_ens, 
                y_base, 
                "Baseline (ConvNext)", 
                0.0, 0.0, "Calc",
                lang='EN'
            )
            count += 1
            if count % 10 == 0:
                print(f"Restored {count} reports...")
            
            if count >= 50:
                print("Limit of 50 files reached. Stopping restoration to prevent repository bloat.")
                break

        except Exception as e:
            print(f"Failed to generate report for {fname}: {e}")

    print(f"Restoration complete. Generated {count} reports.")

if __name__ == "__main__":
    main()
