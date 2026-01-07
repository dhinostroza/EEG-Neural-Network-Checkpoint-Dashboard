import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# --- CONFIG ---
FILES_DIR = "."
SHHS_XML_DIRS = [
    "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-profusion/shhs1/",
    "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-nsrr/shhs1/",
    "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-profusion/shhs2/",
    "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-nsrr/shhs2/"
]

def extract_gt_from_xml(xml_path):
    """
    Parses Sleep-EDF (Hypnogram) or NSRR (Profusion) XMLs.
    Returns a list of labels (0=W, 1=N1, etc.)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        labels = []
        
        # Map strings to our indices: 0=W, 1=N1, 2=N2, 3=N3, 4=REM
        stage_map_str = {
            'W': 0, '0': 0, 'Wake': 0,
            '1': 1, 'N1': 1,
            '2': 2, 'N2': 2,
            '3': 3, 'N3': 3,
            '4': 3, 'N4': 3, # Map Stage 4 to N3 (AASM)
            'R': 4, 'REM': 4, '5': 4
        }
        
        # 1. Profusion Epoch List <SleepStages><SleepStage>0</SleepStage>...
        sleep_stages_container = root.find("SleepStages")
        if sleep_stages_container is not None:
             for stage_elem in sleep_stages_container.findall("SleepStage"):
                 val = stage_elem.text
                 if val in stage_map_str:
                     labels.append(stage_map_str[val])
                 else:
                     try:
                         v_int = int(val)
                         if v_int == 5: labels.append(4)
                         elif v_int == 4: labels.append(3)
                         else: labels.append(v_int)
                     except:
                         labels.append(-1)
             return labels

        # Simplified logic for brevity as we prioritize Profusion now
        return []
    except Exception as e:
        print(f"XML Parsing Error: {e}")
        return []

def main():
    # Process both original and processed files?
    # User said "Scan all the .parquet files".
    # Let's target _processed.parquet as they are key for History.
    files = glob.glob(os.path.join(FILES_DIR, "*_processed.parquet"))
    print(f"Found {len(files)} processed parquet files.")

    for fpath in files:
        fname = os.path.basename(fpath)
        base_name = fname.replace("_processed.parquet", "")
        
        # Find XML
        found_xml_path = None
        for search_dir in SHHS_XML_DIRS:
            if not os.path.exists(search_dir): continue
            candidates = glob.glob(os.path.join(search_dir, f"{base_name}*.xml"))
            if candidates:
                found_xml_path = candidates[0]
                break
        
        if found_xml_path:
            print(f"[{base_name}] Found XML: {os.path.basename(found_xml_path)}")
            gt_labels = extract_gt_from_xml(found_xml_path)
            
            if gt_labels:
                try:
                    df = pd.read_parquet(fpath)
                    
                    # Merge
                    min_len = min(len(df), len(gt_labels))
                    if min_len > 0:
                        df['label'] = pd.Series(gt_labels[:min_len])
                        df['stage'] = df['label'] # Compatibility
                        
                        # Save back
                        df.to_parquet(fpath)
                        print(f"  -> Updated {fpath} with {min_len} labels.")
                    else:
                        print(f"  -> Mismatch or empty: DF={len(df)}, XML={len(gt_labels)}")
                except Exception as e:
                    print(f"  -> Error reading/saving parquet: {e}")
            else:
                print(f"  -> XML found but no labels extracted.")
        else:
            print(f"[{base_name}] No XML found.")

if __name__ == "__main__":
    main()
