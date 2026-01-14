
import pandas as pd
import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SC_PATTERN = os.path.join(BASE_DIR, "parquet_files/SC*.parquet")

files = glob.glob(SC_PATTERN)
if not files:
    # Try looking in local dir
    files = glob.glob("SC*.parquet")

if files:
    f = files[0]
    print(f"Checking {f}...")
    try:
        df = pd.read_parquet(f)
        print(f"Columns: {df.columns.tolist()}")
        if 'true_label' in df.columns or 'label' in df.columns:
            print("CONFIRMATION: Labels found.")
            # check unique values
            if 'true_label' in df.columns:
                print(f"Unique Labels (true_label): {df['true_label'].unique()}")
            else:
                 print(f"Unique Labels (label): {df['label'].unique()}")
        else:
            print("WARNING: No labels found.")
    except Exception as e:
        print(f"Error reading {f}: {e}")
else:
    print("No SC files found to check.")
