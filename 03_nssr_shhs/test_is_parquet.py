
import pandas as pd

FILE = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/Registro de EEG-HECAM_processed.bdf"

try:
    df = pd.read_parquet(FILE)
    print("Success! File is a valid Parquet file.")
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Failed to read as Parquet: {e}")
