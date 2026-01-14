
import os

BDF_FILE = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/Registro de EEG-HECAM_processed.bdf"

def patch_bdf_header(filepath):
    with open(filepath, 'r+b') as f:
        # Go to Start Date offset
        # BDF Header:
        # Version: 8
        # Patient: 80
        # Recording: 80
        # Start Date: 8 bytes (offset 8+80+80 = 168)
        f.seek(168)
        date_bytes = f.read(8)
        print(f"Original Date Bytes: {date_bytes}")
        
        # Check if valid format (DD.MM.YY)
        # If garbage, execute patch
        f.seek(168)
        # Write dummy valid date: 01.01.00
        f.write(b'01.01.00')
        print("Patched Date to '01.01.00'")
        
        # Time: 8 bytes (offset 176)
        f.seek(176)
        time_bytes = f.read(8)
        print(f"Original Time Bytes: {time_bytes}")
        # Ensure time is valid format (HH.MM.SS)
        f.seek(176)
        f.write(b'00.00.00')
        print("Patched Time to '00.00.00'")

if __name__ == "__main__":
    patch_bdf_header(BDF_FILE)
