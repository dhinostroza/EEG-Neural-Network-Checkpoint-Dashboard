
print("Reading first 16 bytes of BDF file...")
BDF_FILE = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/Registro de EEG-HECAM_processed.bdf"
with open(BDF_FILE, 'rb') as f:
    header = f.read(16)
    print(f"Header: {header}")
    if header[0] == 0xFF:
        print("First byte is 0xFF (Valid BDF start marker)")
    else:
        print(f"First byte is {hex(header[0])} (INVALID BDF start marker)")
    
    print(f"ASCII part: {header[1:8]}")
