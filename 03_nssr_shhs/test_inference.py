import torch
import pandas as pd
import numpy as np
import os
import inference

# Create dummy parquet file
dummy_data = np.random.randn(10, 76 * 60) # 10 samples, matches flatten size
df = pd.DataFrame(dummy_data)
# Add label col just to match robust logic
df['label'] = 0 
df.columns = df.columns.astype(str) # Parquet likes string cols
df.to_parquet("dummy_test.parquet")

print("Created dummy_test.parquet")

# Pick a checkpoints
checkpoint_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/"
# Try to find the detailed report to pick a valid one
# Or just pick one from listdir
import glob
ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
valid_ckpt = None
for c in ckpts:
    if "delete" not in c and "convnext" in c:
        valid_ckpt = c
        break

if valid_ckpt:
    print(f"Testing with checkpoint: {os.path.basename(valid_ckpt)}")
    try:
        inference.run_inference(valid_ckpt, "dummy_test.parquet")
        print("SUCCESS: Inference run finished without errors.")
    except Exception as e:
        print(f"FAILURE: {e}")
else:
    print("No valid checkpoint found to test.")
