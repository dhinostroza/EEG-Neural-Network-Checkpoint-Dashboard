import torch
import os

# Define input and output paths
checkpoint_path = r"/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/2000 files/2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt"
output_txt_path = r"/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_metadata_result.txt"

print("Loading checkpoint... this will take a few seconds.")
# Load the checkpoint safely to CPU
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

target_keys = ['hyper_parameters', 'datamodule_hyper_parameters', 'callbacks', 'args', 'metadata']

print(f"Writing metadata to {output_txt_path}...")

# Open the text file in write mode
with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write("=== CHECKPOINT TOP-LEVEL KEYS ===\n")
    f.write(str(list(checkpoint.keys())) + "\n\n")

    # Search for and write the target metadata
    for key in target_keys:
        if key in checkpoint:
            f.write(f"=== CONTENTS OF '{key}' ===\n")
            if isinstance(checkpoint[key], dict):
                for sub_key, sub_value in checkpoint[key].items():
                    f.write(f"{sub_key}: {sub_value}\n")
            else:
                f.write(f"{checkpoint[key]}\n")
            f.write("\n")

print("Done! You can now open the text file to search for your datasets.")