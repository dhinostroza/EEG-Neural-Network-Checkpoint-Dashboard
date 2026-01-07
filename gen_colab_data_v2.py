import os

# Define the descriptions mapping with CLEAN keys (no .py suffix)
DESCRIPTION_MAP = {
    "2025-09-01a ConvNeXt_tiny 500files GCS.ipynb": "Training Script (ConvNeXt Tiny). Early experiment training a ConvNeXt Tiny model on a small subset (500 files) using Google Cloud Storage. Establishes the baseline for the cloud-based pipeline.",
    "2025-09-01b ConvNext_base 1000files GCS.ipynb": "Training Script (ConvNeXt Base). Scaled up experimentation using the heavier ConvNeXt Base architecture on a larger dataset (1000 files). optimized for higher capacity learning.",
    "2025-09-02 ConvNextv2_base cosine_annealing_lr-2e-5_epochs=40 2000files_w2 GCS.ipynb": "Training Script (ConvNeXt v2 Base). Integration of the state-of-the-art ConvNeXt V2 architecture with Cosine Annealing Learning Rate scheduling. Tuned for 40 epochs on 2000 files.",
    "2025-09-04 Performance Plot": "Performance Visualization. A graphical report showing the Training vs. Validation Loss and Accuracy curves for the ConvNeXt Base model, highlighting convergence behavior over 40 epochs.",
    "2025-09-06 ConvNeXTv2 base_epochs=40_1000files_lr-2e-5_gpu.ipynb": "Optimization Script (GPU Normalization). A critical optimization step where data normalization was moved to the GPU within the LightningModule, significantly increasing training throughput.",
    "2025-09-10 Manifest multiple scripts CONVNEXT 2000files Training.ipynb": "Infrastructure: Manifest Creation. The 'Definitive' solution for dataset management. Creates a `manifest.csv` to track valid files and epochs, using a strictly isolated VENV to prevent Colab dependency conflicts.",
    "2025-09-11 Consolidate spectrograms_labels_dataset into large files.ipynb": "Experiment: Large File Consolidation. An attempt to consolidate the dataset into massive single files. MARKED AS LEGACY due to PyArrow binary incompatibility issues discovered during testing.",
    "2025-09-14 Training script manifest and sampler.ipynb": "Training Script (Weighted Sampler). Advanced training setup utilizing the `manifest.csv` and a `WeightedRandomSampler` to mutually address the class imbalance between N2 (common) and N1 (rare) stages.",
    "2025-09-20 Consolidate_Dataset_Small Chunks VENV multiple files OK.ipynb": "Infrastructure: Definitive Consolidation. The successful, production-grade script for consolidating the SHHS dataset. Uses memory-safe chunking (50 files/chunk) and Virtual Environments to ensure stability.",
    "2025-09-22 Google Drive to GCS Migration Script": "Infrastructure: Cloud Migration. Utility script specifically designed to migrate the processed dataset from Google Drive to Google Cloud Storage (GCS) buckets for high-speed I/O access.",
    "2025-10-04 List ACTIVE DATASET FILES.ipynb": "Utility: Dataset Verification. A simple sanity check tool used to list and verify the files currently active in the training dataset directory."
}

# The source directory still has .py files
SOURCE_DIR = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/colab_notebooks/2025-09/extracted_code"
OUTPUT_FILE = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/colab_data.py"

print("Regenerating colab_data.py with proper keys...")

content = "COLAB_NOTEBOOKS = {\n"

# We iterate through the clean keys and look for the corresponding .py file
for clean_name, description in DESCRIPTION_MAP.items():
    if "Performance Plot" in clean_name:
         # Special handling for the image entry
         content += f"    '{clean_name}': {{\n"
         content += f"        'title': '{clean_name}',\n"
         content += f"        'description': {repr(description)},\n"
         content += f"        'code': None,\n"
         content += f"        'image': 'img/perf_plot_2025_09.png'\n"
         content += "    },\n"
         continue

    # Look for the file. Note: The extracted files have .ipynb.py extension if they were notebooks
    # Some might be just .py if they were scripts.
    # The clean name usually ends in .ipynb
    
    candidate_filename = clean_name + ".py"
    file_path = os.path.join(SOURCE_DIR, candidate_filename)
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            code_text = f.read()
            content += f"    '{clean_name}': {{\n"
            content += f"        'title': {repr(clean_name)},\n"
            content += f"        'description': {repr(description)},\n"
            content += f"        'code': {repr(code_text)},\n"
            content += f"        'image': None\n"
            content += "    },\n"
    else:
        # Try without .ipynb in the source filename? No, extracted_code kept the extension usually.
        # Let's check listing if needed, but for now assuming standard extraction naming.
        print(f"Warning: File not found: {candidate_filename}")

content += "}\n"

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done. Written to {OUTPUT_FILE}")
