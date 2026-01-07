# ==============================================================================
# STANDALONE UTILITY SCRIPT: This script only lists the ACTIVE DATASET FILES
# It's a sanity check to get a report of the files being used, but it needs the
# login script to work. I never used it because there are other, pressing
# problems and this is not relevant anymore.
# ==============================================================================
import os
import sys
from google.colab import drive, auth

print("--- Dataset Verification Utility (Standalone) ---")

# 1. SETUP
try:
    print("\n[Step 1/4] Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted.")
    print("\n[Step 2/4] Authenticating to Google Cloud...")
    auth.authenticate_user()
    print("✅ Authentication successful.")
except Exception as e:
    print(f"❌ ERROR during setup: {e}")
    sys.exit()

# 2. CONFIGURATION
try:
    print("\n[Step 3/4] Reading project configuration...")
    project_path = "/content/drive/MyDrive/sleep_study_project"
    if not os.path.exists(project_path):
        raise FileNotFoundError(f"CRITICAL: Project directory not found at '{project_path}'.")
    if project_path not in sys.path:
        sys.path.append(project_path)
    os.chdir(project_path)
    print(f"  -> Successfully changed directory to project folder.")
    from config import LOCAL_DATA_DIR, GCS_DATA_DIR_NAME
    local_data_path = os.path.join(LOCAL_DATA_DIR, GCS_DATA_DIR_NAME)
    print(f"  -> Target data directory (from config): {local_data_path}")
    print("✅ Configuration loaded successfully.")
except ImportError:
    print("❌ ERROR: Could not import settings from config.py.")
    print(f"   Please make sure 'config.py' exists and is up to date inside '{project_path}'.")
    sys.exit()
except Exception as e:
    print(f"❌ ERROR during configuration: {e}")
    sys.exit()

# 3. EXECUTION
print("\n[Step 4/4] Searching for and listing dataset files...")
if not os.path.exists(local_data_path):
    print(f"❌ ERROR: Local data directory not found at '{local_data_path}'.")
    print("   Please run the 'Setup & Data Download' cell in your training notebook first.")
    sys.exit()
try:
    print(f"  -> Searching in: {local_data_path}")
    all_files = [os.path.join(local_data_path, f) for f in os.listdir(local_data_path)]
    label_files = sorted([f for f in all_files if "smote_labels" in f])
    if not label_files:
        print("\n❌ WARNING: No 'smote_labels' files were found in the directory.")
    else:
        total_pairs = len(label_files)
        print(f"\n✅ Found a total of {total_pairs} file pairs.")
        print("\n--- Files are selected in the following alphabetical/numerical order: ---")
        print("\nFirst 5 file pairs:")
        for i in range(min(5, total_pairs)):
            label_file = label_files[i]
            spec_file = label_file.replace("smote_labels", "smote_spectrograms")
            print(f"  {i+1: >3}. {os.path.basename(spec_file): <40} & {os.path.basename(label_file)}")
        if total_pairs > 5:
            print("\n...")
            print("\nLast 5 file pairs:")
            for i in range(max(5, total_pairs - 5), total_pairs):
                label_file = label_files[i]
                spec_file = label_file.replace("smote_labels", "smote_spectrograms")
                print(f"  {i+1: >3}. {os.path.basename(spec_file): <40} & {os.path.basename(label_file)}")
except Exception as e:
    print(f"\n❌ An unexpected error occurred while listing files: {e}")
print("\n--- Verification complete ---")

# ==================== NEW CELL ====================

