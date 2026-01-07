# -*- coding: utf-8 -*-
"""
This script migrates files from a specified Google Drive folder to a
Google Cloud Storage (GCS) bucket.

It's designed to be run in a Google Colab environment.
"""

import os
from google.colab import auth
from google.colab import drive
from google.cloud import storage

# --- 1. User Authentication ---
# This command will prompt you to authenticate with your Google account.
# This single authentication will grant access to both Google Drive and GCS
# for the project associated with your Colab environment.
print("Authenticating user...")
auth.authenticate_user()
print("Authentication successful.")

# --- 2. Mount Google Drive ---
# This makes your Google Drive files accessible within the Colab file system
# at the path '/content/drive'.
print("Mounting Google Drive...")
try:
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully at /content/drive.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    # Exit if drive mounting fails
    exit()

# --- 3. Configuration (PLEASE EDIT THESE VALUES FOR EACH RUN) ---
# Your Google Cloud project ID.
GCP_PROJECT_ID = "shhs-sleepedfx"

# The name of your GCS bucket.
GCS_BUCKET_NAME = "shhs-sleepedfx-data-bucket"

# The path to the folder in your Google Drive you want to migrate.
# This path is relative to "My Drive".
# --- CHANGE THIS VALUE FOR EACH FOLDER ---
# 1st run: "shhs1_processed"
# 2nd run: "shhs2_processed"
# 3rd run: "sleep_edfx_processed"
DRIVE_FOLDER_PATH = "sleep_edfx_processed"

# (Optional) Specify a destination folder within your GCS bucket.
# We will place each Drive folder into a matching folder in the bucket.
GCS_DESTINATION_PREFIX = DRIVE_FOLDER_PATH

# --- 4. Initialize GCS Client ---
try:
    storage_client = storage.Client(project=GCP_PROJECT_ID)
    bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
    print(f"Successfully connected to GCS bucket: '{GCS_BUCKET_NAME}'")
except Exception as e:
    print(f"Error connecting to GCS bucket. Please check your project ID and bucket name.")
    print(f"Details: {e}")
    exit()

# --- 5. File Migration Logic ---
# Construct the full source path in the mounted Drive.
full_drive_path = os.path.join('/content/drive/MyDrive', DRIVE_FOLDER_PATH)

if not os.path.isdir(full_drive_path):
    print(f"ERROR: The specified Google Drive folder does not exist: {full_drive_path}")
    exit()

print("\nStarting file migration...")
print(f"Source:      '{full_drive_path}'")
print(f"Destination: 'gs://{GCS_BUCKET_NAME}/{GCS_DESTINATION_PREFIX}/'")
print("-" * 40)

total_files_migrated = 0
total_files_failed = 0
total_files_skipped = 0 # Counter for skipped files

try:
    # os.walk recursively goes through the directory tree.
    for dirpath, _, filenames in os.walk(full_drive_path):
        for filename in filenames:
            # Full path of the source file
            local_path = os.path.join(dirpath, filename)

            # Create a relative path to maintain the folder structure in GCS
            relative_path = os.path.relpath(local_path, full_drive_path)

            # Construct the destination path (blob name) in GCS
            # This will place files inside a folder matching the source folder name
            gcs_blob_name = os.path.join(GCS_DESTINATION_PREFIX, relative_path)

            # --- Check if file already exists in GCS ---
            blob = bucket.blob(gcs_blob_name)
            if blob.exists():
                print(f"  [SKIPPED] '{relative_path}' already exists in GCS. Skipping.")
                total_files_skipped += 1
                continue # Skip to the next file if it exists

            try:
                # Upload the file if it doesn't exist
                blob.upload_from_filename(local_path)
                print(f"  [SUCCESS] Migrated '{relative_path}'")
                total_files_migrated += 1
            except Exception as e:
                print(f"  [FAILED]  Could not upload '{relative_path}'. Error: {e}")
                total_files_failed += 1

finally:
    # --- 6. Unmount Google Drive ---
    # It's good practice to unmount the drive when done.
    drive.flush_and_unmount()
    print("\nGoogle Drive unmounted.")

print("\n--- Migration Summary ---")
print(f"Folder migrated: '{DRIVE_FOLDER_PATH}'")
print(f"Total files successfully migrated: {total_files_migrated}")
print(f"Total files failed to migrate: {total_files_failed}")
print(f"Total files skipped (already in GCS): {total_files_skipped}") # Print skipped count
print("Migration process complete for this folder.")

# ==================== NEW CELL ====================

# = =============================================================================
# SCRIPT 1: DOWNLOAD AND UPLOAD DEPENDENCY SOURCE CODE
# (Run this in an environment WITH internet access, like Colab Pro+)
# ==============================================================================
import os

# --- CONFIGURATION ---
GCS_BUCKET_NAME = "shhs-sleepedfx-colab-deps"
# -------------------

# 1. Authenticate to Google Cloud
from google.colab import auth
auth.authenticate_user()
print("✅ Authenticated to Google Cloud.")

# 2. Clear the old, incompatible packages from the GCS bucket
gcs_packages_path = f"gs://{GCS_BUCKET_NAME}/packages/"
print(f"Clearing old packages from {gcs_packages_path}...")
!gsutil -m rm -r {gcs_packages_path}*
print("✅ Old packages cleared.")

# 3. Create the requirements.txt file locally
# We add 'build' and 'wheel' which are necessary for compiling from source
requirements_content = """
build
wheel
pytorch-lightning
timm
pandas
pyarrow==19.0.0
gcsfs
mne
scikit-image
matplotlib
gradio
ray[default]
"""
with open("requirements.txt", "w") as f:
    f.write(requirements_content)

print("✅ requirements.txt created.")

# 4. Create a local directory to store the downloaded source code
local_source_dir = "/tmp/pip-sources"
os.makedirs(local_source_dir, exist_ok=True)
print(f"Created local directory: {local_source_dir}")

# 5. Use pip to download the source distributions and wheels for all packages
print("🚀 Starting download of all required package source code and wheels...")
# Remove --no-binary=:all: to allow wheels to be downloaded
!pip download -r requirements.txt -d {local_source_dir}
print("✅ All packages downloaded successfully.")

# 6. Upload the source code and requirements file to your GCS bucket
print(f"🚀 Uploading contents of {local_source_dir} to {gcs_packages_path}...")
upload_result = !gsutil -m cp -r {local_source_dir}/* {gcs_packages_path}
upload_requirements_result = !gsutil cp requirements.txt gs://{GCS_BUCKET_NAME}/

if "CommandException: No URLs matched" in "\n".join(upload_result):
    print("\n❌ No source code files were uploaded to your GCS bucket because the download failed.")
else:
    print("\n✅ All source code files have been successfully uploaded to your GCS bucket.")

if "CommandException" in "\n".join(upload_requirements_result):
     print("❌ requirements.txt failed to upload.")
else:
     print("✅ requirements.txt has been successfully uploaded to your GCS bucket.")


print("\nMigration process complete.")

# ==================== NEW CELL ====================

