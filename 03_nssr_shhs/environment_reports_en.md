# Development Environment and Scripts Evolution Report (August 2025)

This document details the evolution of scripts and execution environments used during the development of sleep stage classification models. A clear transition is observed from basic scripts in local/standard environments towards complex cloud-oriented workflows (Cloud-Native) and Colab Enterprise.

## Phase 1: Data Preparation and Basic Environment
**Objective:** Initial data transformation and organization.
**Environment:** Google Colab Standard + Google Drive Mount.

### 1. `2025-08-10 pre_SHHS_edf2parquet.ipynb`
*   **Description:** Initial preprocessing script. Converts original SHHS dataset EDF files to Parquet format for more efficient reading.
*   **Environment Details:**
    *   Direct Google Drive mount (`/content/drive`).
    *   Basic library installation (`mne`, `pytorch-lightning`, `timm`) via `pip`.
    *   Handling of mixed data type warnings (Pandas/Parquet).
*   **Milestone:** Establishment of the base data pipeline.

### 2. `2025-08-11 Split shhs1and2.ipynb`
*   **Description:** Utility script for file organization. Separates and moves processed files into structured directories (`shhs1_processed`, `shhs2_processed`).
*   **Environment Details:**
    *   Use of standard libraries `shutil` and `pathlib`.
    *   Simple execution on the mounted Drive file system.

---

## Phase 2: Experimentation and Architecture Search
**Objective:** Proofs of concept, hyperparameter tuning, and model comparison.
**Environment:** Google Colab Standard/Pro (GPU T4/V100).

### 3. Experimentation Scripts (Various)
*   **Files:**
    *   `2025-08-17 SHHS subset_ReduceLROnPlateau scheduler_lr_tests.ipynb`
    *   `2025-08-23 Sleep Stage Model - ConvNeXt_tiny_base_ViT_and_EfficientNet B0.ipynb`
    *   `2025-08-25 Adv_model_comparison_1000files_convNeXt_tiny_and_base.ipynb`
*   **Description:**
    *   Transition towards more complex architectures (ConvNeXt, ViT, EfficientNet).
    *   Implementation of dynamic schedulers (`ReduceLROnPlateau`).
    *   Experimentation with data subsets (100-1000 files) for rapid iteration.
*   **Evolution:** Start of training code modularization (`SleepStageClassifierLightning`).

---

## Phase 3: Performance Optimization and Scaling
**Objective:** Overcome I/O bottlenecks and train with the full dataset.
**Environment:** Google Colab Pro+ (GPU A100/High-RAM).

### 4. `2025-08-31 Sleep Stage Classif Test environment, ConvNeXt_tiny_GDrive.ipynb`
*   **Description:** Optimized training using ephemeral local storage.
*   **Environment Details:**
    *   **Local Cache Strategy:** Conditional copying of data from Drive to the VM's local disk (`/content/local_shhs_data`) to eliminate Drive network latency during training.
    *   **Hardware Verification:** Explicit check for GPU availability (A100) and memory management.
*   **Milestone:** Solution to dataloader speed issues ("bottleneck") typical of large trainings on Drive.

---

## Phase 4: Cloud-Native and Colab Enterprise (Production)
**Objective:** Large-scale training, reproducibility, and professional dependency management.
**Environment:** Google Colab Enterprise + Google Cloud Storage (GCS Buckets).

### 5. `2025-08-23 sleep-stage-training-full-parquet-dataset_GColab_Enterprise.ipynb`
*   **Description:** "State of the Art" (SOTA) training script prepared for the full dataset.
*   **Environment Details:**
    *   **Google Cloud Storage (GCS):** Abandonment of Google Drive in favor of Buckets (`gs://shhs-sleepedfx-data-bucket`) for massive, high-speed storage.
    *   **Optimized Dataset with Metadata:** Implementation of `OptimizedCombinedDataset` which generates and reads a metadata file (`dataset_metadata.csv`) in the bucket, avoiding re-scanning thousands of files at each start.
    *   **Authentication:** Use of `google.colab.auth` for secure access to Cloud resources.
    *   **Architecture:** Robust multi-model support (`convnext_base`, `vit_base`).

### 6. `2025-08-24 Download dependencies and upload to bucket.ipynb`
*   **Description:** "Air-gapped" / Enterprise infrastructure management.
*   **Environment Details:**
    *   **Source Downloads:** Downloads source distributions (`.tar.gz`, `.whl`) of all critical dependencies (`pytorch-lightning`, `timm`, etc.) locally.
    *   **Private Repository:** Uploads these packages to a dedicated bucket (`gs://shhs-sleepedfx-colab-deps/packages/`).
    *   **Purpose:** Ensure exact reproducibility and allow installations in enterprise environments without public internet access (PyPI), guaranteeing an immutable and secure training environment.

---

## Transition Summary
1.  **Local/Drive:** Simple scripts, slow data access, exclusive CPU use due to incompatibility between Matlab and Apple Silicon mps GPU, manual management.
2.  **Optimized:** VM disk caching, use of advanced GPUs (A100).
3.  **Enterprise/Cloud:** Bucket storage, asynchronous I/O, immutable dependency management, scaling to full datasets (TB+).
