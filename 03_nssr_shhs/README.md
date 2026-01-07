# EEG Neural Network Checkpoint Dashboard
![Version](https://img.shields.io/badge/version-v1.5.0-blue)

A Streamlit-based dashboard for managing, analyzing, and running inference with PyTorch Lightning checkpoints trained on SHHS EEG data.

### 🆕 v1.5.0 Changes
- **Hypnogram Precision**: Fixed Ground Truth drawing logic (z-order) to ensure true labels are always visible over predictions.
- **UI UX Refinement**: 
    -   Split "Resumen" and "Desglose" in notebook descriptions for better readability.
    -   Renamed "Archivos procesados" to "**EEG de Sleep-EDFX y SHHS**".
    -   Improved layout of Glyph Glossary and File Uploader labels.
- **SMOTE & Notebooks**: Updated timeline and notebook reporting to accurately reflect SMOTE implementation and notebook details.

## 🌟 Features

*   **Model Registry**: Automatically scans and lists trained `.ckpt` models, extracting metadata (Validation Loss, Architecture, Parameters, Epochs) directly from filenames and checkpoints.
*   **Performance Tracking**: Highlights high-performing models (Green for good, Red for poor) based on validation loss.
*   **Inference Playground**:
    *   **Drag & Drop**: Upload EEG spectrograms (`.parquet`), `.edf`, or `.bdf` files.
    *   **Instant Results**: Checks for pre-computed results in `predictions.sql` for instant feedback.
    *   **Visualization**: Interactive bar charts of sleep stage distribution (Stage 1, 2, 3, REM, Wake).
*   **Bi-lingual UI**: Full Spanish support (default) and English.

## 📂 Project Structure

- **`app.py`**: The main Streamlit web application.
- **`analyzer.py`**: Backend logic for scanning checkpoint files and caching metadata for performance.
- **`inference.py`**: Core inference logic using PyTorch and Timm models.
- **`batch_inference.py`**: Script for high-performance batch processing of new files.
- **`generate_report.py`**: Generates a static Markdown report of all checkpoints.

## 🚀 Usage

### 1. Run the Dashboard
```bash
streamlit run app.py
```
Expected URL: `http://localhost:8501`

### 2. Checkpoint Management
*   **Folder Location**: Place your `.ckpt` files in `checkpoint_files/`.
*   **Scanning**: Click "Refresh" in the sidebar to scan for new models.

### 3. Inference
*   Go to the **"Carga y resultados"** tab.
*   Upload a `.parquet`, `.edf`, or `.bdf` file.
*   Results (Sleep Stage labels and counts) will appear automatically.

## 🛠 Requirements

*   Python 3.11+
*   Dependencies:
    ```bash
    pip install torch pandas numpy timm streamlit pytorch_lightning altair
    ```

## 📝 Notes
*   **EDF/BDF Support**: Full support for `.edf` and `.bdf` files. The app automatically converts uploaded `.edf`/`.bdf` files to `.parquet` (spectrograms) using the integrated preprocessing module (implicitly scaling BDF units by 1e6 to match uV).
*   **Git**: Large files (`.ckpt`, `.parquet`, `.edf`) are excluded via `.gitignore`.
