# EEG Neural Network Checkpoint Dashboard

A Streamlit-based dashboard for managing, analyzing, and running inference with PyTorch Lightning checkpoints trained on SHHS EEG data.

## 🌟 Features

*   **Model Registry**: Automatically scans and lists trained `.ckpt` models, extracting metadata (Validation Loss, Architecture, Parameters, Epochs) directly from filenames and checkpoints.
*   **Performance Tracking**: Highlights high-performing models (Green for good, Red for poor) based on validation loss.
*   **Inference Playground**:
    *   **Drag & Drop**: Upload EEG spectrograms (`.parquet`) or `.edf` files.
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
*   Upload a `.parquet` or `.edf` file.
*   Results (Sleep Stage labels and counts) will appear automatically.

## 🛠 Requirements

*   Python 3.11+
*   Dependencies:
    ```bash
    pip install torch pandas numpy timm streamlit pytorch_lightning altair
    ```

## 📝 Notes
*   **EDF Support**: Full support for `.edf` files. The app automatically converts uploaded `.edf` files to `.parquet` (spectrograms) using the integrated preprocessing module.
*   **Git**: Large files (`.ckpt`, `.parquet`, `.edf`) are excluded via `.gitignore`.
