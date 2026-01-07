from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import scipy.io
import numpy as np
import h5py
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Tesis API is running"}

@app.get("/api/project/01/results")
async def get_project_01_results(stage: int = 1):
    logger.info(f"Received request for stage: {stage}")
    try:
        is_dataset = False
        file_path = ""
        root_key = ""
        info_key = ""
        dataset_meta_path = "" # Path to dataset file for metadata (subjects/epochs)

        # Map stages to files
        # Map stages to files
        if stage == 8: # Dataset: Sleep-edf-8
            file_path = "01_matlab_eeg/processed_data/SleepEDFX8_processed_parallel.mat"
            is_dataset = True
            dataset_meta_path = file_path
        elif stage == 20: # Dataset: Sleep-edfx-78 (20 subjects)
            file_path = "01_matlab_eeg/processed_data/SleepEDFX20_processed_parallel.mat"
            is_dataset = True
            dataset_meta_path = file_path
        elif stage == 41: # Experiment 40: CNN
            file_path = "01_matlab_eeg/processed_data/Stage1_CNN_Training_Results_SleepEDFX_SC40_processed_parallel.mat_Sequential.mat"
            root_key = 'results_cnn'
            info_key = 'TrainInfo'
            dataset_meta_path = "01_matlab_eeg/processed_data/SleepEDFX_SC40_processed_parallel.mat"
        elif stage == 42: # Experiment 40: LSTM
            file_path = "01_matlab_eeg/processed_data/Stage2_LSTM_Training_Results_SleepEDFX_SC40_processed_parallel.mat.mat"
            root_key = 'resultsOverall_S2' # Use Overall results for CM
            info_key = 'TrainInfoLSTM' # Fallback for training curve? Or use results_stage2?
            # Note: Stage 2 file has 'resultsOverall_S2' (ConfMat) and 'results_stage2' (TrainInfoLSTM per subject)
            dataset_meta_path = "01_matlab_eeg/processed_data/SleepEDFX_SC40_processed_parallel.mat"
        elif stage == 78: # Placeholder
            return {
                "status": "success", 
                "type": "placeholder",
                "stage": 78,
                "data": {"message": "Data for 78 subjects not yet processed.", "subject_count": 78}
            }
        else:
             # Fallback/Legacy
             return {"status": "error", "message": "Invalid stage ID."}

        # 1. DATASET METADATA (Subjects/Epochs)
        # We try to fetch this for BOTH Experiments and Datasets
        meta_info = {}
        if dataset_meta_path:
             try:
                abs_meta_path = os.path.abspath(dataset_meta_path)
                if os.path.exists(abs_meta_path):
                    with h5py.File(abs_meta_path, 'r') as f:
                        if 'all_labels' in f:
                            dset = f['all_labels']
                            meta_info['sample_count'] = dset.shape[1] if len(dset.shape) > 1 else dset.shape[0]
                        if 'processed_subject_info' in f:
                            dset = f['processed_subject_info']
                            meta_info['subject_count'] = dset.shape[1] if len(dset.shape) > 1 else dset.shape[0]
                        if 'all_spectrograms' in f:
                            meta_info['dims'] = list(f['all_spectrograms'].shape)
             except Exception as e:
                 logger.warning(f"Could not read metadata from {dataset_meta_path}: {e}")

        # If it's PURE dataset view, return now
        # Check if Python Results exist for Stage 8
        if stage == 8:
            # Use absolute path to avoid CWD ambiguity
            # v1.3 is now the PRIMARY result we look for
            python_res_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/python_results/results_8_subjects_python_v1_3.json"
            if not os.path.exists(python_res_path):
                 # Fallback to v1.2 if v1.3 not yet ready
                 python_res_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/python_results/results_8_subjects_python_v1_2.json"
            
            if os.path.exists(python_res_path):
                import json
                with open(python_res_path, 'r') as f:
                    py_data = json.load(f)
                 
                # Format for Dashboard
                acc_flat = []
                loss_flat = []
                cm = [] # Initialize safeguards
                # Combine folds for chart? Or just show Fold 1?
                # Let's show Fold 1 for simplicity of the chart or average?
                # Dashboard expects simple list. Concatenating epochs from all folds is confusing.
                # Let's show Fold 1 training curve.
                if len(py_data['folds']) > 0:
                    # Show Fold 1 training curve (representative)
                    f1 = py_data['folds'][0]
                    acc_flat = f1['train_acc']
                    loss_flat = f1['train_loss']
                    
                    # Confusion Matrix: AGGREGATE across all folds for Global v1.0 Performance
                    try:
                        # Stack all CMs and sum them
                        all_cms = [np.array(f['confusion_matrix']) for f in py_data['folds']]
                        cm_sum = np.sum(all_cms, axis=0)
                        cm = cm_sum.tolist()
                    except Exception as e:
                        logger.warning(f"Error aggregating CMs: {e}")
                        cm = f1['confusion_matrix'] # Fallback

                # --- Load Previous Versions (v1.2 BCW) ---
                cm_v1_2 = None
                v1_2_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/python_results/results_8_subjects_python_v1_2.json"
                if os.path.exists(v1_2_path):
                    try:
                        with open(v1_2_path, 'r') as fv12:
                            v1_2_data = json.load(fv12)
                            v1_2_cms = [np.array(f['confusion_matrix']) for f in v1_2_data['folds']]
                            cm_v1_2 = np.sum(v1_2_cms, axis=0).tolist()
                    except Exception as e:
                        logger.warning(f"Could not load v1.2: {e}")

                # --- Load Baseline v1.0 (if exists) ---
                baseline_cm = None
                v1_path = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/01_matlab_eeg/processed_data/python_results/results_8_subjects_v1_0.json"
                if os.path.exists(v1_path):
                    try:
                        with open(v1_path, 'r') as fv1:
                            v1_data = json.load(fv1)
                            # Aggregate CM for v1.0
                            v1_cms = [np.array(f['confusion_matrix']) for f in v1_data['folds']]
                            v1_cm_sum = np.sum(v1_cms, axis=0)
                            baseline_cm = v1_cm_sum.tolist()
                    except Exception as e:
                        logger.warning(f"Could not load v1.0 baseline: {e}")

                return {
                    "status": "success",
                    "type": "experiment", # Treat as experiment now!
                    "stage": stage,
                    "data": {
                        "subject_count": 8,
                        "sample_count": meta_info.get('sample_count', 15206),
                        "dims": meta_info.get('dims', [3, 60, 76]),
                        "accuracy": acc_flat,
                        "loss": loss_flat,
                        "total_iterations": len(acc_flat),
                        "final_accuracy": py_data['average_acc'],
                        "final_accuracy": py_data['average_acc'],
                        "confusion_matrix": cm, # Defaults to v1.3 (Current)
                        "baseline_confusion_matrix": baseline_cm,
                        "confusion_matrix_v1_2": cm_v1_2, # Explicit v1.2 for comparison
                        "class_names": ['W', 'N1', 'N2', 'N3', 'REM'],
                        # Advanced Metrics - Handle both intermediate (kappa) and final (average_kappa) keys
                        "kappa": py_data.get('kappa') or py_data.get('average_kappa', 0),
                        "f1": py_data.get('f1') or py_data.get('average_f1', 0),
                        "precision": py_data.get('precision') or py_data.get('average_precision', 0),
                        "recall": py_data.get('recall') or py_data.get('average_recall', 0),
                        "progress": py_data.get('progress', '')
                    },
                    "message": "Python MPS Results Loaded"
                }

        if is_dataset:
            return {
                "status": "success",
                "type": "dataset",
                "id": stage,
                "data": meta_info,
                "message": "Dataset metadata extracted"
            }

        # 2. EXPERIMENT RESULTS (SCIPY)
        try:
            abs_path = os.path.abspath(file_path)
            data = scipy.io.loadmat(abs_path)
        except Exception as e:
            return {"status": "error", "message": f"Error loading mat: {str(e)}"}

        response_data = meta_info # Start with subject/count info

        if stage == 42: # LSTM - Logic for ConfMat
             if 'resultsOverall_S2' in data:
                 overall = data['resultsOverall_S2'] # Shape (1,1) struct
                 if overall.shape == (1,1):
                     ov = overall[0,0]
                     if 'ConfMat' in ov.dtype.names:
                         response_data['confusion_matrix'] = ov['ConfMat'].tolist()
                     if 'Accuracy' in ov.dtype.names:
                         # Single value
                         response_data['final_accuracy'] = float(ov['Accuracy'].flatten()[0])
                     if 'ClassNames' in ov.dtype.names:
                         # might be array of strings
                         try:
                             # Handle cell array or array of chars
                             raw_names = ov['ClassNames'][0]
                             response_data['class_names'] = [str(c[0]) for c in raw_names]
                         except:
                             response_data['class_names'] = ['W', 'N1', 'N2', 'N3', 'REM'] # Default

             # Also try to get Training Curve from 'results_stage2' (first subject)
             if 'results_stage2' in data:
                 rs2 = data['results_stage2']
                 if rs2.shape[0] > 0 and 'TrainInfoLSTM' in rs2.dtype.names:
                     ti = rs2[0,0]['TrainInfoLSTM']
                     if ti.size > 0:
                         t = ti[0,0]
                         if 'TrainingAccuracy' in t.dtype.names:
                             acc = t['TrainingAccuracy'].flatten()
                             step = max(1, len(acc) // 1000)
                             response_data['accuracy'] = acc[::step].tolist()
                             response_data['total_iterations'] = len(acc)
                         if 'BaseLearnRate' in t.dtype.names:
                             lr = t['BaseLearnRate'].flatten()
                             step = max(1, len(lr) // 1000)
                             response_data['learn_rate'] = lr[::step].tolist()
                         if 'TrainingLoss' in t.dtype.names:
                             loss = t['TrainingLoss'].flatten()
                             step = max(1, len(loss) // 1000)
                             response_data['loss'] = loss[::step].tolist()

             return {
                "status": "success",
                "type": "experiment",
                "stage": stage,
                "data": response_data,
                "message": "LSTM Data extracted"
             }

        elif stage == 41: # CNN
            if root_key in data:
                results = data[root_key]
                if results.shape[0] > 0 and info_key in results.dtype.names:
                     train_info = results[0,0][info_key]
                     if train_info.size > 0:
                         ti = train_info[0,0]
                         if 'TrainingAccuracy' in ti.dtype.names:
                             acc = ti['TrainingAccuracy'].flatten()
                             step = max(1, len(acc) // 1000)
                             response_data['accuracy'] = acc[::step].tolist()
                             response_data['total_iterations'] = len(acc)
                         if 'BaseLearnRate' in ti.dtype.names:
                             lr = ti['BaseLearnRate'].flatten()
                             step = max(1, len(lr) // 1000)
                             response_data['learn_rate'] = lr[::step].tolist()
                         if 'TrainingLoss' in ti.dtype.names:
                             loss = ti['TrainingLoss'].flatten()
                             step = max(1, len(loss) // 1000)
                             response_data['loss'] = loss[::step].tolist()
            
            return {
                "status": "success",
                "type": "experiment",
                "stage": stage,
                "data": response_data,
                "message": "CNN Data extracted"
            }

        return {"status": "error", "message": "Unknown stage configuration"}

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/upload-template")
async def upload_template(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "success"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
