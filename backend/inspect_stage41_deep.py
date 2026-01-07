
import scipy.io
import os
import numpy as np

file_path = "01_matlab_eeg/processed_data/Stage1_CNN_Training_Results_SleepEDFX_SC40_processed_parallel.mat_Sequential.mat"

try:
    data = scipy.io.loadmat(file_path)
    if 'results_cnn' in data:
        res = data['results_cnn']
        if res.shape[0] > 0:
            item = res[0,0]
            # Inspect TrainInfo
            train_info = item['TrainInfo']
            print(f"TrainInfo shape: {train_info.shape}")
            if train_info.size > 0:
                ti = train_info[0,0]
                print(f"TrainInfo fields: {ti.dtype.names}")
except Exception as e:
    print(f"Error: {e}")
