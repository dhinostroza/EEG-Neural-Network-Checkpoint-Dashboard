import scipy.io
import numpy as np

file_cnn = "../01_matlab_eeg/processed_data/Stage1_CNN_Training_Results_SleepEDFX_SC40_processed_parallel.mat_Sequential.mat"
file_lstm = "../01_matlab_eeg/processed_data/Stage2_LSTM_Training_Results_SleepEDFX_SC40_processed_parallel.mat.mat"

try:
    data = scipy.io.loadmat(file_cnn)
    print("--- CNN TrainInfo ---")
    if 'results_cnn' in data:
        ti = data['results_cnn'][0,0]['TrainInfo'][0,0]
        print("Keys:", ti.dtype.names)
except Exception as e:
    print("CNN Error:", e)

try:
    data = scipy.io.loadmat(file_lstm)
    print("\n--- LSTM TrainInfo ---")
    if 'results_stage2' in data:
        ti = data['results_stage2'][0,0]['TrainInfoLSTM'][0,0]
        print("Keys:", ti.dtype.names)
except Exception as e:
    print("LSTM Error:", e)
