
import scipy.io
import os

file_path = "01_matlab_eeg/processed_data/Stage1_CNN_Training_Results_SleepEDFX_SC40_processed_parallel.mat_Sequential.mat"

try:
    data = scipy.io.loadmat(file_path)
    print(f"Keys: {data.keys()}")
    
    if 'results_cnn' in data:
        res = data['results_cnn']
        print(f"results_cnn shape: {res.shape}")
        if res.shape[0] > 0:
            item = res[0,0]
            print(f"Item fields: {item.dtype.names}")
            
            # Check for Confusion Matrix keywords
            for field in item.dtype.names:
                print(f" - {field}")
                if 'conf' in field.lower() or 'mat' in field.lower():
                     print(f"   Potential match: {field}")
                     
except Exception as e:
    print(f"Error: {e}")
