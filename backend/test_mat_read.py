
import h5py
import scipy.io
import os

files = [
    "../01_matlab_eeg/processed_data/SleepEDFX8_processed_parallel.mat",
    "../01_matlab_eeg/processed_data/SleepEDFX20_processed_parallel.mat",
    "../01_matlab_eeg/processed_data/Stage1_CNN_Training_Results_SleepEDFX_SC40_processed_parallel.mat_Sequential.mat"
]

for fpath in files:
    print(f"Testing {fpath}...")
    abs_path = os.path.abspath(fpath)
    if not os.path.exists(abs_path):
        print(f"  FILE NOT FOUND: {abs_path}")
        continue
        
    # Try HDF5
    try:
        with h5py.File(abs_path, 'r') as f:
            print("  SUCCESS using h5py (v7.3)")
            print("  Keys:", list(f.keys()))
    except OSError:
        print("  FAILED using h5py. Not a valid HDF5 file.")
        
        # Try Scipy
        try:
            mat = scipy.io.loadmat(abs_path)
            print("  SUCCESS using scipy.io.loadmat (v7 or earlier)")
            print("  Keys:", list(mat.keys())[:5])
            if 'results_cnn' in mat:
                print("  Found 'results_cnn'. Checking content...")
                val = mat['results_cnn']
                print("  Type:", type(val), "Shape:", val.shape)
                print("  Dtype names:", val.dtype.names)
                if val.shape == (1,1):
                     struct = val[0,0]
                     if 'TrainInfo' in struct.dtype.names:
                         print("  Found 'TrainInfo'.")
                         info = struct['TrainInfo']
                         print("  TrainInfo Dtypes:", info.dtype.names)
            
        except Exception as e:
            print(f"  FAILED using scipy.io.loadmat: {e}")
            
    print("-" * 20)
