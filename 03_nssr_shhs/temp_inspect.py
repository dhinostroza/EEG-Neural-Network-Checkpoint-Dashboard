import torch
import sys
import os

def inspect(path):
    print(f"--- Inspecting {os.path.basename(path)} ---")
    try:
        data = torch.load(path, map_location='cpu')
        
        # Hyperparameters
        if 'hyper_parameters' in data:
            print("Hyperparameters:")
            for k, v in data['hyper_parameters'].items():
                print(f"  {k}: {v}")
        
        # Epoch
        print(f"Epoch: {data.get('epoch', 'N/A')}")
        
        # Callbacks (Val Loss)
        if 'callbacks' in data:
            for cb_key, cb_val in data['callbacks'].items():
                if 'ModelCheckpoint' in cb_key:
                    print(f"  {cb_key}: {cb_val}")
                    
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            inspect(f)
