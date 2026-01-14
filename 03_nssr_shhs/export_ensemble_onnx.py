import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import onnx
import onnxruntime as ort
import numpy as np

# Add path to import inference util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import get_model, load_checkpoint_weights, detect_architecture
from ensemble_logic import load_ensemble_models

# --- 1. Wrapper Class ---
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        # Accumulate probabilities
        prob_sum = None
        
        # Iterate through models in the list
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            
            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum = prob_sum + probs
                
        # Average
        avg_probs = prob_sum / len(self.models)
        return avg_probs

def main():
    base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs"
    ckpt_dir = os.path.join(base_dir, "checkpoint_files/2000 files")
    
    # 1. Define Checkpoints
    ckpts = [
        os.path.join(ckpt_dir, "2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt"),
        os.path.join(ckpt_dir, "2025-09-09_15-43_convnext_base_2000files_Augmented_cwN1-6.5.ckpt"),
        os.path.join(ckpt_dir, "2025-09-19_04-34_convnext_base_consolidated_cwN1-6.5-epoch=3-val_loss=0.5971.ckpt")
    ]
    
    print("Loading models for export...")
    models = load_ensemble_models(ckpts)
    
    # 2. Wrap
    ensemble = EnsembleModel(models)
    ensemble.eval()
    
    # 3. Dummy Input (Batch Size 1, Channel 1, H 76, W 60)
    # Use batch_size=1 because ONNX usually exports with fixed batch size unless dynamic axes are set.
    # We will set dynamic axes for batch size.
    dummy_input = torch.randn(1, 1, 76, 60, requires_grad=False)
    
    output_path = os.path.join(base_dir, "ensemble_model.onnx")
    
    print(f"Exporting to {output_path}...")
    
    try:
        torch.onnx.export(
            ensemble,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            # dynamic_axes={
            #     'input': {0: 'batch_size'},
            #     'output': {0: 'batch_size'}
            # }
        )
        print("Export Success!")
    except Exception as e:
        print(f"Export Failed: {e}")
        return

    # 4. Verify with ONNX Runtime
    print("Verifying with ONNX Runtime...")
    try:
        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Verify shape
        print(f"Output shape: {ort_outs[0].shape}")
        
        # Verify values vs PyTorch
        with torch.no_grad():
            torch_out = ensemble(dummy_input)
            
        np.testing.assert_allclose(torch_out.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Verification Passed! Outputs match.")
        
        # Check file size
        mb_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"ONNX Model Size: {mb_size:.2f} MB")
        
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    main()
