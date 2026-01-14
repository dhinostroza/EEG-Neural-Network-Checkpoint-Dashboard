import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np

# Add path to import inference util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import get_model, load_checkpoint_weights, detect_architecture
from ensemble_logic import load_ensemble_models

# --- Wrapper Class ---
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        # Use ModuleList so they are registered properly
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        # Accumulate probabilities
        # AVOID torch.zeros to prevent device hardcoding in Trace!
        
        # 1. First Model
        logits_0 = self.models[0](x)
        prob_sum = F.softmax(logits_0, dim=1)
        
        # 2. Remaining Models
        # Iterate manually or slice if ModuleList supports it in Script
        # ModuleList iteration is supported in Trace/Script
        for i, model in enumerate(self.models):
            if i == 0:
                continue
            logits = model(x)
            prob_sum += F.softmax(logits, dim=1)
                
        # Average
        avg_probs = prob_sum / len(self.models)
        return avg_probs

def main():
    base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs"
    ckpt_dir = os.path.join(base_dir, "checkpoint_files/2000 files")
    
    ckpts = [
        os.path.join(ckpt_dir, "2025-09-04_05-36_convnext_base_2000files_lr2e-05_cwN1-8.0_workers2.ckpt"),
        os.path.join(ckpt_dir, "2025-09-09_15-43_convnext_base_2000files_Augmented_cwN1-6.5.ckpt"),
        os.path.join(ckpt_dir, "2025-09-19_04-34_convnext_base_consolidated_cwN1-6.5-epoch=3-val_loss=0.5971.ckpt")
    ]
    
    print("Loading models for TorchScript export...")
    models = load_ensemble_models(ckpts)
    
    # Wrap
    ensemble = EnsembleModel(models)
    ensemble.eval()
    
    # Dummy Input
    # Batch size 1 or 64, whatever. Tracing records the graph.
    dummy_input = torch.randn(1, 1, 76, 60)
    
    output_path = os.path.join(base_dir, "ensemble_model_scripted.pt")
    
    print(f"Tracing and Exporting to {output_path}...")
    
    try:
        # Use Trace
        traced_script_module = torch.jit.trace(ensemble, dummy_input)
        traced_script_module.save(output_path)
        print("Export Success!")
        
        # Verify load
        print("Verifying reload...")
        loaded_model = torch.jit.load(output_path)
        loaded_model.eval()
        
        with torch.no_grad():
            out_trace = loaded_model(dummy_input)
            out_orig = ensemble(dummy_input)
            
        np.testing.assert_allclose(out_trace.numpy(), out_orig.numpy(), rtol=1e-05, atol=1e-06)
        print("Verification Passed! outputs match.")
        
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"Model Size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Export Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
