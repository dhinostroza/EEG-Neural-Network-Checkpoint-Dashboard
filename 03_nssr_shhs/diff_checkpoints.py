import torch
import os

base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/2000 files"
f1 = os.path.join(base_dir, "2026-01-10_Modelo_Definitivo_Fusionado.ckpt")
f2 = os.path.join(base_dir, "2026-01-11_Modelo_Fusionado_Recalibrated.ckpt")

print(f"Loading {os.path.basename(f1)}...")
c1 = torch.load(f1, map_location='cpu')
s1 = c1.get('state_dict', c1)

print(f"Loading {os.path.basename(f2)}...")
c2 = torch.load(f2, map_location='cpu')
s2 = c2.get('state_dict', c2)

# Find a BN running mean key
print("Sample keys from ckpt 1:", list(s1.keys())[:10])
bn_key = None
for k in s1.keys():
    if 'running_mean' in k:
        bn_key = k
        break
        
if bn_key:
    print(f"Comparing key: {bn_key}")
    val1 = s1[bn_key]
    val2 = s2[bn_key]
    
    print(f"Stats 1 (First 5): {val1[:5]}")
    print(f"Stats 2 (First 5): {val2[:5]}")
    
    if torch.allclose(val1, val2):
        print("FAIL: Stats Identical!")
    else:
        print("SUCCESS: Stats Changed!")
else:
    print("No BN keys found ??")
