
from thesis_code.config import PROJECT_ROOT
import os
import h5py

file_path = os.path.join(PROJECT_ROOT, "results", "checkpoints", "vae_carbonate.weights.h5")
with h5py.File(file_path, 'r') as f:
    def print_structure(name, obj):
        if hasattr(obj, 'shape'):
            print(f'{name}  shape={obj.shape}')
        else:
            print(f'{name}  (group)')
    
    print("=== ENCODER ===")
    f['encoder'].visititems(print_structure)
    print("\n=== DECODER ===")
    f['decoder'].visititems(print_structure)
    print("\n=== TOP LEVEL ===")
    f['top_level_model_weights'].visititems(print_structure)
