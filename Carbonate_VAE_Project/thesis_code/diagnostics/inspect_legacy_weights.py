
from thesis_code.config import PROJECT_ROOT
import os
import h5py, numpy as np
f = h5py.File(os.path.join(PROJECT_ROOT, "results", "checkpoints", "vae_carbonate.weights.h5"), 'r')

# Check actual paths - the internal structure
print("=== Full recursive listing ===")
def all_paths(name, obj):
    path = name
    if hasattr(obj, 'shape'):
        d = obj[:]
        print(f"  {path}  shape={d.shape}  mean={d.mean():.6f}  std={d.std():.6f}")
    else:
        print(f"  {path}  (group)")
f.visititems(all_paths)
f.close()
