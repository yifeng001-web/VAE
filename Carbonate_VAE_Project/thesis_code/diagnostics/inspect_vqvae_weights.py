"""Quick check VQ-VAE checkpoint h5 structure."""

from thesis_code.config import PROJECT_ROOT
import h5py
import os

ckpt_path = os.path.join(PROJECT_ROOT,
                         'results', 'checkpoints', 'encoder.weights.h5')
print(f"File: {ckpt_path}")
print(f"Size: {os.path.getsize(ckpt_path)} bytes")

with h5py.File(ckpt_path, 'r') as f:
    def print_structure(name, obj):
        if hasattr(obj, 'shape'):
            print(f'  {name:80s} shape={obj.shape}')
    print("\n=== Encoder h5 structure ===")
    f.visititems(print_structure)
