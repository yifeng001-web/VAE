
from thesis_code.config import PROJECT_ROOT
import os
import h5py

file_path = os.path.join(PROJECT_ROOT, "results", "checkpoints", "vae_carbonate.weights.h5")
with h5py.File(file_path, 'r') as f:
    # 打印最外层的组
    print("文件内的主要结构：", list(f.keys()))
    
    # 递归打印所有层级
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)