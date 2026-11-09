"""Report dependencies and TensorFlow devices without changing the environment."""

import importlib.metadata
import importlib.util
import sys

from thesis_code.config import DATA_DIR, PROJECT_ROOT


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"CT data: {DATA_DIR}")
    dependencies = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "scipy": "scipy",
        "skimage": "scikit-image",
        "matplotlib": "matplotlib",
        "h5py": "h5py",
        "tifffile": "tifffile",
        "tensorflow": "tensorflow",
    }
    missing = []
    for module, distribution in dependencies.items():
        if importlib.util.find_spec(module) is None:
            missing.append(distribution)
            print(f"{distribution}: MISSING")
        else:
            print(f"{distribution}: {importlib.metadata.version(distribution)}")
    if missing:
        print("Install dependencies: python -m pip install -r requirements.txt")
        return 1

    import tensorflow as tf

    print(f"TensorFlow GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    if importlib.util.find_spec("torch") is not None:
        print(f"PyTorch (optional): {importlib.metadata.version('torch')}")
    else:
        print("PyTorch (optional): not installed; required for GPT experiments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
