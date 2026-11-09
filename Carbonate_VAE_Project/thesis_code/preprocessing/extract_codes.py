"""
[Stage 1] Extract Global Codebook Indices
Loads VQ-VAE Epoch 50 weights, encodes all 4000+ carbonate CT slices,
extracts discrete 32x32 index maps, and saves as latent_codes.npy.

Firewalls:
  - GPU check: tf.config.list_physical_devices('GPU') must be non-empty
  - Data count: os.walk TIF count must be > 4000
"""

import os
import sys
import numpy as np
import tensorflow as tf

# ===== FIREWALL: GPU check =====
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected! Training cannot proceed.")
print(f"[FIREWALL] GPU check PASSED: {len(gpus)} GPU(s) available")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from thesis_code.models.vae_model import build_encoder, VectorQuantizer

from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
OUTPUT_PATH = os.path.join(CHECKPOINT_DIR, "latent_codes.npy")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
BATCH_SIZE = 16
IMAGE_SIZE = 256


def get_indices(vq_layer, z_e):
    """
    Extract discrete encoding indices from a VectorQuantizer layer
    without triggering loss computation. Returns (batch, h, w) int32.
    """
    input_shape = tf.shape(z_e)
    flattened = tf.reshape(z_e, [-1, LATENT_DIM])

    similarity = tf.matmul(flattened, vq_layer.embeddings)
    flat_sq = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(tf.reduce_sum(vq_layer.embeddings ** 2, axis=0), [1, -1])
    distances = flat_sq + emb_sq - 2.0 * similarity

    encoding_indices = tf.argmin(distances, axis=1)  # (N,)
    encoding_indices = tf.reshape(encoding_indices, input_shape[:3])  # (b, h, w)
    return encoding_indices


def main():
    # ===== FIREWALL: Data count check =====
    TARGET_EXTENSIONS = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXTENSIONS:
                file_list.append(os.path.join(root, f))

    num_files = len(file_list)
    print(f"[FIREWALL] TIF files found via os.walk: {num_files}")
    if num_files < 4000:
        raise ValueError(
            f"[FATAL] File count {num_files} < 4000! Aborting to prevent fake training."
        )
    print(f"[FIREWALL] Data count check PASSED: {num_files} files\n")

    # ── Load model ──
    print("[INFO] Building encoder and loading VQ weights...")
    encoder = build_encoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    # Init graph
    dummy = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])
    _ = vq_layer(encoder(dummy))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])
    print(f"[INFO] Epoch {EPOCH_TO_LOAD} weights loaded.\n")

    # ── Load data with PIL bridge ──
    from PIL import Image

    # Build a tf.data pipeline for efficient processing
    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    def load_and_preprocess(path):
        def _decode(p):
            p_str = p.numpy().decode('utf-8')
            img = Image.open(p_str)
            arr = np.array(img, dtype=np.float32)
            if len(arr.shape) == 2:
                arr = np.expand_dims(arr, axis=-1)
            return arr

        img = tf.py_function(func=_decode, inp=[path], Tout=tf.float32)
        img.set_shape([None, None, 1])
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        # Min-Max normalize (matching training)
        min_val = tf.reduce_min(img)
        max_val = tf.reduce_max(img)
        img = tf.math.divide_no_nan(img - min_val, max_val - min_val)
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
        return img

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # ── Extract indices ──
    all_indices = []
    processed = 0

    print(f"[INFO] Extracting indices from {num_files} images...")
    for batch in dataset:
        z_e = encoder(batch, training=False)
        indices = get_indices(vq_layer, z_e)
        all_indices.append(indices.numpy().astype(np.int32))
        processed += batch.shape[0]
        if processed % 200 == 0:
            print(f"  Progress: {processed}/{num_files}")

    all_indices = np.concatenate(all_indices, axis=0)
    print(f"\n[INFO] Total indices extracted: {all_indices.shape} (N, 32, 32)")
    print(f"[INFO] Index range: [{all_indices.min()}, {all_indices.max()}]")

    # ── Save ──
    np.save(OUTPUT_PATH, all_indices)
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"[DONE] Saved to: {OUTPUT_PATH}")
    print(f"[DONE] File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
