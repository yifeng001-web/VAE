"""
[批量随机密度生成] Random-Density Batch Digital Rock Generation
=================================================================
Generates 16 highly diverse digital carbonate rock slices via:
  1. Random Matrix_A / Pore_B pair selection (from pool of 32)
  2. Histogram matching (unified background color)
  3. Random organic cloud masks (sigma, coverage threshold)
  4. Continuous latent space fusion (Z_e, before VQ) — zero block artifacts

Algorithm per slice:
  1. Pick random A ≠ B from 32-image pool
  2. Histogram-match B → A (eliminate background mismatch)
  3. Encode → Z_eA, Z_eB (continuous, BEFORE VQ)
  4. Random sigma ∈ [2.0, 5.0] → pore blob size
  5. Random coverage ∈ [60%, 90%] → pore density
  6. Generate organic mask → Z_fused → VQ → Decode

Output: 4×4 scientific grid at results/evaluation/random_batch_rocks.png

Firewalls: GPU, checkpoints, skimage, scipy
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# ===== FIREWALL: GPU check =====
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected!")
print(f"[FIREWALL] GPU check PASSED: {len(gpus)} GPU(s) available")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

# ===== Dependency checks =====
try:
    from skimage.exposure import match_histograms
except ImportError:
    raise ImportError("[FATAL] scikit-image not installed! Run: pip install scikit-image")
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    raise ImportError("[FATAL] scipy not installed!")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

# ===== Paths =====
from thesis_code.config import PROJECT_ROOT
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
os.makedirs(EVAL_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(EVAL_DIR, "fixed_sigma_batch_rocks.png")

# ===== Constants =====
NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
LATENT_SIZE = 32
IMAGE_SIZE = 256
POOL_SIZE = 32       # load 32 random real slices as the gene pool
BATCH_ROWS = 4
BATCH_COLS = 4
NUM_GENERATED = BATCH_ROWS * BATCH_COLS  # 16

# Mask parameter ranges
FIXED_SIGMA = 4.0                            # pore blob size (fixed for all 16)
SIGMA_MIN, SIGMA_MAX = 2.0, 5.0             # pore blob size (unused in fixed mode)
COVERAGE_MIN, COVERAGE_MAX = 0.60, 0.90     # pore density percentile
SOFTEN_SIGMA = 1.0                           # edge softening (fixed)


def load_tif_manual(filepath):
    """Decode single TIF via PIL, resize 256x256, Min-Max normalize."""
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t)
    max_v = tf.reduce_max(img_t)
    img_n = tf.math.divide_no_nan(img_t - min_v, max_v - min_v)
    return img_n.numpy()  # (256, 256, 1)


def load_pool(data_dir, pool_size=POOL_SIZE):
    """Randomly load pool_size TIFs from data_dir as the gene pool."""
    TARGET_EXTENSIONS = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXTENSIONS:
                file_list.append(os.path.join(root, f))

    print(f"[POOL] Found {len(file_list)} TIFs total, sampling {pool_size}...")
    np.random.shuffle(file_list)
    sampled = file_list[:pool_size]

    pool = []
    means = []
    for fp in sampled:
        img_np = load_tif_manual(fp)
        pool.append(img_np)
        means.append(img_np.mean())

    means = np.array(means)
    print(f"[POOL] Pool means: min={means.min():.4f}, max={means.max():.4f}, "
          f"mean={means.mean():.4f}")
    return pool, means


def generate_organic_mask(h, w, cloud_sigma, coverage_threshold, soften_sigma=SOFTEN_SIGMA):
    """
    Generate an organic cloud mask with controlled pore density.

    Args:
        h, w: latent space dimensions (32, 32)
        cloud_sigma: Gaussian blur sigma for blob size (larger → bigger blobs)
        coverage_threshold: percentile threshold in [0.6, 0.9]
            HIGHER threshold → fewer pixels above → SPARSER pores
            LOWER threshold  → more pixels above  → DENSER pores
        soften_sigma: edge softening sigma

    Returns:
        matrix_mask: (1, h, w, 1) float32, values in [0,1]
            1.0 = matrix parent (Z_eA), 0.0 = pore donor (Z_eB)
        pore_mask: (1, h, w, 1) float32, 1.0 = pore region
    """
    # Step 1: Random noise
    noise = np.random.randn(h, w).astype(np.float32)

    # Step 2: Large sigma blur → organic cloud blobs
    cloud = gaussian_filter(noise, sigma=cloud_sigma)

    # Step 3: Binarize at the given percentile threshold
    thresh_val = np.percentile(cloud, coverage_threshold * 100.0)
    pore_binary = (cloud > thresh_val).astype(np.float32)

    # Step 4: Soften hard edges
    pore_soft = gaussian_filter(pore_binary, sigma=soften_sigma)

    # pore_mask: 1.0 = pore donor region (Z_eB), 0.0 = matrix (Z_eA)
    pore_mask = pore_soft[np.newaxis, :, :, np.newaxis]
    matrix_mask = 1.0 - pore_mask

    return matrix_mask, pore_mask


def density_label(coverage):
    """Convert coverage threshold to human-readable density label."""
    if coverage >= 0.82:
        return "Low"
    elif coverage >= 0.72:
        return "Med"
    else:
        return "High"


def main():
    print("=" * 60)
    print("  Random-Density Batch Digital Rock Generation")
    print("=" * 60 + "\n")

    # ===== FIREWALL: Check checkpoints =====
    for fname in [f"encoder_ep{EPOCH_TO_LOAD}.weights.h5",
                  f"decoder_ep{EPOCH_TO_LOAD}.weights.h5",
                  f"vq_layer_ep{EPOCH_TO_LOAD}.npy"]:
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"[FATAL] {fname} not found!")
    print("[FIREWALL] All checkpoints present.\n")

    # ── Load models ──
    print("[INFO] Loading Encoder, VQ codebook, and Decoder...")
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])
    print("[INFO] All models loaded.\n")

    # ── Load random image pool ──
    print("[INFO] Loading random gene pool...")
    pool, pool_means = load_pool(DATA_DIR, POOL_SIZE)

    # ── Generate 16 diverse rocks ──
    print(f"\n[INFO] Generating {NUM_GENERATED} random-density rock slices...\n")

    generated_images = []
    meta_info = []  # store (sigma, coverage, label) for each

    for i in range(NUM_GENERATED):
        # Randomly pick A (matrix parent) and B (pore donor), ensuring A ≠ B
        idx_a, idx_b = np.random.choice(POOL_SIZE, size=2, replace=False)
        img_A = pool[idx_a]  # (256, 256, 1)
        img_B = pool[idx_b]  # (256, 256, 1)

        # Fixed sigma + random coverage
        sigma = FIXED_SIGMA
        coverage = np.random.uniform(COVERAGE_MIN, COVERAGE_MAX)

        # ── Histogram matching: B → A ──
        img_B_matched = match_histograms(
            img_B[:, :, 0], img_A[:, :, 0]
        ).astype(np.float32)
        img_B_matched = np.clip(img_B_matched, 0.0, 1.0)
        img_B_matched = np.expand_dims(img_B_matched, axis=-1)  # (256, 256, 1)

        # ── Encode to continuous Z_e ──
        z_eA = encoder(tf.constant(np.expand_dims(img_A, 0)), training=False)
        z_eB = encoder(tf.constant(np.expand_dims(img_B_matched, 0)), training=False)

        # ── Generate organic mask ──
        matrix_mask, pore_mask = generate_organic_mask(
            LATENT_SIZE, LATENT_SIZE, cloud_sigma=sigma,
            coverage_threshold=coverage
        )

        # ── Continuous latent space fusion ──
        # Z_novel = Z_eA × matrix_mask + Z_eB × pore_mask
        z_fused = z_eA * tf.constant(matrix_mask, dtype=tf.float32) + \
                  z_eB * tf.constant(pore_mask, dtype=tf.float32)

        # VQ quantize + decode
        z_q = vq_layer(z_fused, training=False)
        generated = decoder(z_q, training=False)
        img_gen = np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)
        generated_images.append(img_gen)

        label = density_label(coverage)
        meta_info.append((sigma, coverage, label))
        pore_frac = pore_mask.mean()
        print(f"  [#{i+1:02d}] sigma={sigma:.2f}, coverage={coverage:.2f} "
              f"({label} density), pore_frac={pore_frac:.2%}")

    # ── Render 4×4 scientific grid ──
    print("\n[INFO] Rendering 4x4 scientific grid...")
    fig, axes = plt.subplots(BATCH_ROWS, BATCH_COLS, figsize=(16, 16))

    for idx, ax in enumerate(axes.flat):
        sigma, coverage, label = meta_info[idx]
        ax.imshow(generated_images[idx], cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(f'#{idx+1} | sigma={sigma:.1f} | {label} Density',
                     fontsize=9, fontweight='bold', color='darkorange')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('darkorange')
            spine.set_linewidth(1.5)

    plt.suptitle(
        f'Batch Digital Rock Generation (Fixed sigma={FIXED_SIGMA:.1f})\n'
        'Histogram-Matched Continuous Latent Fusion (Z_e) — Zero Block Artifacts',
        fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    # ── Stats ──
    sigmas = [m[0] for m in meta_info]
    coverages = [m[1] for m in meta_info]
    print(f"\n[DONE] 4x4 batch saved to:\n  {OUTPUT_PATH}")
    print(f"[STATS] sigma (fixed): {FIXED_SIGMA:.2f}")
    print(f"[STATS] coverage range: [{min(coverages):.2f}, {max(coverages):.2f}]")
    print(f"[STATS] Density distribution: "
          f"High={sum(1 for m in meta_info if m[2]=='High')}, "
          f"Med={sum(1 for m in meta_info if m[2]=='Med')}, "
          f"Low={sum(1 for m in meta_info if m[2]=='Low')}")


if __name__ == "__main__":
    main()
