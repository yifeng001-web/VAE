"""
[终极融合生成] Histogram-Matched Continuous Latent Space Fusion
===============================================================
废弃 PixelCNN 离散硬拼接，退回到连续隐空间 Z_e（VQ 量化前）进行柔性基因融合。

核心创新：
  1. 直方图匹配 (Histogram Matching): 将孔隙供体 B 的底色强行对齐到基质母本 A，
     从根本上消除"阴阳脸/背景色差"问题。
  2. 云雾状有机掩膜 (Organic Cloud Mask): 随机噪声 → 大 sigma 高斯模糊 (σ=4.0)
     → 二值化 → 小 sigma 柔化边缘 (σ=1.0)，产生自然过渡带。
  3. 连续隐空间融合: Z_fused = Z_eA × (1-mask) + Z_eB × mask，
     在 VQ 量化前完成，再统一离散化 + 解码，杜绝方块伪影。

Firewalls:
  - GPU check
  - Checkpoint existence
  - skimage/scipy availability
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# ===== FIREWALL: GPU check =====
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected! Generation cannot proceed.")
print(f"[FIREWALL] GPU check PASSED: {len(gpus)} GPU(s) available")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

# ===== Dependency checks =====
try:
    from skimage.exposure import match_histograms
    print("[DEPS] skimage.exposure.match_histograms imported OK")
except ImportError:
    raise ImportError("[FATAL] scikit-image not installed! Run: pip install scikit-image")

try:
    from scipy.ndimage import gaussian_filter
    print("[DEPS] scipy.ndimage.gaussian_filter imported OK")
except ImportError:
    raise ImportError("[FATAL] scipy not installed! Run: pip install scipy")

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

OUTPUT_PATH = os.path.join(EVAL_DIR, "perfect_fusion_generation.png")

# ===== Constants =====
NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
LATENT_SIZE = 32
IMAGE_SIZE = 256
NUM_FUSIONS = 4

# Organic mask params
CLOUD_SIGMA = 4.0      # large sigma → big cloud-like blobs
SOFTEN_SIGMA = 1.0     # small sigma → smooth hard edges after binarization


def load_tif_manual(filepath):
    """Decode a single TIF using PIL, resize to 256×256, Min-Max normalize."""
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t)
    max_v = tf.reduce_max(img_t)
    img_n = tf.math.divide_no_nan(img_t - min_v, max_v - min_v)
    return img_n.numpy()  # (256, 256, 1)


def pick_two_diverse_images(data_dir, num_candidates=64):
    """
    Scan data_dir for TIFs, randomly pick num_candidates, then select
    two images with maximum pairwise mean-intensity difference.
    Returns (img_A, img_B) where A = brighter (matrix-heavy), B = darker (pore-heavy).
    """
    TARGET_EXTENSIONS = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXTENSIONS:
                file_list.append(os.path.join(root, f))

    print(f"[PICK] Found {len(file_list)} TIFs, sampling {num_candidates}...")
    np.random.shuffle(file_list)
    sampled = file_list[:num_candidates]

    # Load all candidates, compute means
    images = []
    means = []
    for fp in sampled:
        img_np = load_tif_manual(fp)
        images.append(img_np)
        means.append(img_np.mean())

    means = np.array(means)
    # Pairwise absolute differences
    diff_matrix = np.abs(means[:, None] - means[None, :])
    # Find the pair with maximum difference
    idx_A, idx_B = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
    # A = brighter (higher mean), B = darker (more pores)
    if means[idx_A] < means[idx_B]:
        idx_A, idx_B = idx_B, idx_A

    img_A = images[idx_A]  # (256, 256, 1)
    img_B = images[idx_B]  # (256, 256, 1)
    print(f"[PICK] Pair selected: mean(A)={means[idx_A]:.4f}, "
          f"mean(B)={means[idx_B]:.4f}, delta={diff_matrix[idx_A, idx_B]:.4f}")
    return img_A, img_B


def generate_organic_mask(h, w, cloud_sigma=4.0, soften_sigma=1.0):
    """
    Generate an organic, cloud-like binary mask with soft, gradient edges.

    Algorithm:
      1. Random Gaussian noise (h, w)
      2. Gaussian blur with large sigma → big organic blobs
      3. Binarize at median → hard 0/1 mask
      4. Gaussian blur with small sigma → soft gradient transitions
         (this is THE key innovation that eliminates checkerboard artifacts)
    """
    # Step 1: Random noise
    noise = np.random.randn(h, w).astype(np.float32)

    # Step 2: Large sigma blur → cloud-like organic blobs
    cloud = gaussian_filter(noise, sigma=cloud_sigma)

    # Step 3: Binarize at median
    threshold = np.median(cloud)
    binary = (cloud > threshold).astype(np.float32)

    # Step 4: Soften the hard edges with mild Gaussian blur
    soft_mask = gaussian_filter(binary, sigma=soften_sigma)

    # Expand dims for broadcasting: (1, h, w, 1)
    return soft_mask[np.newaxis, :, :, np.newaxis]


def main():
    print("=" * 60)
    print("  Histogram-Matched Continuous Latent Space Fusion")
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

    # Init graphs
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])
    print("[INFO] All models loaded.\n")

    # ── Pick two diverse images ──
    print("[INFO] Selecting two maximally-different real rock slices...")
    img_A_np, img_B_np = pick_two_diverse_images(DATA_DIR, num_candidates=64)

    # ── CRITICAL: Histogram matching ──
    # Force img_B's intensity distribution to match img_A's,
    # eliminating the background color mismatch ("yin-yang face" root cause)
    print("[INFO] Performing histogram matching: B -> A ...")
    img_B_matched_np = match_histograms(
        img_B_np[:, :, 0],   # (256, 256) — reference shape for skimage
        img_A_np[:, :, 0],   # (256, 256) — target histogram
    )
    img_B_matched_np = img_B_matched_np.astype(np.float32)
    # Clip any out-of-range values from matching
    img_B_matched_np = np.clip(img_B_matched_np, 0.0, 1.0)
    img_B_matched = np.expand_dims(img_B_matched_np, axis=-1)  # (256, 256, 1)

    print(f"    After matching: mean(B) was {img_B_np.mean():.4f} -> "
          f"now {img_B_matched.mean():.4f} (target A mean={img_A_np.mean():.4f})")

    # ── Encode to continuous latent space Z_e (BEFORE VQ) ──
    print("[INFO] Encoding to continuous latent space Z_e...")
    z_eA = encoder(tf.constant(np.expand_dims(img_A_np, 0)), training=False)       # (1, 32, 32, 64)
    z_eB = encoder(tf.constant(np.expand_dims(img_B_matched, 0)), training=False)  # (1, 32, 32, 64)

    # ── Generate 4 fused novel rocks ──
    print(f"[INFO] Generating {NUM_FUSIONS} novel rock slices via continuous fusion...\n")
    fusion_images = []

    for i in range(NUM_FUSIONS):
        print(f"  Fusion #{i+1}/{NUM_FUSIONS}...")

        # Generate organic cloud mask
        mask = generate_organic_mask(LATENT_SIZE, LATENT_SIZE,
                                     cloud_sigma=CLOUD_SIGMA,
                                     soften_sigma=SOFTEN_SIGMA)
        # mask shape: (1, 32, 32, 1), values in [0, 1] with soft gradients

        # Continuous fusion in Z_e space
        z_fused = z_eA * (1.0 - mask) + z_eB * mask  # (1, 32, 32, 64)

        # VQ quantize the fused continuous features
        z_q = vq_layer(z_fused, training=False)  # (1, 32, 32, 64)

        # Decode to image
        generated = decoder(z_q, training=False)
        img_gen = np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)
        fusion_images.append(img_gen)

        # Report mask coverage
        coverage = mask.mean()
        print(f"    Mask coverage: {coverage:.2%} (B-zone fraction)")

    # ── Decode real A and real B_matched for display ──
    print("\n[INFO] Decoding real references for display...")
    z_qA = vq_layer(z_eA, training=False)
    img_A_decoded = np.clip(decoder(z_qA, training=False).numpy()[0, :, :, 0], 0.0, 1.0)

    z_qB = vq_layer(z_eB, training=False)
    img_B_decoded = np.clip(decoder(z_qB, training=False).numpy()[0, :, :, 0], 0.0, 1.0)

    # ── Render 1×6 grid ──
    print("[INFO] Rendering 1×6 visualization...")
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))

    titles = [f'Real A\n(Matrix Parent)'] + \
             [f'Fusion #{i+1}' for i in range(NUM_FUSIONS)] + \
             [f'Real B\n(Pore Donor)']

    all_images = [img_A_decoded] + fusion_images + [img_B_decoded]
    colors = ['royalblue'] + ['darkorange'] * NUM_FUSIONS + ['crimson']

    for idx, ax in enumerate(axes.flat):
        ax.imshow(all_images[idx], cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(titles[idx], fontsize=11, fontweight='bold', color=colors[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[idx])
            spine.set_linewidth(2)

    plt.suptitle(
        'Histogram-Matched Continuous Latent Fusion — Zero Block Artifacts',
        fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[DONE] Perfect fusion generation saved to:\n  {OUTPUT_PATH}")
    print(f"[STATS] Source pair: mean(A)={img_A_np.mean():.4f}, "
          f"mean(B)={img_B_np.mean():.4f} (matched->{img_B_matched.mean():.4f})")


if __name__ == "__main__":
    main()
