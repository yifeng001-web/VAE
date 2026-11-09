"""
Chapter 4 Quantitative Evaluation: Figure 4-3, 4-4, 4-5
========================================================
Task 4-3: Porosity Cross-Validation (Figure_4_3_Porosity.png)
Task 4-4: Two-Point Probability S2(r) (Figure_4_4_S2_Function.png)
Task 4-5: PixelCNN Creativity Generation Matrix (Figure_4_5_Generated_Matrix.png)

Shared: VQ-VAE composite-constraint model (Epoch 50) + PixelCNN prior.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from PIL import Image
import time

# ===== Paths =====
from thesis_code.config import PROJECT_ROOT


from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===== Chinese font =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
                                   'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===== Config =====
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")  # Test set
os.makedirs(EVAL_DIR, exist_ok=True)

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
IMAGE_SIZE = 256
LATENT_SIZE = 32
EPOCH_TO_LOAD = 50
HIDDEN_CHANNELS = 128
NUM_RES_BLOCKS = 12

# Output paths
OUTPUT_4_3 = os.path.join(EVAL_DIR, "Figure_4_3_Porosity.png")
OUTPUT_4_4 = os.path.join(EVAL_DIR, "Figure_4_4_S2_Function.png")
OUTPUT_4_5 = os.path.join(EVAL_DIR, "Figure_4_5_Generated_Matrix.png")

# PixelCNN generation params
TEMPERATURE = 0.95
TOP_P = 0.95
NUM_SEEDS = 6
SEED_PATCH_SIZE = 2
GAUSS_SIGMA = 1.0
GAUSS_KERNEL = 5
NUM_GENERATED = 16
CIRCLE_CENTER = (128, 128)
CIRCLE_RADIUS = 120

# Academic colors
C_BLUE = '#2C6FAC'
C_RED = '#D9583B'
C_GREEN = '#27AE60'
C_ORANGE = '#E67E22'
C_PURPLE = '#8E44AD'
C_DARK = '#1a1a1a'
C_GRAY = '#888888'


# ============================================================
# MaskedConv2D & ResidualBlock (PixelCNN, same as train_pixelcnn.py)
# ============================================================
class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, mask_type='B', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type.upper()

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv = tf.keras.layers.Conv2D(
            self.filters, self.kernel_size, padding='same', use_bias=True,
            kernel_initializer='he_normal')
        self.conv.build(input_shape)
        k_h, k_w = self.kernel_size, self.kernel_size
        center_h, center_w = k_h // 2, k_w // 2
        mask = np.ones((k_h, k_w, in_channels, self.filters), dtype=np.float32)
        mask[center_h + 1:, :, :, :] = 0.0
        mask[center_h, center_w + 1:, :, :] = 0.0
        if self.mask_type == 'A':
            mask[center_h, center_w, :, :] = 0.0
        self.mask = tf.constant(mask)
        super().build(input_shape)

    def call(self, x):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(x)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.masked_conv1 = MaskedConv2D(self.filters // 2, 3, mask_type='B')
        self.masked_conv2 = MaskedConv2D(self.filters, 3, mask_type='B')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.skip_conv = tf.keras.layers.Conv2D(self.filters, 1, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.masked_conv1(x)
        h = self.bn1(h)
        h = tf.nn.relu(h)
        h = self.dropout(h, training=training)
        h = self.masked_conv2(h)
        h = self.bn2(h)
        if x.shape[-1] != self.filters:
            x = self.skip_conv(x)
        return tf.nn.relu(x + h)


def build_pixelcnn_inference():
    inputs = tf.keras.Input(shape=(LATENT_SIZE, LATENT_SIZE), dtype=tf.int32)
    x = tf.one_hot(inputs, depth=NUM_EMBEDDINGS)
    x = MaskedConv2D(HIDDEN_CHANNELS, 7, mask_type='A', name='masked_conv_A')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    for i in range(NUM_RES_BLOCKS):
        x = ResidualBlock(HIDDEN_CHANNELS, name=f'res_block_{i}')(x)
    outputs = tf.keras.layers.Conv2D(NUM_EMBEDDINGS, 1, name='output_logits')(x)
    return tf.keras.Model(inputs, outputs, name='PixelCNN')


# ============================================================
# Model Loading
# ============================================================
def load_all_models():
    """Load VQ-VAE encoder, decoder, VQ layer, and PixelCNN."""
    print("[INFO] Loading VQ-VAE models...")
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])
    print("[INFO] VQ-VAE models loaded.")

    # PixelCNN
    pixelcnn_weights = os.path.join(CHECKPOINT_DIR, "pixelcnn.weights.h5")
    if not os.path.exists(pixelcnn_weights):
        raise FileNotFoundError(f"[FATAL] PixelCNN weights not found: {pixelcnn_weights}")
    print("[INFO] Loading PixelCNN...")
    pixelcnn = build_pixelcnn_inference()
    _ = pixelcnn(tf.zeros([1, LATENT_SIZE, LATENT_SIZE], dtype=tf.int32))
    pixelcnn.load_weights(pixelcnn_weights)
    print("[INFO] PixelCNN loaded.")

    return encoder, decoder, vq_layer, pixelcnn


# ============================================================
# Image I/O helpers
# ============================================================
def load_one_tif(filepath):
    """Load TIF, resize to 256x256, per-image min-max normalize to [0,1]."""
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t)
    max_v = tf.reduce_max(img_t)
    img_n = (img_t - min_v) / (max_v - min_v + 1e-8)
    return img_n.numpy()


def load_one_tif_raw(filepath):
    """Load TIF, resize to 256x256, return BOTH raw 16-bit AND normalized versions.
    Returns: (img_raw_2d, img_norm, raw_min, raw_max)
        img_raw_2d: (256, 256) float32 in original 16-bit space
        img_norm:   (256, 256, 1) float32 in per-image [0, 1] space
        raw_min, raw_max: float scalars for denormalization
    """
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    raw_min = float(tf.reduce_min(img_t).numpy())
    raw_max = float(tf.reduce_max(img_t).numpy())
    img_raw_2d = img_t.numpy()[:, :, 0]  # (256, 256) raw 16-bit
    img_n = ((img_t.numpy() - raw_min) / (raw_max - raw_min + 1e-8))
    return img_raw_2d, img_n, raw_min, raw_max


def denormalize_to_raw(img_norm_2d, raw_min, raw_max):
    """Convert [0,1] normalized image back to raw 16-bit space.
    img_norm_2d: (256, 256) float32 in [0, 1]
    Returns: (256, 256) float32 in [raw_min, raw_max]
    """
    return img_norm_2d * (raw_max - raw_min) + raw_min


def gather_tif_paths(data_dir):
    file_list = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(('.tif', '.tiff')):
            file_list.append(os.path.join(data_dir, fname))
    return file_list


def reconstruct(encoder, decoder, vq_layer, img_array):
    inp = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(inp, training=False)
    quantized = vq_layer(z_e)
    recon = decoder(quantized, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)


# ============================================================
# Binarization (Fixed Physical Threshold for Carbonate CT)
# ============================================================
# Carbonate CT histograms are unimodal + long-tail (NOT bimodal),
# so Otsu fails badly (~34% porosity instead of <5%).
# We use a fixed physical threshold in 16-bit raw space:
# pore = pixel value < 4500 (CT attenuation units).
FIXED_THRESHOLD_16BIT = 4500.0


def otsu_threshold(img_2d):
    """Otsu's method for binary thresholding.
    DEPRECATED: Otsu assumes bimodal histogram and is unsuitable for
    carbonate CT images (unimodal + long tail). Use binarize_physical() instead.
    Kept for reference only.
    """
    hist, bin_edges = np.histogram(img_2d.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    sum_all = np.dot(np.arange(256), hist)
    weight_bg = 0.0
    sum_bg = 0.0
    max_val = 0.0
    best_thresh = 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_val:
            max_val = between
            best_thresh = t
    return bin_edges[best_thresh]


def binarize_physical(img_raw_2d, threshold=FIXED_THRESHOLD_16BIT):
    """Binarize raw 16-bit image: pore (dark < threshold) = 1, matrix = 0."""
    return (img_raw_2d < threshold).astype(np.float32)


def binarize(img_2d):
    """DEPRECATED: Use binarize_physical() instead.
    Kept for backward compatibility only.
    """
    thresh = otsu_threshold(img_2d)
    return (img_2d < thresh).astype(np.float32)


# ============================================================
# Task 4-3: Porosity Cross-Validation
# ============================================================
def compute_porosity_physical(img_raw_2d, threshold=FIXED_THRESHOLD_16BIT):
    """Binary areal porosity from raw 16-bit image with fixed physical threshold."""
    binary = binarize_physical(img_raw_2d, threshold)
    return float(np.mean(binary))


def compute_porosity_binary(img_2d):
    """DEPRECATED: Use compute_porosity_physical() instead.
    Kept for backward compatibility only.
    """
    binary = binarize(img_2d)
    return float(np.mean(binary))


def task_4_3_porosity(encoder, decoder, vq_layer):
    """Porosity quantitative validation: 100 real vs reconstructed.
    Uses fixed physical threshold (FIXED_THRESHOLD_16BIT = 4500) in raw 16-bit space,
    because Otsu fails on unimodal carbonate CT histograms.
    """
    print("\n" + "=" * 70)
    print("  TASK 4-3: Porosity Cross-Validation (Figure 4-3)")
    print("=" * 70)

    all_paths = gather_tif_paths(DATA_DIR)
    print(f"[INFO] Test set: {len(all_paths)} TIF files")
    print(f"[INFO] Binarization: fixed threshold < {FIXED_THRESHOLD_16BIT:.0f} (16-bit CT units)")

    # Randomly sample 100
    np.random.seed(42)
    sampled = np.random.choice(all_paths, min(100, len(all_paths)), replace=False)

    real_porosities = []
    recon_porosities = []

    for i, fp in enumerate(sampled):
        # Load both raw 16-bit and normalized versions
        img_raw_2d, img_norm, raw_min, raw_max = load_one_tif_raw(fp)
        # Porosity from raw 16-bit with physical threshold
        real_por = compute_porosity_physical(img_raw_2d, FIXED_THRESHOLD_16BIT)
        # Reconstruct in normalized space, then denormalize to raw for fair comparison
        recon_norm = reconstruct(encoder, decoder, vq_layer, img_norm)
        recon_raw_2d = denormalize_to_raw(recon_norm, raw_min, raw_max)
        recon_por = compute_porosity_physical(recon_raw_2d, FIXED_THRESHOLD_16BIT)
        real_porosities.append(real_por)
        recon_porosities.append(recon_por)
        if (i + 1) % 20 == 0:
            print(f"  ... processed {i+1}/{len(sampled)}")

    real_por = np.array(real_porosities)
    recon_por = np.array(recon_porosities)

    # Statistics
    slope, intercept = np.polyfit(real_por, recon_por, 1)
    recon_pred = slope * real_por + intercept
    ss_res = np.sum((recon_por - recon_pred) ** 2)
    ss_tot = np.sum((recon_por - np.mean(recon_por)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(real_por - recon_por))
    mean_real = np.mean(real_por)
    mean_recon = np.mean(recon_por)

    print(f"\n  {'='*50}")
    sign = "+" if intercept >= 0 else ""
    print(f"  [RESULT] Linear fit: y = {slope:.6f} x {sign}{intercept:.6f}")
    print(f"           a = {slope:.6f}, b = {intercept:.6f}")
    print(f"           R^2 = {r2:.6f}")
    print(f"           MAE = {mae:.6f}")
    print(f"           Mean real porosity  = {mean_real:.6f}")
    print(f"           Mean recon porosity = {mean_recon:.6f}")
    print(f"  {'='*50}")

    # ── Plot: 1x2 figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

    # Left: scatter with y=x and fit line
    ax1.scatter(real_por, recon_por, alpha=0.55, s=35,
                c=C_BLUE, edgecolors='white', linewidth=0.4, zorder=5)

    all_min = min(real_por.min(), recon_por.min())
    all_max = max(real_por.max(), recon_por.max())
    pad = (all_max - all_min) * 0.08
    lim = [all_min - pad, all_max + pad]

    ax1.plot(lim, lim, '--', color=C_RED, linewidth=1.8, label='y = x (Ideal)', zorder=3)
    x_fit = np.linspace(lim[0], lim[1], 100)
    y_fit = slope * x_fit + intercept
    ax1.plot(x_fit, y_fit, '-', color=C_ORANGE, linewidth=2.0,
             label=f'Fit: y = {slope:.4f}x {"+" if intercept >= 0 else ""}{intercept:.4f}', zorder=4)

    ax1.set_xlabel('True Areal Porosity (Ground Truth)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Reconstructed Areal Porosity (VQ-VAE)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Porosity Cross-Plot (N={len(sampled)})\n'
                  f'R$^2$ = {r2:.4f}  |  MAE = {mae:.4f}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_aspect('equal')

    # Right: overlapping density histogram
    bins = np.linspace(min(all_min, 0), all_max + 0.02, 28)
    ax2.hist(real_por, bins=bins, alpha=0.55, color=C_BLUE, edgecolor='navy',
             linewidth=0.5, label=f'Ground Truth (mean={mean_real:.4f})', density=True)
    ax2.hist(recon_por, bins=bins, alpha=0.55, color=C_ORANGE, edgecolor='darkred',
             linewidth=0.5, label=f'Reconstructed (mean={mean_recon:.4f})', density=True)

    # Vertical lines for means
    ax2.axvline(mean_real, color=C_BLUE, linestyle='--', linewidth=1.8, alpha=0.7)
    ax2.axvline(mean_recon, color=C_ORANGE, linestyle='--', linewidth=1.8, alpha=0.7)

    ax2.set_xlabel('Areal Porosity', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax2.set_title('Porosity Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_4_3, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[DONE] Figure 4-3 saved to: {OUTPUT_4_3}")

    return real_por, recon_por


# ============================================================
# Task 4-4: Two-Point Probability S2(r)
# ============================================================
def compute_s2_fft(binary_img):
    """
    Compute two-point probability function S2(r) via FFT autocorrelation.
    S2(r) = P(pixel at x is pore AND pixel at x+r is pore).
    Uses FFT acceleration and radial binning.
    Returns: (r_values, s2_values)
    """
    h, w = binary_img.shape
    # Zero-mean for autocorrelation
    f = binary_img.astype(np.float64)

    # 2D autocorrelation via FFT
    f_fft = np.fft.fft2(f)
    power = np.abs(f_fft) ** 2
    autocorr = np.fft.fftshift(np.fft.ifft2(power).real)
    # Normalize
    autocorr /= (h * w)

    # Radial binning
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx) - 1  # up to ~127 pixels

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    r_values = np.arange(0, max_r + 1, dtype=np.float64)
    s2_values = np.zeros(len(r_values), dtype=np.float64)

    for i, r in enumerate(r_values):
        mask = (dist >= r - 0.5) & (dist < r + 0.5)
        if mask.sum() > 0:
            s2_values[i] = autocorr[mask].mean()

    # Normalize so S2(0) = porosity (which equals mean pore fraction)
    # Actually autocorr[center] / (h*w) ~ mean(pore^2) = mean(pore) for binary
    # S2(0) should equal porosity
    por = np.mean(f)
    if s2_values[0] > 0:
        # scale so S2(0) = porosity
        scale = por / s2_values[0]
        s2_values *= scale
    s2_values[0] = por  # enforce exact

    return r_values, s2_values


def task_4_4_s2(encoder, decoder, vq_layer):
    """Two-point probability S2(r) validation.
    Uses fixed physical threshold (FIXED_THRESHOLD_16BIT = 4500) in raw 16-bit space.
    """
    print("\n" + "=" * 70)
    print("  TASK 4-4: Two-Point Probability S2(r) (Figure 4-4)")
    print("=" * 70)

    all_paths = gather_tif_paths(DATA_DIR)

    # Select a representative sample: scan for medium porosity + high edge structure
    np.random.seed(123)
    candidate_indices = np.random.choice(len(all_paths), min(40, len(all_paths)), replace=False)

    best_path = None
    best_score = -1.0
    best_raw_2d = None
    best_norm = None
    best_raw_min = None
    best_raw_max = None
    for idx in candidate_indices:
        fp = all_paths[idx]
        img_raw_2d, img_norm, raw_min, raw_max = load_one_tif_raw(fp)
        # Porosity via physical threshold
        por = compute_porosity_physical(img_raw_2d, FIXED_THRESHOLD_16BIT)
        # Compute edge density via Sobel on normalized image
        t = tf.convert_to_tensor(img_norm[np.newaxis, :, :, :1], dtype=tf.float32)
        edges = tf.image.sobel_edges(t)
        mag = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
        edge_density = float(tf.reduce_mean(mag))
        # Prefer medium porosity + clear edge structure (tuned for <0.05 range)
        score = edge_density * 5.0 - abs(por - 0.03) * 15.0
        if score > best_score:
            best_score = score
            best_path = fp
            best_raw_2d = img_raw_2d
            best_norm = img_norm
            best_raw_min = raw_min
            best_raw_max = raw_max

    print(f"[INFO] Selected: {os.path.basename(best_path)} (score={best_score:.4f})")

    # Binarize real with physical threshold
    real_bin = binarize_physical(best_raw_2d, FIXED_THRESHOLD_16BIT)
    # Reconstruct in normalized space, denormalize, then binarize
    recon_norm = reconstruct(encoder, decoder, vq_layer, best_norm)
    recon_raw_2d = denormalize_to_raw(recon_norm, best_raw_min, best_raw_max)
    recon_bin = binarize_physical(recon_raw_2d, FIXED_THRESHOLD_16BIT)

    por_real = float(np.mean(real_bin))
    por_recon = float(np.mean(recon_bin))
    print(f"[INFO] Porosity (threshold < {FIXED_THRESHOLD_16BIT:.0f}): real={por_real:.6f}, recon={por_recon:.6f}")

    # Compute S2(r)
    print("[INFO] Computing S2(r) for real image (FFT autocorrelation)...")
    t0 = time.time()
    r_real, s2_real = compute_s2_fft(real_bin)
    print(f"  Real S2(r) done in {time.time()-t0:.1f}s")

    print("[INFO] Computing S2(r) for reconstructed image...")
    t0 = time.time()
    r_recon, s2_recon = compute_s2_fft(recon_bin)
    print(f"  Recon S2(r) done in {time.time()-t0:.1f}s")

    # Pearson correlation between two S2 curves
    # Use same r range
    min_len = min(len(s2_real), len(s2_recon))
    pearson_r = np.corrcoef(s2_real[:min_len], s2_recon[:min_len])[0, 1]

    # Find "short-range" distance where S2 drops sharply
    # Define as the distance where S2(r) - porosity^2 drops to 1/e of its initial value
    plateau = por_real ** 2  # asymptotic value
    initial_excess = s2_real[0] - plateau
    if initial_excess > 0:
        threshold = plateau + initial_excess / np.e
        short_range = 0
        for i in range(1, len(r_real)):
            if s2_real[i] <= threshold:
                short_range = int(r_real[i])
                break
        if short_range == 0:
            short_range = int(r_real[-1])
    else:
        short_range = 10

    print(f"\n  {'='*50}")
    print(f"  [RESULT] Pearson r (S2 real vs recon): {pearson_r:.6f}")
    print(f"  [RESULT] Short-range correlation distance: 0 ~ {short_range} pixels")
    print(f"           (where S2(r) drops to 1/e of excess above plateau)")
    print(f"  {'='*50}")

    # ── Plot ──
    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='white',
                                            gridspec_kw={'width_ratios': [1.0, 0.7]})

    # Main plot: full S2(r) curves
    ax_main.plot(r_real, s2_real, '-', color=C_BLUE, linewidth=2.2, alpha=0.85,
                 label=f'Ground Truth (porosity={por_real:.4f})')
    ax_main.plot(r_recon, s2_recon, '--', color=C_RED, linewidth=2.2, alpha=0.85,
                 label=f'Reconstructed (porosity={por_recon:.4f})')

    # Plateau line
    ax_main.axhline(plateau, color=C_GRAY, linestyle=':', linewidth=1.2,
                    alpha=0.5, label=f'Plateau ($\\phi^2$ = {plateau:.4f})')

    # Mark short-range zone
    ax_main.axvspan(0, short_range, alpha=0.08, color=C_ORANGE)
    ax_main.annotate(f'Short-range\ncorrelation\n(0~{short_range} px)',
                     xy=(short_range / 2, s2_real[0] * 0.5),
                     fontsize=9, ha='center', color=C_ORANGE, fontweight='bold')

    ax_main.set_xlabel('Distance $r$ (pixels)', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Two-Point Probability $S_2(r)$', fontsize=13, fontweight='bold')
    ax_main.set_title('Two-Point Probability Function $S_2(r)$\nReal vs Reconstructed',
                      fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_xlim(0, max(r_real[-1], r_recon[-1]))

    # Zoom-in: first 50 pixels
    zoom_max = min(50, len(r_real) - 1)
    ax_zoom.plot(r_real[:zoom_max], s2_real[:zoom_max], '-', color=C_BLUE,
                 linewidth=2.5, alpha=0.9, label='Ground Truth')
    ax_zoom.plot(r_recon[:zoom_max], s2_recon[:zoom_max], '--', color=C_RED,
                 linewidth=2.5, alpha=0.9, label='Reconstructed')
    ax_zoom.axvline(short_range, color=C_ORANGE, linestyle='--', linewidth=1.5,
                    alpha=0.7, label=f'r = {short_range} px')
    ax_zoom.set_xlabel('Distance $r$ (pixels)', fontsize=12, fontweight='bold')
    ax_zoom.set_ylabel('$S_2(r)$', fontsize=12, fontweight='bold')
    ax_zoom.set_title(f'Zoom: Short-Range (0~{zoom_max} px)\n'
                      f'Pearson $r$ = {pearson_r:.6f}',
                      fontsize=13, fontweight='bold')
    ax_zoom.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_zoom.grid(True, alpha=0.3, linestyle='--')
    ax_zoom.set_xlim(0, zoom_max)

    fig.suptitle('Spatial Autocorrelation Validation — Two-Point Probability $S_2(r)$',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_4_4, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[DONE] Figure 4-4 saved to: {OUTPUT_4_4}")

    return r_real, s2_real, r_recon, s2_recon


# ============================================================
# Task 4-5: PixelCNN Creativity Generation Matrix
# ============================================================
def autoregressive_sample(pixelcnn, real_code_map, temperature=0.95, top_p=0.95,
                          num_seeds=6, seed_patch_size=2):
    """Seed-prompted autoregressive PixelCNN sampling (same as generate_unconditional_thesis.py)."""
    code_map = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=np.int32)
    seed_mask = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=bool)

    max_row = LATENT_SIZE - seed_patch_size
    max_col = LATENT_SIZE - seed_patch_size

    all_positions = [(r, c) for r in range(0, max_row + 1, seed_patch_size)
                      for c in range(0, max_col + 1, seed_patch_size)]
    np.random.shuffle(all_positions)
    chosen = all_positions[:num_seeds]

    for (sr, sc) in chosen:
        for dr in range(seed_patch_size):
            for dc in range(seed_patch_size):
                r, c = sr + dr, sc + dc
                code_map[0, r, c] = real_code_map[0, r, c]
                seed_mask[0, r, c] = True

    seed_count = seed_mask.sum()
    print(f"    [SEED] Planted {seed_count} seed pixels across {len(chosen)} "
          f"{seed_patch_size}x{seed_patch_size} patches")

    for h in range(LATENT_SIZE):
        for w in range(LATENT_SIZE):
            if seed_mask[0, h, w]:
                continue
            logits = pixelcnn(tf.constant(code_map), training=False)
            logits_hw = logits[0, h, w, :]
            if temperature > 0:
                logits_hw = logits_hw / temperature
            sorted_logits, sorted_indices = tf.math.top_k(logits_hw, k=NUM_EMBEDDINGS)
            probs = tf.nn.softmax(sorted_logits)
            cumsum = tf.cumsum(probs)
            nucleus_mask = tf.concat([tf.ones(1, dtype=tf.bool), cumsum[:-1] < top_p], axis=0)
            nucleus_logits = tf.where(nucleus_mask, sorted_logits,
                                      tf.ones_like(sorted_logits) * (-1e9))
            sampled_pos = tf.random.categorical(
                tf.expand_dims(nucleus_logits, 0), num_samples=1)[0, 0]
            sampled_idx = sorted_indices[sampled_pos]
            code_map[0, h, w] = sampled_idx.numpy()

    survived = seed_mask.sum()
    print(f"    [SEED] Survived seeds after growth: {survived}/{seed_count}")
    return code_map


def indices_to_image(decoder, vq_layer, code_map):
    indices_flat = tf.reshape(code_map, [-1])
    embeddings = vq_layer.embeddings
    embeddings_t = tf.transpose(embeddings)
    quantized_flat = tf.gather(embeddings_t, indices_flat)
    quantized = tf.reshape(quantized_flat, [1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM])
    generated = decoder(quantized, training=False)
    img = np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)
    return img


def apply_circular_mask(image, center=CIRCLE_CENTER, radius=CIRCLE_RADIUS):
    h, w = image.shape
    cy, cx = center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    mask = (dist <= radius).astype(np.float32)
    return image * mask


def task_4_5_generation(encoder, decoder, vq_layer, pixelcnn):
    """PixelCNN creativity generation matrix (16 samples, 4x4 grid)."""
    print("\n" + "=" * 70)
    print("  TASK 4-5: PixelCNN Creativity Generation Matrix (Figure 4-5)")
    print("=" * 70)

    # ── Step 1: Extract seed code map from a real high-porosity rock ──
    print("[INFO] Extracting pore seed crystal from real rock...")
    all_paths = gather_tif_paths(DATA_DIR)
    np.random.seed(777)
    candidate_paths = np.random.choice(all_paths, min(32, len(all_paths)), replace=False)

    # Pick the image with highest porosity (most pore seeds)
    best_img = None
    best_mean = float('inf')  # darkest = highest porosity
    for fp in candidate_paths:
        img = load_one_tif(fp)
        m = np.mean(img)
        if m < best_mean:
            best_mean = m
            best_img = img

    img_real = np.expand_dims(best_img, axis=0)
    print(f"    Selected seed image (mean intensity: {best_mean:.4f}, "
          f"porosity proxy: {1.0-best_mean:.4f})")

    # Encode to discrete code map
    z_e = encoder(tf.constant(img_real), training=False)
    z_e_flat = tf.reshape(z_e, [-1, LATENT_DIM])
    embeddings = vq_layer.embeddings
    similarity = tf.matmul(z_e_flat, embeddings)
    flat_sq = tf.reduce_sum(z_e_flat ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(tf.reduce_sum(embeddings ** 2, axis=0), [1, -1])
    distances = flat_sq + emb_sq - 2.0 * similarity
    encoding_indices = tf.argmin(distances, axis=1)
    real_code_map = tf.reshape(encoding_indices, [1, LATENT_SIZE, LATENT_SIZE]).numpy().astype(np.int32)
    print(f"    Seed code map: {len(np.unique(real_code_map))} unique codes")

    # ── Step 2: Generate 16 novel rocks ──
    print(f"\n[INFO] Generating {NUM_GENERATED} novel rock slices...")
    print(f"[INFO] Temperature: {TEMPERATURE}, Top-P: {TOP_P}, Seeds: {NUM_SEEDS}")

    generated_images = []
    code_maps = []

    for i in range(NUM_GENERATED):
        print(f"  Generating sample {i+1}/{NUM_GENERATED}...")
        code_map = autoregressive_sample(pixelcnn, real_code_map,
                                         temperature=TEMPERATURE, top_p=TOP_P,
                                         num_seeds=NUM_SEEDS, seed_patch_size=SEED_PATCH_SIZE)
        code_maps.append(code_map)

        img = indices_to_image(decoder, vq_layer, code_map)
        # Apply circular mask
        img = apply_circular_mask(img)
        # Apply Gaussian smoothing
        img = gaussian_filter(img, sigma=GAUSS_SIGMA)
        img = np.clip(img, 0.0, 1.0)
        generated_images.append(img)

        unique_codes = len(np.unique(code_map))
        por = 1.0 - np.mean(img[img > 0]) if np.any(img > 0) else 0.0
        print(f"    Unique codes: {unique_codes}/{NUM_EMBEDDINGS}, "
              f"porosity approx: {por:.4f}")

    # ── Step 3: 4x4 grid plot ──
    print("\n[INFO] Rendering 4x4 generation matrix...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), facecolor='white')
    fig.subplots_adjust(wspace=0.04, hspace=0.08)

    # Classify each generated image for labeling
    for i, ax in enumerate(axes.flat):
        img = generated_images[i]
        por = 1.0 - np.mean(img[img > 0.01]) if np.any(img > 0.01) else 0.0

        # Classify: dense (<0.30), micro-crack (0.30-0.55), dissolved (>0.55)
        if por < 0.30:
            label = 'Dense'
            color = C_BLUE
        elif por < 0.55:
            label = 'Micro-crack'
            color = C_ORANGE
        else:
            label = 'Dissolved'
            color = C_RED

        ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0, interpolation='bilinear')
        ax.set_title(f'#{i+1}  {label} ($\\phi$={por:.3f})',
                     fontsize=10, fontweight='bold', color=color, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.8)

    fig.suptitle('VQ-VAE Unconditional Novel Rock Generation — PixelCNN Prior (Stage 2)\n'
                 f'Fixed Seed Prompt + Autoregressive Sampling ($T$={TEMPERATURE}, '
                 f'Top-$P$={TOP_P}, Seeds={NUM_SEEDS}) — $4\\times4$ Matrix',
                 fontsize=16, fontweight='bold', y=1.01)

    plt.savefig(OUTPUT_4_5, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[DONE] Figure 4-5 saved to: {OUTPUT_4_5}")

    # Diversity report
    all_unique = set()
    for cm in code_maps:
        all_unique.update(cm.flatten().tolist())
    porosities = [1.0 - np.mean(img[img > 0.01]) if np.any(img > 0.01) else 0.0
                  for img in generated_images]
    print(f"\n  [DIVERSITY REPORT]")
    print(f"    Total unique codes across {NUM_GENERATED} samples: "
          f"{len(all_unique)}/{NUM_EMBEDDINGS}")
    print(f"    Porosity range: [{min(porosities):.4f}, {max(porosities):.4f}]")
    print(f"    Porosity mean: {np.mean(porosities):.4f}, std: {np.std(porosities):.4f}")

    return generated_images


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("  Chapter 4 Quantitative Evaluation")
    print("  VQ-VAE + PixelCNN Composite Model")
    print("=" * 70)

    # ── Load all models once ──
    encoder, decoder, vq_layer, pixelcnn = load_all_models()

    # ── Task 4-3: Porosity ──
    real_por, recon_por = task_4_3_porosity(encoder, decoder, vq_layer)

    # ── Task 4-4: S2(r) ──
    r_real, s2_real, r_recon, s2_recon = task_4_4_s2(encoder, decoder, vq_layer)

    # ── Task 4-5: Generation Matrix ──
    generated_images = task_4_5_generation(encoder, decoder, vq_layer, pixelcnn)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  Chapter 4 Evaluation Complete!")
    print("=" * 70)
    print(f"  Figure 4-3: {OUTPUT_4_3}")
    print(f"  Figure 4-4: {OUTPUT_4_4}")
    print(f"  Figure 4-5: {OUTPUT_4_5}")
    print("=" * 70)


if __name__ == "__main__":
    main()
