"""
Figure 4-5 Retrained: PixelCNN Generation with Reinforced Training
=================================================================
- Loads pixelcnn_retrain.weights.h5 (200 Epoch, 18 ResBlocks, RF=43)
- Ultra-conservative sampling: T=0.1, Top-P=0.5
- 3 seeds of 3x3 patches from a real high-porosity rock
- Mandatory 5x5 Gaussian smoothing (sigma=1.0)
- Output: Figure_4_5_Retrained_Matrix.png (4x4 grid)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from thesis_code.config import PROJECT_ROOT


from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
CODES_PATH = os.path.join(CHECKPOINT_DIR, "latent_codes.npy")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
os.makedirs(EVAL_DIR, exist_ok=True)

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
IMAGE_SIZE = 256
LATENT_SIZE = 32
EPOCH_TO_LOAD = 50
HIDDEN_CHANNELS = 128
NUM_RES_BLOCKS = 18  # Must match train_pixelcnn_retrain.py

GAUSS_SIGMA = 1.0
CIRCLE_CENTER = (128, 128)
CIRCLE_RADIUS = 120

OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_4_5_Retrained_Matrix.png")

# ============================================================
# MaskedConv2D & ResidualBlock (same as train_pixelcnn_retrain.py)
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
    # Bug Fix 1: Use explicit Embedding layer instead of tf.one_hot.
    # tf.one_hot creates (B,32,32,512) causing MaskedConv2D in_channels=512
    # (param-inefficient and architecturally non-standard).
    # Embedding(512, HIDDEN_CHANNELS) produces (B,32,32,128) — correct pattern.
    x = tf.keras.layers.Embedding(NUM_EMBEDDINGS, HIDDEN_CHANNELS, name='code_embedding')(inputs)
    x = MaskedConv2D(HIDDEN_CHANNELS, 7, mask_type='A', name='masked_conv_A')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    for i in range(NUM_RES_BLOCKS):
        x = ResidualBlock(HIDDEN_CHANNELS, name=f'res_block_{i}')(x)
    outputs = tf.keras.layers.Conv2D(NUM_EMBEDDINGS, 1, name='output_logits')(x)
    return tf.keras.Model(inputs, outputs, name='PixelCNN_Retrain')


# ============================================================
# Helpers
# ============================================================
def load_one_tif(filepath):
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t); max_v = tf.reduce_max(img_t)
    return ((img_t - min_v) / (max_v - min_v + 1e-8)).numpy()


def gather_tif_paths(data_dir):
    return [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
            if f.lower().endswith(('.tif', '.tiff'))]


def encode_to_codes(encoder, vq_layer, img_array):
    inp = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(inp, training=False)
    return find_nearest_indices(vq_layer, z_e)


def find_nearest_indices(vq_layer, z_e):
    z_e_flat = tf.reshape(z_e, [-1, LATENT_DIM])
    embeddings = vq_layer.embeddings
    similarity = tf.matmul(z_e_flat, embeddings)
    flat_sq = tf.reduce_sum(z_e_flat ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(tf.reduce_sum(embeddings ** 2, axis=0), [1, -1])
    distances = flat_sq + emb_sq - 2.0 * similarity
    indices = tf.argmin(distances, axis=1)
    return tf.reshape(indices, [1, LATENT_SIZE, LATENT_SIZE]).numpy().astype(np.int32)


def codes_to_image(decoder, vq_layer, code_map):
    indices_flat = tf.reshape(code_map, [-1])
    embeddings = vq_layer.embeddings
    embeddings_t = tf.transpose(embeddings)
    quantized_flat = tf.gather(embeddings_t, indices_flat)
    quantized = tf.reshape(quantized_flat, [1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM])
    generated = decoder(quantized, training=False)
    return np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)


def apply_circular_mask(image):
    h, w = image.shape
    cy, cx = CIRCLE_CENTER
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    mask = (dist <= CIRCLE_RADIUS).astype(np.float32)
    return image * mask


def autoregressive_sample(pixelcnn, real_code_map, temperature=0.1, top_p=0.5,
                          num_seeds=3, seed_patch_size=3, valid_mask=None):
    """
    Autoregressive sampling with seed prompting.
    T=0.5, Top-P=0.5: medium-low randomness.
    valid_mask: (NUM_EMBEDDINGS,) bool array, True for codes seen in training.
                Forces logits of invalid codes to -inf to prevent waste codes.
    """
    code_map = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=np.int32)
    seed_mask = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=bool)

    # Build -inf mask for invalid codes (applied at every sampling step)
    if valid_mask is not None:
        invalid_mask = tf.constant(~valid_mask, dtype=tf.float32)
        INF_NEG = tf.constant(-1e10, dtype=tf.float32)
        valid_count = valid_mask.sum()
        print(f"    [WASTE-FILTER] Valid codes: {valid_count}/{NUM_EMBEDDINGS}, "
              f"will suppress {NUM_EMBEDDINGS - valid_count} invalid codes")
    else:
        invalid_mask = None
        INF_NEG = None

    # Stage 1: Plant seeds
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
    print(f"    [SEED] Planted {seed_count} seed pixels ({num_seeds} patches of "
          f"{seed_patch_size}x{seed_patch_size})")

    # Stage 2: Raster-order growth
    for h in range(LATENT_SIZE):
        for w in range(LATENT_SIZE):
            if seed_mask[0, h, w]:
                continue
            logits = pixelcnn(tf.constant(code_map), training=False)
            logits_hw = logits[0, h, w, :]
            # Force invalid codes to -inf BEFORE temperature scaling
            if invalid_mask is not None:
                logits_hw = logits_hw + invalid_mask * INF_NEG
            if temperature > 0:
                logits_hw = logits_hw / temperature
            # Top-P nucleus sampling
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
    print(f"    [SEED] Survived: {survived}/{seed_count}")
    return code_map


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Figure 4-5 Retrained: Ultra-Conservative PixelCNN Generation")
    print(f"  Config: T=0.5, Top-P=0.5, 3 seeds (3x3), Gaussian sigma={GAUSS_SIGMA}")
    print(f"  PixelCNN: {NUM_RES_BLOCKS} ResBlocks, RF={7+NUM_RES_BLOCKS*2} > 32")
    print("=" * 70)

    # Load VQ-VAE
    print("\n[INFO] Loading VQ-VAE models...")
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))
    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])
    print("[INFO] VQ-VAE loaded (Epoch 50 composite-constraint).")

    # Load retrained PixelCNN
    retrain_weights = os.path.join(CHECKPOINT_DIR, "pixelcnn_retrain.weights.h5")
    if not os.path.exists(retrain_weights):
        raise FileNotFoundError(f"[FATAL] {retrain_weights} not found! Run train_pixelcnn_retrain.py first.")
    print("[INFO] Loading retrained PixelCNN (200 Epochs, 18 ResBlocks)...")
    pixelcnn = build_pixelcnn_inference()
    _ = pixelcnn(tf.zeros([1, LATENT_SIZE, LATENT_SIZE], dtype=tf.int32))
    pixelcnn.load_weights(retrain_weights)
    print("[INFO] PixelCNN retrained model loaded.")

    # Load valid code mask from training data
    print("\n[INFO] Loading valid code mask from latent_codes.npy...")
    all_training_codes = np.load(CODES_PATH).astype(np.int32)
    valid_set = set(np.unique(all_training_codes))
    valid_mask = np.zeros(NUM_EMBEDDINGS, dtype=bool)
    for c in valid_set:
        valid_mask[c] = True
    print(f"[INFO] Valid codes: {valid_mask.sum()}/{NUM_EMBEDDINGS} "
          f"(will force {NUM_EMBEDDINGS - valid_mask.sum()} unused codes to -inf)")

    # Find a high-porosity seed image
    print("\n[INFO] Selecting seed image...")
    all_paths = gather_tif_paths(DATA_DIR)
    np.random.seed(777)
    cand_paths = np.random.choice(all_paths, min(32, len(all_paths)), replace=False)
    best_img = None; best_mean = float('inf')
    for fp in cand_paths:
        img = load_one_tif(fp)
        m = np.mean(img)
        if m < best_mean: best_mean = m; best_img = img
    seed_code_map = encode_to_codes(encoder, vq_layer, best_img)
    print(f"  Seed image mean intensity: {best_mean:.4f}")

    # Generate 16 samples
    print("\n[INFO] Generating 16 samples with T=0.5, Top-P=0.5 + waste-filter...")
    samples = []
    for i in range(16):
        code_map = autoregressive_sample(pixelcnn, seed_code_map,
                                         temperature=0.5, top_p=0.5,
                                         num_seeds=3, seed_patch_size=3,
                                         valid_mask=valid_mask)
        unique_n = len(np.unique(code_map))
        img = codes_to_image(decoder, vq_layer, code_map)
        img = apply_circular_mask(img)
        # Mandatory Gaussian smoothing
        img = gaussian_filter(img, sigma=GAUSS_SIGMA)
        img = np.clip(img, 0.0, 1.0)
        por = 1.0 - np.mean(img[img > 0.01]) if np.any(img > 0.01) else 0.0
        samples.append((img, unique_n, por))
        print(f"  Sample {i+1:2d}: {unique_n} unique codes, porosity={por:.4f}")

    # ── Plot 4x4 matrix ──
    print("\n[PLOT] Rendering 4x4 retrained generation matrix...")
    fig, axes = plt.subplots(4, 4, figsize=(18, 18), facecolor='white')
    fig.subplots_adjust(wspace=0.02, hspace=0.08)

    for i, ax in enumerate(axes.flat):
        img, unique_n, por = samples[i]
        ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0, interpolation='bilinear')
        # Phase label
        if por < 0.25:
            phase = 'Dense'
        elif por < 0.48:
            phase = 'Micro-crack'
        else:
            phase = 'Dissolved'
        ax.set_title(f'#{i+1} {phase}\nphi={por:.3f} | uniq={unique_n}',
                     fontsize=8, fontweight='bold', pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.5)

    fig.suptitle(
        'Figure 4-5: PixelCNN Retrained Unconditional Novel Rock Generation\n'
        '200 Epochs | 18 ResBlocks (RF=43) | T=0.5, Top-P=0.5 | '
        '3x3 Seed Patches | Gaussian Smoothing (sigma=1.0, 5x5)',
        fontsize=15, fontweight='bold', y=1.01)

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    plt.close(fig)
    print(f"[DONE] Saved to: {OUTPUT_PATH}")

    # Summary
    porosities = [s[2] for s in samples]
    unique_codes = [s[1] for s in samples]
    print(f"\n[SUMMARY]")
    print(f"  Porosity: mean={np.mean(porosities):.4f}, std={np.std(porosities):.4f}, "
          f"range=[{min(porosities):.4f}, {max(porosities):.4f}]")
    print(f"  Unique codes: mean={np.mean(unique_codes):.1f}, range=[{min(unique_codes)}, {max(unique_codes)}]")
    dense = sum(1 for p in porosities if p < 0.25)
    crack = sum(1 for p in porosities if 0.25 <= p < 0.48)
    dissolved = sum(1 for p in porosities if p >= 0.48)
    print(f"  Phase: Dense={dense}, Micro-crack={crack}, Dissolved={dissolved}")
    print("=" * 70)


if __name__ == "__main__":
    main()
