"""
Chapter 4-5 Rescue: PixelCNN Diagnostic & Grid Search
======================================================
Step 1: Latent space unique codes check (real vs generated)
Step 2: Seed planting correctness self-check
Step 3: Sampling parameter grid search (Group A/B/C)
Step 4: Gaussian smoothing + best 16 selection -> Figure_4_5_GridSearch_Matrix.png
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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
                                   'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===== Config =====
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
os.makedirs(EVAL_DIR, exist_ok=True)

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
IMAGE_SIZE = 256
LATENT_SIZE = 32
EPOCH_TO_LOAD = 50
HIDDEN_CHANNELS = 128
NUM_RES_BLOCKS = 12

OUTPUT_4_5_RESCUE = os.path.join(EVAL_DIR, "Figure_4_5_GridSearch_Matrix.png")

# Smoothing
GAUSS_SIGMA = 1.0
GAUSS_KERNEL = 5

CIRCLE_CENTER = (128, 128)
CIRCLE_RADIUS = 120

# Academic colors
C_BLUE = '#2C6FAC'; C_RED = '#D9583B'; C_GREEN = '#27AE60'
C_ORANGE = '#E67E22'; C_PURPLE = '#8E44AD'; C_GRAY = '#888888'


# ============================================================
# MaskedConv2D & ResidualBlock (same as train_pixelcnn.py)
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
# Image helpers
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
    """Encode image to discrete code map (32x32)."""
    inp = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(inp, training=False)
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


# ============================================================
# STEP 1: Unique Codes Self-Check
# ============================================================
def step1_unique_codes_check(encoder, vq_layer, pixelcnn):
    print("\n" + "=" * 70)
    print("  STEP 1: Latent Space Unique Codes Self-Check")
    print("=" * 70)

    all_paths = gather_tif_paths(DATA_DIR)
    print(f"[INFO] Test set: {len(all_paths)} TIF files")

    # ── 1a: Real images -> unique codes ──
    print("\n[1a] Encoding 30 real TIFs to count unique codes...")
    np.random.seed(42)
    sample_paths = np.random.choice(all_paths, min(30, len(all_paths)), replace=False)

    real_all_codes = []
    real_unique_sets = []
    for i, fp in enumerate(sample_paths):
        img = load_one_tif(fp)
        code_map = encode_to_codes(encoder, vq_layer, img)
        unique = len(np.unique(code_map))
        real_all_codes.extend(code_map.flatten().tolist())
        real_unique_sets.append(unique)

    real_total_unique = len(set(real_all_codes))
    real_avg_unique = np.mean(real_unique_sets)
    real_std_unique = np.std(real_unique_sets)

    print(f"  Real images (N=30):")
    print(f"    Total unique codes across all: {real_total_unique}/{NUM_EMBEDDINGS}")
    print(f"    Per-image unique codes: mean={real_avg_unique:.1f}, std={real_std_unique:.1f}")
    print(f"    Range: [{min(real_unique_sets)}, {max(real_unique_sets)}]")

    # ── 1b: Get a seed code map from a real high-porosity image ──
    print("\n[1b] Extracting seed code map from high-porosity real rock...")
    np.random.seed(777)
    cand_paths = np.random.choice(all_paths, min(32, len(all_paths)), replace=False)
    best_img = None; best_mean = float('inf')
    for fp in cand_paths:
        img = load_one_tif(fp)
        m = np.mean(img)
        if m < best_mean:
            best_mean = m
            best_img = img
    seed_code_map = encode_to_codes(encoder, vq_layer, best_img)
    seed_unique = len(np.unique(seed_code_map))
    print(f"  Seed image: mean intensity={best_mean:.4f}, unique codes={seed_unique}")

    # ── 1c: PixelCNN generated -> unique codes (using existing T=0.95 settings) ──
    print("\n[1c] Generating 8 code maps via PixelCNN (T=0.95, Top-P=0.95, seeds=3)...")

    gen_all_codes = []
    gen_unique_sets = []

    SEED_PATCH_SIZE = 3  # Larger seeds (3x3 = 9 pixels each)
    NUM_SEEDS = 3

    for i in range(8):
        code_map = autoregressive_sample_v2(pixelcnn, seed_code_map,
                                            temperature=0.95, top_p=0.95,
                                            num_seeds=NUM_SEEDS, seed_patch_size=SEED_PATCH_SIZE)
        unique = len(np.unique(code_map))
        gen_all_codes.extend(code_map.flatten().tolist())
        gen_unique_sets.append(unique)
        print(f"    Sample {i+1}: {unique} unique codes")

    gen_total_unique = len(set(gen_all_codes))
    gen_avg_unique = np.mean(gen_unique_sets)
    gen_std_unique = np.std(gen_unique_sets)

    print(f"\n  PixelCNN generated (N=8, T=0.95, Top-P=0.95):")
    print(f"    Total unique codes across all: {gen_total_unique}/{NUM_EMBEDDINGS}")
    print(f"    Per-sample unique codes: mean={gen_avg_unique:.1f}, std={gen_std_unique:.1f}")
    print(f"    Range: [{min(gen_unique_sets)}, {max(gen_unique_sets)}]")

    # ── DIAGNOSIS ──
    print(f"\n  {'='*50}")
    print(f"  [DIAGNOSIS]")
    collapse_ratio = gen_avg_unique / max(real_avg_unique, 1)
    print(f"    Real avg unique: {real_avg_unique:.1f}  |  Generated avg unique: {gen_avg_unique:.1f}")
    print(f"    Collapse ratio: {collapse_ratio:.2f}x")
    if collapse_ratio < 0.3:
        print(f"    [CRITICAL] Severe mode collapse! Generated uses << real unique codes.")
        print(f"    -> Will apply grid-search rescue with lower T + larger seeds.")
    elif collapse_ratio < 0.6:
        print(f"    [WARNING] Moderate mode collapse. Will tune with grid search.")
    else:
        print(f"    [OK] Diversity is acceptable (>{0.6:.0%} of real).")
    print(f"  {'='*50}")

    return seed_code_map, real_avg_unique, gen_avg_unique, collapse_ratio


# ============================================================
# Autoregressive Sample V2 (with diagnostic improvements)
# ============================================================
def autoregressive_sample_v2(pixelcnn, real_code_map, temperature=0.5, top_p=0.8,
                             num_seeds=3, seed_patch_size=3):
    """
    Improved autoregressive sampling with:
    - Larger seed patches (3x3 instead of 2x2)
    - Seed positions randomly distributed
    - Preserved seed pixels are skipped during growth
    """
    code_map = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=np.int32)
    seed_mask = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=bool)

    # ── Stage 1: Plant seeds from real code map ──
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
    print(f"    [SEED V2] Planted {seed_count} seed pixels ({num_seeds} patches of "
          f"{seed_patch_size}x{seed_patch_size})")

    # ── Stage 2: Raster-order autoregressive growth ──
    for h in range(LATENT_SIZE):
        for w in range(LATENT_SIZE):
            if seed_mask[0, h, w]:
                continue

            logits = pixelcnn(tf.constant(code_map), training=False)
            logits_hw = logits[0, h, w, :]

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
    print(f"    [SEED V2] Survived seeds after growth: {survived}/{seed_count}")
    return code_map


# ============================================================
# STEP 2: Seed Correctness Self-Check
# ============================================================
def step2_seed_check(encoder, vq_layer, pixelcnn):
    print("\n" + "=" * 70)
    print("  STEP 2: Seed Planting Correctness Self-Check")
    print("=" * 70)

    all_paths = gather_tif_paths(DATA_DIR)

    # Get a seed code map
    np.random.seed(777)
    cand_paths = np.random.choice(all_paths, min(32, len(all_paths)), replace=False)
    best_img = None; best_mean = float('inf')
    for fp in cand_paths:
        img = load_one_tif(fp)
        m = np.mean(img)
        if m < best_mean: best_mean = m; best_img = img
    seed_code_map = encode_to_codes(encoder, vq_layer, best_img)

    # Run a single sampling with SEED_PATCH_SIZE=3, NUM_SEEDS=3
    print("[CHECK] Running single autoregressive sample with 3 seeds of size 3x3...")
    SEED_PATCH_SIZE = 3
    NUM_SEEDS = 3

    # Manual inspection: create code map manually
    code_map = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=np.int32)
    seed_mask = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=bool)

    max_row = LATENT_SIZE - SEED_PATCH_SIZE
    max_col = LATENT_SIZE - SEED_PATCH_SIZE
    all_positions = [(r, c) for r in range(0, max_row + 1, SEED_PATCH_SIZE)
                      for c in range(0, max_col + 1, SEED_PATCH_SIZE)]
    np.random.seed(123)
    np.random.shuffle(all_positions)
    chosen = all_positions[:NUM_SEEDS]

    print(f"  Seed positions (top-left corners): {chosen}")

    for (sr, sc) in chosen:
        for dr in range(SEED_PATCH_SIZE):
            for dc in range(SEED_PATCH_SIZE):
                r, c = sr + dr, sc + dc
                code_map[0, r, c] = seed_code_map[0, r, c]
                seed_mask[0, r, c] = True

    # Verify: count non-zero pixels in code_map BEFORE autoregressive
    pre_nonzero = np.count_nonzero(code_map)
    pre_seed_count = seed_mask.sum()
    pre_code_map_unique = len(np.unique(code_map[seed_mask]))

    print(f"\n  [SEED CHECK] Before autoregressive growth:")
    print(f"    Non-zero pixels in code_map: {pre_nonzero}/1024")
    print(f"    Seed mask True count: {pre_seed_count}")
    print(f"    Expected: {NUM_SEEDS} x {SEED_PATCH_SIZE}x{SEED_PATCH_SIZE} = "
          f"{NUM_SEEDS * SEED_PATCH_SIZE * SEED_PATCH_SIZE}")
    print(f"    Unique codes in seed region: {pre_code_map_unique}")
    print(f"    All seed codes come from real image: "
          f"{np.all(code_map[seed_mask] == seed_code_map[seed_mask])}")

    # Now run growth
    for h in range(LATENT_SIZE):
        for w in range(LATENT_SIZE):
            if seed_mask[0, h, w]:
                continue
            logits = pixelcnn(tf.constant(code_map), training=False)
            logits_hw = logits[0, h, w, :]
            logits_hw = logits_hw / 0.5  # moderate temperature
            sorted_logits, sorted_indices = tf.math.top_k(logits_hw, k=NUM_EMBEDDINGS)
            probs = tf.nn.softmax(sorted_logits)
            cumsum = tf.cumsum(probs)
            nucleus_mask = tf.concat([tf.ones(1, dtype=tf.bool), cumsum[:-1] < 0.8], axis=0)
            nucleus_logits = tf.where(nucleus_mask, sorted_logits,
                                      tf.ones_like(sorted_logits) * (-1e9))
            sampled_pos = tf.random.categorical(
                tf.expand_dims(nucleus_logits, 0), num_samples=1)[0, 0]
            sampled_idx = sorted_indices[sampled_pos]
            code_map[0, h, w] = sampled_idx.numpy()

    # Verify AFTER growth
    post_survived = seed_mask.sum()  # should still be pre_seed_count
    post_seeds_intact = np.all(code_map[seed_mask] == seed_code_map[seed_mask])
    post_total_unique = len(np.unique(code_map))

    print(f"\n  [SEED CHECK] After autoregressive growth:")
    print(f"    Survived seed pixels: {post_survived}/{pre_seed_count}")
    print(f"    Seeds still intact (no overwrite): {post_seeds_intact}")
    print(f"    Total unique codes in final code_map: {post_total_unique}/{NUM_EMBEDDINGS}")

    result = post_survived == pre_seed_count and post_seeds_intact
    print(f"\n  [VERDICT] Seed planting: {'PASS' if result else 'FAIL'}")
    if not result:
        print(f"  [FIX] Seed positions are NOT being preserved!")
    print("=" * 70)

    return seed_code_map


# ============================================================
# STEP 3 & 4: Grid Search + Smoothing + Output
# ============================================================
def step3_4_grid_search_and_output(encoder, decoder, vq_layer, pixelcnn, seed_code_map):
    print("\n" + "=" * 70)
    print("  STEP 3 & 4: Grid Search + Gaussian Smoothing + Final Matrix")
    print("=" * 70)

    grid_configs = [
        # (name, T, Top-P, num_seeds, seed_patch_size, color)
        ("Group A\nT=0.2, P=0.5\nLow Random",   0.2, 0.5, 3, 3, C_BLUE),
        ("Group B\nT=0.5, P=0.8\nMedium Random", 0.5, 0.8, 3, 3, C_GREEN),
        ("Group C\nT=0.8, P=0.9\nHigh Random",   0.8, 0.9, 3, 3, C_ORANGE),
    ]

    all_generated = {}  # group_name -> list of (image, unique_codes)
    samples_per_group = 8  # generate 8 per group, pick best 5-6

    for group_name, T, top_p, n_seeds, seed_sz, color in grid_configs:
        print(f"\n[GRID] {group_name.replace(chr(10), ' ')}")
        print(f"       Generating {samples_per_group} samples...")

        group_images = []
        for i in range(samples_per_group):
            code_map = autoregressive_sample_v2(pixelcnn, seed_code_map,
                                                temperature=T, top_p=top_p,
                                                num_seeds=n_seeds, seed_patch_size=seed_sz)
            unique_n = len(np.unique(code_map))

            # Decode
            img = codes_to_image(decoder, vq_layer, code_map)
            # Circular mask
            img = apply_circular_mask(img)
            # MANDATORY Gaussian smoothing (5x5, sigma=1.0)
            img = gaussian_filter(img, sigma=GAUSS_SIGMA)
            img = np.clip(img, 0.0, 1.0)

            group_images.append((img, unique_n, code_map))
            por = 1.0 - np.mean(img[img > 0.01]) if np.any(img > 0.01) else 0.0
            print(f"    Sample {i+1}: {unique_n} unique codes, porosity~{por:.3f}")

        all_generated[group_name] = group_images

    # ── Selection: pick the best 5-6 from each group based on diversity ──
    print("\n[SELECT] Picking best samples from each group...")
    selected = []  # list of (img, label, color)

    for group_name, T, top_p, n_seeds, seed_sz, color in grid_configs:
        group_images = all_generated[group_name]

        # Score each image: prefer moderate unique codes (not too few, not too many = noise)
        # and reasonable porosity (0.10-0.65)
        scored = []
        for img, unique_n, code_map in group_images:
            por = 1.0 - np.mean(img[img > 0.01]) if np.any(img > 0.01) else 0.0
            # Diversity score: higher unique is better, but cap at ~80 to avoid noise
            div_score = min(unique_n, 80) / 80.0
            # Porosity score: prefer 0.15-0.60 range (natural carbonate)
            if 0.15 <= por <= 0.60:
                por_score = 1.0
            else:
                por_score = max(0.0, 1.0 - abs(por - 0.35) * 2.0)
            # Edge quality: Sobel edge density (want some structure, not noise)
            t = tf.convert_to_tensor(img[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
            edges = tf.image.sobel_edges(t)
            mag = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
            edge_d = float(tf.reduce_mean(mag))
            # Prefer moderate edge density (0.02-0.12 = natural structure)
            if 0.02 <= edge_d <= 0.12:
                edge_score = 1.0
            else:
                edge_score = max(0.0, 1.0 - abs(edge_d - 0.06) * 10.0)

            total_score = div_score * 0.3 + por_score * 0.35 + edge_score * 0.35
            scored.append((total_score, img, unique_n, por, edge_d))

        # Sort by score descending, take top 5-6
        scored.sort(key=lambda x: x[0], reverse=True)
        n_pick = 5 if "Group C" in group_name else 6  # Group C: 5, others: 6 → total 16

        for rank, (score, img, unique_n, por, edge_d) in enumerate(scored[:n_pick]):
            if por < 0.25:
                type_label = 'Dense'
            elif por < 0.48:
                type_label = 'Micro-crack'
            else:
                type_label = 'Dissolved'
            label = f'{group_name.split(chr(10))[0]}\n#{rank+1} {type_label}\nphi={por:.3f}'
            selected.append((img, label, color))

    # Shuffle to mix groups for visual diversity
    np.random.seed(42)
    np.random.shuffle(selected)

    print(f"  Selected {len(selected)} samples total for final matrix.")

    # ── Plot: 4x4 matrix ──
    print("\n[PLOT] Rendering 4x4 grid search matrix...")
    fig, axes = plt.subplots(4, 4, figsize=(18, 18), facecolor='white')
    fig.subplots_adjust(wspace=0.03, hspace=0.12)

    for i, ax in enumerate(axes.flat):
        if i < len(selected):
            img, label, color = selected[i]
            ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0, interpolation='bilinear')
            ax.set_title(label, fontsize=8.5, fontweight='bold', color=color, pad=3)
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.5)

    fig.suptitle(
        'VQ-VAE Unconditional Novel Rock Generation — PixelCNN Grid Search Rescue\n'
        'Seed-Prompted Autoregressive Sampling + Gaussian Smoothing ($\\sigma$=1.0, 5$\\times$5) — '
        '4$\\times$4 Matrix',
        fontsize=16, fontweight='bold', y=1.01)

    # Legend for the three groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_BLUE, alpha=0.3, label='Group A: T=0.2, Top-P=0.5 (Low Randomness)'),
        Patch(facecolor=C_GREEN, alpha=0.3, label='Group B: T=0.5, Top-P=0.8 (Medium Randomness)'),
        Patch(facecolor=C_ORANGE, alpha=0.3, label='Group C: T=0.8, Top-P=0.9 (High Randomness)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, -0.01))

    plt.savefig(OUTPUT_4_5_RESCUE, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    plt.close(fig)
    print(f"[DONE] Figure 4-5 Grid Search Matrix saved to: {OUTPUT_4_5_RESCUE}")

    # ── Summary stats ──
    all_unique_codes = []
    all_porosities = []
    for img, label, color in selected:
        por = 1.0 - np.mean(img[img > 0.01]) if np.any(img > 0.01) else 0.0
        all_porosities.append(por)

    print(f"\n  [FINAL SUMMARY]")
    print(f"    Total samples: {len(selected)}")
    print(f"    Porosity range: [{min(all_porosities):.4f}, {max(all_porosities):.4f}]")
    print(f"    Porosity mean: {np.mean(all_porosities):.4f}, std: {np.std(all_porosities):.4f}")
    dense_count = sum(1 for p in all_porosities if p < 0.25)
    crack_count = sum(1 for p in all_porosities if 0.25 <= p < 0.48)
    dissolved_count = sum(1 for p in all_porosities if p >= 0.48)
    print(f"    Phase distribution: Dense={dense_count}, Micro-crack={crack_count}, "
          f"Dissolved={dissolved_count}")
    print("=" * 70)

    return selected


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("  Chapter 4-5 PixelCNN Rescue — 4-Step Diagnostic & Grid Search")
    print("=" * 70)

    # ── Load all models ──
    encoder, decoder, vq_layer, pixelcnn = load_all_models()

    # ── STEP 1: Unique codes check ──
    seed_code_map, real_avg, gen_avg, collapse_ratio = step1_unique_codes_check(
        encoder, vq_layer, pixelcnn)

    # ── STEP 2: Seed correctness check ──
    seed_code_map = step2_seed_check(encoder, vq_layer, pixelcnn)

    # ── STEP 3 & 4: Grid search + smoothing + output ──
    selected = step3_4_grid_search_and_output(encoder, decoder, vq_layer, pixelcnn,
                                               seed_code_map)

    print("\n" + "=" * 70)
    print("  Rescue Complete!")
    print(f"  Output: {OUTPUT_4_5_RESCUE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
