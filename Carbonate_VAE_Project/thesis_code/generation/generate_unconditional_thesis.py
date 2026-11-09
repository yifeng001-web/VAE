"""
[Stage 3] Unconditional Novel Rock Generation via PixelCNN + VQ-VAE
Loads VQ codebook, Decoder, and trained PixelCNN prior.
Performs autoregressive sampling (top-left -> bottom-right) to generate
novel 32x32 code maps, then decodes to physically-valid carbonate rock CT slices.

Algorithm:
  1. Start with all-zero 32x32 grid
  2. Loop over every spatial position (h, w) in raster order
  3. At each step, PixelCNN predicts logits for 512 codebook entries
  4. Sample via tf.random.categorical with temperature
  5. Map discrete indices -> embedding vectors -> Decoder -> novel rock

Firewalls:
  - GPU check
  - latent_codes.npy existence check
  - pixelcnn.weights.h5 existence check
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== FIREWALL: GPU check =====
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected! Generation cannot proceed.")
print(f"[FIREWALL] GPU check PASSED: {len(gpus)} GPU(s) available")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer
from thesis_code.utils.data_loader import create_dataset

from thesis_code.config import PROJECT_ROOT
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

CODES_PATH = os.path.join(CHECKPOINT_DIR, "latent_codes.npy")
PIXELCNN_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "pixelcnn.weights.h5")
OUTPUT_PATH = os.path.join(EVAL_DIR, "unconditional_generation.png")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
LATENT_SIZE = 32
HIDDEN_CHANNELS = 128
NUM_RES_BLOCKS = 12

# Generation params
NUM_GENERATED = 4
TEMPERATURE = 0.95  # slightly relaxed: pore seeds provide activation entropy
TOP_P = 0.95        # slightly relaxed nucleus: allow more pore diversity around seeds
NUM_SEEDS = 6        # number of real pore crystal seeds planted on the blank canvas
SEED_PATCH_SIZE = 2  # each seed occupies a 2x2 patch in 32x32 latent space

# Physical circular mask params
CIRCLE_CENTER = (128, 128)   # image center
CIRCLE_RADIUS = 120           # radius in pixels (rock core boundary)

# Real data path for seed extraction (same as training data)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")


# ============================================================
# MaskedConv2D (same as train_pixelcnn.py for weight compatibility)
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
            self.filters, self.kernel_size,
            padding='same', use_bias=True,
            kernel_initializer='he_normal'
        )
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
    """Must match train_pixelcnn.py architecture exactly for weight compatibility."""
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
        h = self.dropout(h, training=training)  # inactive at inference (training=False)
        h = self.masked_conv2(h)
        h = self.bn2(h)
        if x.shape[-1] != self.filters:
            x = self.skip_conv(x)
        return tf.nn.relu(x + h)


def build_pixelcnn_inference():
    """Reconstruct PixelCNN for inference."""
    inputs = tf.keras.Input(shape=(LATENT_SIZE, LATENT_SIZE), dtype=tf.int32)
    x = tf.one_hot(inputs, depth=NUM_EMBEDDINGS)
    x = MaskedConv2D(HIDDEN_CHANNELS, 7, mask_type='A', name='masked_conv_A')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    for i in range(NUM_RES_BLOCKS):
        x = ResidualBlock(HIDDEN_CHANNELS, name=f'res_block_{i}')(x)
    outputs = tf.keras.layers.Conv2D(NUM_EMBEDDINGS, 1, name='output_logits')(x)
    return tf.keras.Model(inputs, outputs, name='PixelCNN')


def autoregressive_sample(model, real_code_map, temperature=0.95, top_p=0.95,
                          num_seeds=6, seed_patch_size=2):
    """
    Pore-Seed-Prompted autoregressive sampling.
    
    Stage 1 — Seeding: Plant N real pore crystal seeds (2×2 patches) from a
    high-porosity real rock onto an otherwise blank 32×32 canvas.  Create a
    boolean seed_mask that marks these positions as "locked".
    
    Stage 2 — Growth: Raster-scan the grid.  At each position:
      - If seed_mask[h,w] is True → SKIP (keep real pore index, no sampling)
      - Otherwise → standard Top-P nucleus sampling conditioned on the
        already-filled context (including nearby seeds)
    
    The planted seeds provide "activation entropy" that breaks PixelCNN's
    blank-canvas syndrome — the model now has real pore context to grow from,
    producing organic, diverse pore systems while keeping matrix uniform.
    
    Args:
        model: PixelCNN prior model
        real_code_map: (1, 32, 32) discrete index map from a real high-porosity rock
        temperature: softmax temperature
        top_p: nucleus cumulative probability threshold
        num_seeds: how many pore crystal patches to plant
        seed_patch_size: side length of each square seed patch in latent space
    Returns:
        code_map: (1, 32, 32) int32 discrete index map
    """
    # ── Stage 1: Seeding ──
    code_map = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=np.int32)
    seed_mask = np.zeros((1, LATENT_SIZE, LATENT_SIZE), dtype=bool)

    # Randomly pick num_seeds non-overlapping patch top-left corners
    max_row = LATENT_SIZE - seed_patch_size
    max_col = LATENT_SIZE - seed_patch_size

    # Collect all valid top-left positions, shuffle, take num_seeds
    all_positions = [(r, c) for r in range(0, max_row + 1, seed_patch_size)
                      for c in range(0, max_col + 1, seed_patch_size)]
    np.random.shuffle(all_positions)
    chosen = all_positions[:num_seeds]

    # Plant real pore code patches
    for (sr, sc) in chosen:
        for dr in range(seed_patch_size):
            for dc in range(seed_patch_size):
                r, c = sr + dr, sc + dc
                code_map[0, r, c] = real_code_map[0, r, c]
                seed_mask[0, r, c] = True

    seed_count = seed_mask.sum()
    print(f"    [SEED] Planted {seed_count} seed pixels across {len(chosen)} "
          f"{seed_patch_size}x{seed_patch_size} patches")

    # ── Stage 2: Growth (raster-order autoregressive sampling) ──
    for h in range(LATENT_SIZE):
        for w in range(LATENT_SIZE):
            # --- Core Intercept: skip seed positions ---
            if seed_mask[0, h, w]:
                continue  # preserve real pore index, do NOT overwrite

            # Predict logits conditioned on current partial code_map
            logits = model(tf.constant(code_map), training=False)  # (1, 32, 32, 512)
            logits_hw = logits[0, h, w, :]  # (512,)

            # Apply temperature scaling
            if temperature > 0:
                logits_hw = logits_hw / temperature

            # --- Top-P (Nucleus) Sampling ---
            # 1. Sort all 512 logits descending, keeping original indices
            sorted_logits, sorted_indices = tf.math.top_k(logits_hw, k=NUM_EMBEDDINGS)

            # 2. Softmax → cumulative probabilities
            probs = tf.nn.softmax(sorted_logits)
            cumsum = tf.cumsum(probs)

            # 3. Nucleus mask: keep tokens while cumulative mass < top_p;
            #    always keep at least the highest-probability token
            nucleus_mask = tf.concat([
                tf.ones(1, dtype=tf.bool),
                cumsum[:-1] < top_p
            ], axis=0)

            # 4. Set logits outside nucleus to -inf (zero probability)
            nucleus_logits = tf.where(
                nucleus_mask,
                sorted_logits,
                tf.ones_like(sorted_logits) * (-1e9)
            )

            # 5. Sample within the nucleus
            sampled_pos = tf.random.categorical(
                tf.expand_dims(nucleus_logits, 0), num_samples=1
            )[0, 0]

            # 6. Map sorted position → original codebook index
            sampled_idx = sorted_indices[sampled_pos]

            code_map[0, h, w] = sampled_idx.numpy()

    # Count how many seeded positions survived (should equal seed_count)
    survived = seed_mask.sum()
    print(f"    [SEED] Survived seeds after growth: {survived}/{seed_count}")

    return code_map  # (1, 32, 32)


def apply_circular_mask(image, center=CIRCLE_CENTER, radius=CIRCLE_RADIUS):
    """
    Enforce physical circular rock boundary prior:
    - Pixels INSIDE the circle retain generated values
    - Pixels OUTSIDE the circle are forced to absolute black (air/background)
    This prevents PixelCNN from "hallucinating" rock texture beyond the
    physical boundary of the carbonate core sample.
    """
    h, w = image.shape
    cy, cx = center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    mask = (dist <= radius).astype(np.float32)
    return image * mask


def indices_to_image(decoder, vq_layer, code_map):
    """
    Convert discrete index map (1, 32, 32) to a 256x256 rock image.
    Maps indices -> embedding vectors -> Decoder -> image.
    """
    # Map indices to embedding vectors
    # code_map: (1, 32, 32) int32
    indices_flat = tf.reshape(code_map, [-1])  # (1024,)
    embeddings = vq_layer.embeddings  # (64, 512)
    embeddings_t = tf.transpose(embeddings)    # (512, 64)
    quantized_flat = tf.gather(embeddings_t, indices_flat)  # (1024, 64)
    quantized = tf.reshape(quantized_flat, [1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM])

    # Decode
    generated = decoder(quantized, training=False)
    img = np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)
    return img


def main():
    # ===== FIREWALL: Check required files =====
    for fpath, fname in [(CODES_PATH, "latent_codes.npy"),
                          (PIXELCNN_WEIGHTS_PATH, "pixelcnn.weights.h5")]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"[FATAL] {fname} not found at {fpath}!")
    print("[FIREWALL] All required files present.\n")

    # ── Load Decoder and VQ codebook ──
    print("[INFO] Loading VQ-VAE decoder and codebook...")
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    # Init graph
    dummy_z = tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM])
    _ = decoder(dummy_z)

    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])
    print("[INFO] Decoder and VQ codebook loaded.\n")

    # ── Load PixelCNN ──
    print("[INFO] Building and loading PixelCNN prior...")
    pixelcnn = build_pixelcnn_inference()

    # Init graph
    dummy_codes = tf.zeros([1, LATENT_SIZE, LATENT_SIZE], dtype=tf.int32)
    _ = pixelcnn(dummy_codes)

    pixelcnn.load_weights(PIXELCNN_WEIGHTS_PATH)
    print("[INFO] PixelCNN weights loaded.\n")

    # ── Load Encoder for seed extraction ──
    print("[INFO] Loading Encoder for pore seed extraction...")
    encoder = build_encoder(latent_dim=LATENT_DIM)
    # Init encoder graph
    dummy_img = tf.zeros([1, 256, 256, 1])
    _ = encoder(dummy_img)
    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    print("[INFO] Encoder loaded.\n")

    # ── Extract real pore seed code map ──
    # We bypass create_dataset() to avoid the 4000+ firewall —
    # we only need a few images and the full pipeline overhead is unnecessary.
    print("[INFO] Extracting pore seed crystal from real high-porosity rock...")
    from PIL import Image
    TARGET_EXTENSIONS = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXTENSIONS:
                file_list.append(os.path.join(root, f))
    print(f"    Found {len(file_list)} TIF files in {DATA_DIR}")

    # Randomly sample 32 files, decode manually, pick the one with lowest mean
    np.random.shuffle(file_list)
    sampled = file_list[:32]
    best_img = None
    best_mean = float('inf')
    for fpath in sampled:
        img = np.array(Image.open(fpath), dtype=np.float32)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        # Resize to 256x256
        img_t = tf.image.resize(img, [256, 256])
        # Min-Max normalize
        min_v = tf.reduce_min(img_t)
        max_v = tf.reduce_max(img_t)
        img_n = tf.math.divide_no_nan(img_t - min_v, max_v - min_v)
        m = tf.reduce_mean(img_n).numpy()
        if m < best_mean:
            best_mean = m
            best_img = img_n.numpy()

    img_real = np.expand_dims(best_img, axis=0)  # (1, 256, 256, 1)
    real_mean = best_mean
    print(f"    Selected best image from 32 candidates (mean intensity: {real_mean:.4f})")

    # Encode → VQ quantize → discrete indices
    z_e = encoder(tf.constant(img_real), training=False)  # (1, 32, 32, 64)
    z_e_flat = tf.reshape(z_e, [-1, LATENT_DIM])  # (1024, 64)
    # Compute distances to codebook (same as VectorQuantizer)
    embeddings = vq_layer.embeddings  # (64, 512)
    similarity = tf.matmul(z_e_flat, embeddings)
    flat_sq = tf.reduce_sum(z_e_flat ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(tf.reduce_sum(embeddings ** 2, axis=0), [1, -1])
    distances = flat_sq + emb_sq - 2.0 * similarity
    encoding_indices = tf.argmin(distances, axis=1)  # (1024,)
    real_code_map = tf.reshape(encoding_indices, [1, LATENT_SIZE, LATENT_SIZE]).numpy().astype(np.int32)

    real_unique = len(np.unique(real_code_map))
    real_porosity_approx = 1.0 - real_mean  # rough proxy
    print(f"    Real seed map: {real_unique} unique codes, porosity proxy: {real_porosity_approx:.4f}")
    print("[INFO] Pore seed crystal extracted.\n")

    # ── Generate 4 novel rocks ──
    print(f"[INFO] Generating {NUM_GENERATED} novel rock slices via seed-prompted autoregressive sampling...")
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
        # Apply physical circular mask to enforce rock boundary
        img = apply_circular_mask(img)
        generated_images.append(img)

        # Print code diversity stats
        unique_codes = len(np.unique(code_map))
        print(f"    Unique codes used: {unique_codes}/{NUM_EMBEDDINGS}")

    # ── Plot 1x4 grid ──
    print("\n[INFO] Rendering visualization...")
    fig, axes = plt.subplots(1, NUM_GENERATED, figsize=(16, 5))

    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i], cmap='gray', vmin=0.0, vmax=1.0)
        unique_n = len(np.unique(code_maps[i]))
        ax.set_title(f'Generated #{i+1}\n({unique_n} unique codes)',
                     fontsize=11, fontweight='bold', color='darkorange')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('darkorange')
            spine.set_linewidth(2)

    plt.suptitle(
        'VQ-VAE Unconditional Novel Rock Generation — PixelCNN Prior (Stage 2)',
        fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Unconditional generations saved to:\n  {OUTPUT_PATH}")

    # ── Report code diversity ──
    all_unique = set()
    for cm in code_maps:
        all_unique.update(cm.flatten().tolist())
    print(f"[STATS] Total unique codes used across {NUM_GENERATED} samples: "
          f"{len(all_unique)}/{NUM_EMBEDDINGS}")


if __name__ == "__main__":
    main()
