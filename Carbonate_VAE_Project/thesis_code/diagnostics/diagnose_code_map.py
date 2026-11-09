"""
Diagnostic: Print PixelCNN-generated 32x32 index map BEFORE VQ-VAE decoding.
- Count how many indices fall outside the 67 valid training codes ("waste codes").
- Check if Top-P / temperature sampling is producing garbage.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from thesis_code.config import PROJECT_ROOT


from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
CODES_PATH = os.path.join(CHECKPOINT_DIR, "latent_codes.npy")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
IMAGE_SIZE = 256
LATENT_SIZE = 32
HIDDEN_CHANNELS = 128
NUM_RES_BLOCKS = 18

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
    x = tf.keras.layers.Embedding(NUM_EMBEDDINGS, HIDDEN_CHANNELS, name='code_embedding')(inputs)
    x = MaskedConv2D(HIDDEN_CHANNELS, 7, mask_type='A', name='masked_conv_A')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    for i in range(NUM_RES_BLOCKS):
        x = ResidualBlock(HIDDEN_CHANNELS, name=f'res_block_{i}')(x)
    outputs = tf.keras.layers.Conv2D(NUM_EMBEDDINGS, 1, name='output_logits')(x)
    return tf.keras.Model(inputs, outputs, name='PixelCNN_Retrain')


def load_one_tif(filepath):
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t); max_v = tf.reduce_max(img_t)
    return ((img_t - min_v) / (max_v - min_v + 1e-8)).numpy()


def encode_to_codes(encoder, vq_layer, img_array):
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


def autoregressive_sample(pixelcnn, real_code_map, temperature=0.5, top_p=0.5,
                          num_seeds=3, seed_patch_size=3):
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

    print(f"    Seeds planted: {seed_mask.sum()} pixels")

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

    return code_map


def main():
    # ── Load valid codes from training data ──
    print("=" * 70)
    print("  PixelCNN Index Map Diagnostic: Waste-Code Detection")
    print("=" * 70)

    all_codes = np.load(CODES_PATH).astype(np.int32)
    valid_set = set(np.unique(all_codes))
    print(f"\n[VALID CODES] {len(valid_set)} unique codes from training: "
          f"{sorted(valid_set)[:20]}{'...' if len(valid_set) > 20 else ''}")

    # ── Load VQ-VAE ──
    print("\n[INFO] Loading VQ-VAE (Epoch 50)...")
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))
    encoder.load_weights(os.path.join(CHECKPOINT_DIR, "encoder_ep50.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, "decoder_ep50.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, "vq_layer_ep50.npy"))
    vq_layer.set_weights([vq_embedding])
    print("[INFO] VQ-VAE loaded.")

    # ── Load PixelCNN ──
    retrain_weights = os.path.join(CHECKPOINT_DIR, "pixelcnn_retrain.weights.h5")
    if not os.path.exists(retrain_weights):
        print(f"[FATAL] {retrain_weights} not found!")
        return
    pixelcnn = build_pixelcnn_inference()
    _ = pixelcnn(tf.zeros([1, LATENT_SIZE, LATENT_SIZE], dtype=tf.int32))
    pixelcnn.load_weights(retrain_weights)
    print("[INFO] PixelCNN loaded.\n")

    # ── Get seed image ──
    tif_files = [os.path.join(DATA_DIR, f) for f in sorted(os.listdir(DATA_DIR))
                 if f.lower().endswith(('.tif', '.tiff'))]
    np.random.seed(42)
    cand_paths = np.random.choice(tif_files, min(32, len(tif_files)), replace=False)
    best_img = None; best_mean = float('inf')
    for fp in cand_paths:
        img = load_one_tif(fp)
        m = np.mean(img)
        if m < best_mean: best_mean = m; best_img = img
    seed_code_map = encode_to_codes(encoder, vq_layer, best_img)

    # ── Generate 4 code maps and inspect ──
    print("=" * 70)
    print("  Generating 4 code maps for waste-code inspection...")
    print("=" * 70)

    for sample_idx in range(4):
        print(f"\n{'─' * 70}")
        print(f"  SAMPLE {sample_idx + 1}")
        print(f"{'─' * 70}")

        code_map = autoregressive_sample(pixelcnn, seed_code_map,
                                         temperature=0.5, top_p=0.5,
                                         num_seeds=3, seed_patch_size=3)
        flat = code_map.flatten()
        unique_in_map = set(np.unique(flat))

        # Waste code analysis
        waste_codes = unique_in_map - valid_set
        valid_in_map = unique_in_map & valid_set

        print(f"\n  Total pixels: {len(flat)} (32x32)")
        print(f"  Unique codes in map: {len(unique_in_map)}")
        print(f"  Valid codes (in training): {len(valid_in_map)}")
        print(f"  WASTE CODES (never seen!): {len(waste_codes)}")
        if waste_codes:
            print(f"  Waste code values: {sorted(waste_codes)}")
            waste_count = sum(1 for v in flat if v in waste_codes)
            print(f"  Pixels using waste codes: {waste_count}/{len(flat)} "
                  f"({100*waste_count/len(flat):.1f}%)")
        else:
            print(f"  [OK] NO waste codes detected!")

        # Code frequency in map
        from collections import Counter
        freq = Counter(flat)
        top10 = freq.most_common(10)
        print(f"\n  Top-10 most frequent codes in this map:")
        for code, count in top10:
            tag = "[VALID]" if code in valid_set else "[WASTE!]"
            print(f"    Code {code:4d}: {count:5d} pixels ({100*count/len(flat):5.1f}%) {tag}")

        # Print the 32x32 index map
        print(f"\n  Full 32x32 Index Map (sample {sample_idx + 1}):")
        print(f"  {'─' * 50}")
        for row in range(LATENT_SIZE):
            row_str = "  ".join(f"{code_map[0, row, col]:3d}" for col in range(LATENT_SIZE))
            print(f"  [{row_str}]")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"  Valid training codes: {sorted(valid_set)}")
    print(f"  Total valid codes: {len(valid_set)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
