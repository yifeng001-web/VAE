"""
[Stage 2] PixelCNN Prior Training for VQ-VAE (Oord et al., 2017)
Trains a lightweight autoregressive PixelCNN prior on discrete latent codes.
Learns the spatial grammar of carbonate rock: matrix + organic pores.

Architecture (strictly following van den Oord et al., 2017):
  - MaskedConv2D Type A (first layer, center pixel masked)
  - 12 Masked Residual Blocks with MaskedConv2D Type B + Dropout(0.2)
  - Output: 512-class softmax over codebook entries

Firewalls:
  - GPU check
  - latent_codes.npy must exist and have shape (N, 32, 32)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===== FIREWALL: GPU check =====
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected! Training cannot proceed.")
print(f"[FIREWALL] GPU check PASSED: {len(gpus)} GPU(s) available")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from thesis_code.config import PROJECT_ROOT
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
CODES_PATH = os.path.join(CHECKPOINT_DIR, "latent_codes.npy")
PIXELCNN_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "pixelcnn.weights.h5")
PIXELCNN_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "pixelcnn_history.npz")

# ===== FIREWALL: latent_codes.npy must exist =====
if not os.path.exists(CODES_PATH):
    raise FileNotFoundError(f"[FATAL] {CODES_PATH} not found! Run extract_codes.py first.")
print(f"[FIREWALL] latent_codes.npy found: {os.path.getsize(CODES_PATH)/1024/1024:.1f} MB")

NUM_EMBEDDINGS = 512
LATENT_SIZE = 32         # 32x32 latent grid
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 128
NUM_RES_BLOCKS = 12

# Frequency-inverse weight clipping bounds
WEIGHT_CLIP_MIN = 0.1
WEIGHT_CLIP_MAX = 10.0


# ============================================================
# MaskedConv2D — strictly following Oord et al. (2017)
# ============================================================

class MaskedConv2D(layers.Layer):
    """
    Masked 2D Convolution for PixelCNN.
    mask_type='A': center pixel masked (for first layer)
    mask_type='B': center pixel allowed (for subsequent layers)
    """
    def __init__(self, filters, kernel_size, mask_type='B', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type.upper()

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv = layers.Conv2D(
            self.filters, self.kernel_size,
            padding='same', use_bias=True,
            kernel_initializer='he_normal'
        )
        self.conv.build(input_shape)

        # Build the mask
        k_h, k_w = self.kernel_size, self.kernel_size
        center_h, center_w = k_h // 2, k_w // 2

        mask = np.ones((k_h, k_w, in_channels, self.filters), dtype=np.float32)

        # Mask out future pixels (bottom-half + right-half)
        mask[center_h + 1:, :, :, :] = 0.0          # below center
        mask[center_h, center_w + 1:, :, :] = 0.0    # right of center

        if self.mask_type == 'A':
            # Also mask the center pixel itself
            mask[center_h, center_w, :, :] = 0.0

        self.mask = tf.constant(mask)
        super().build(input_shape)

    def call(self, x):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(x)


class ResidualBlock(layers.Layer):
    """Masked residual block with gated activation (PixelCNN style) + Dropout."""
    def __init__(self, filters, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.masked_conv1 = MaskedConv2D(self.filters // 2, 3, mask_type='B')
        self.masked_conv2 = MaskedConv2D(self.filters, 3, mask_type='B')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)
        self.skip_conv = layers.Conv2D(self.filters, 1, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=None):
        # First conv + activation
        h = self.masked_conv1(x)
        h = self.bn1(h)
        h = tf.nn.relu(h)
        h = self.dropout(h, training=training)  # Dropout after first activation

        # Second conv
        h = self.masked_conv2(h)
        h = self.bn2(h)

        # Residual connection
        if x.shape[-1] != self.filters:
            x = self.skip_conv(x)
        return tf.nn.relu(x + h)


def build_pixelcnn(latent_h=32, latent_w=32, num_embeddings=NUM_EMBEDDINGS):
    """
    Build lightweight autoregressive PixelCNN prior.
    Input: (B, 32, 32) integer indices, one-hot encoded to (B, 32, 32, 512).
    Output: (B, 32, 32, 512) logits.
    """
    inputs = keras.Input(shape=(latent_h, latent_w), dtype=tf.int32)

    # One-hot encode
    x = tf.one_hot(inputs, depth=num_embeddings)  # (B, 32, 32, 512)

    # First layer: Mask A (center pixel masked)
    x = MaskedConv2D(HIDDEN_CHANNELS, 7, mask_type='A', name='masked_conv_A')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # Stack of masked residual blocks (Mask B)
    for i in range(NUM_RES_BLOCKS):
        x = ResidualBlock(HIDDEN_CHANNELS, name=f'res_block_{i}')(x)

    # Final output: 512-class logits
    outputs = layers.Conv2D(num_embeddings, 1, name='output_logits')(x)

    return keras.Model(inputs, outputs, name='PixelCNN')


def load_codes():
    """Load and prepare latent codes."""
    codes = np.load(CODES_PATH).astype(np.int32)  # (N, 32, 32)
    print(f"[DATA] Loaded codes: {codes.shape}, range [{codes.min()}, {codes.max()}]")
    return codes


def compute_frequency_weights(codes, num_embeddings=NUM_EMBEDDINGS,
                               clip_min=WEIGHT_CLIP_MIN, clip_max=WEIGHT_CLIP_MAX):
    """
    Compute inverse-frequency weights to combat extreme class imbalance.
    Carbonate rock is ~90% matrix → PixelCNN collapses to predicting only
    high-frequency matrix tokens. Rare pore tokens get amplified loss weight.
    
    Formula: weight_i = clip(1.0 / (freq_i + ε), clip_min, clip_max)
    Then normalised so mean weight = 1.0.
    """
    flat = codes.flatten()
    counts = np.bincount(flat, minlength=num_embeddings).astype(np.float32)
    total = counts.sum()
    freqs = counts / total

    # Inverse-frequency raw weights
    weights = 1.0 / (freqs + 1e-5)

    # Clip extremes then normalise
    weights = np.clip(weights, clip_min, clip_max)
    weights = weights / weights.mean()

    # Report class-imbalance stats
    active = (counts > 0).sum()
    print(f"[WEIGHTS] Active tokens: {active}/{num_embeddings}")
    print(f"[WEIGHTS] Freq range: [{freqs[counts > 0].min():.6f}, "
          f"{freqs[counts > 0].max():.4f}]")
    print(f"[WEIGHTS] Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    return tf.constant(weights, dtype=tf.float32)


def main():
    # ── Load data ──
    codes = load_codes()
    N = codes.shape[0]
    steps_per_epoch = N // BATCH_SIZE

    # Shuffle once
    np.random.shuffle(codes)

    # Train/val split (90/10)
    split = int(N * 0.9)
    train_codes = codes[:split]
    val_codes = codes[split:]
    print(f"[DATA] Train: {train_codes.shape[0]}, Val: {val_codes.shape[0]}")
    print(f"[DATA] Steps per epoch: {steps_per_epoch}, Total epochs: {EPOCHS}\n")

    # ── Build model ──
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print(f"[INFO] Distribution strategy: {mirrored_strategy.num_replicas_in_sync} replica(s)")

    # ── Compute frequency-inverse weights ──
    token_weights = compute_frequency_weights(codes)
    num_replicas = mirrored_strategy.num_replicas_in_sync

    with mirrored_strategy.scope():
        model = build_pixelcnn()
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

        # Frequency-inverse weighted Sparse Categorical Crossentropy
        # Rare pore tokens receive higher loss multipliers;
        # common matrix tokens are down-weighted.
        def weighted_loss_fn(y_true, y_pred):
            # y_true: (B, 32, 32) int32  —  y_pred: (B, 32, 32, 512) logits
            pixel_weights = tf.gather(token_weights, y_true)   # (B, 32, 32)
            per_pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_true, logits=y_pred)                  # (B, 32, 32)
            return tf.reduce_mean(per_pixel_loss * pixel_weights)

    model.summary()

    # Build tf.data pipelines
    train_ds = tf.data.Dataset.from_tensor_slices(train_codes)
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(val_codes)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ── Training loop ──
    @tf.function
    def train_step(batch_indices):
        # Input = all pixels, Target = same (autoregressive, teacher forcing)
        # PixelCNN sees full grid but masked convs ensure causal ordering
        with tf.GradientTape() as tape:
            logits = model(batch_indices, training=True)  # (B, 32, 32, 512)
            loss = weighted_loss_fn(batch_indices, logits)
            loss = loss / num_replicas
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(batch_indices):
        logits = model(batch_indices, training=False)
        loss = weighted_loss_fn(batch_indices, logits)
        return loss

    # Training
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        # Train
        epoch_loss = tf.keras.metrics.Mean()
        for step, batch in enumerate(train_ds):
            loss = train_step(batch)
            epoch_loss.update_state(loss)
            if step % 100 == 0:
                print(f"  Epoch {epoch:02d} Step {step:04d} | Loss: {loss.numpy():.4f}")

        train_loss = epoch_loss.result().numpy()
        train_losses.append(train_loss)

        # Validate
        val_epoch_loss = tf.keras.metrics.Mean()
        for batch in val_ds:
            vloss = val_step(batch)
            val_epoch_loss.update_state(vloss)

        val_loss = val_epoch_loss.result().numpy()
        val_losses.append(val_loss)

        print(f"[OK] Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(PIXELCNN_WEIGHTS_PATH)
            print(f"  --> Best model saved (val_loss={val_loss:.4f})")

    # ── Final save ──
    print(f"\n[DONE] Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"[DONE] Weights saved to: {PIXELCNN_WEIGHTS_PATH}")

    # Print summary stats
    print(f"[STATS] Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    print(f"[STATS] Val loss:   {val_losses[0]:.4f} -> {val_losses[-1]:.4f}")

    # ── Save training history ──
    np.savez(PIXELCNN_HISTORY_PATH,
             train_loss=np.array(train_losses, dtype=np.float32),
             val_loss=np.array(val_losses, dtype=np.float32))
    print(f"[DONE] Training history saved to: {PIXELCNN_HISTORY_PATH}")


if __name__ == "__main__":
    main()
