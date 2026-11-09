"""
[Stage 2 - Retrain] PixelCNN Prior Retraining (REINFORCED)
===========================================================
Changes from original train_pixelcnn.py:
  - Epochs: 25 -> 200 (progressive, save every 25 epochs)
  - ResBlocks: 12 -> 18 (receptive field: 7 + 18*2 = 43 >> 32, full coverage)
  - Cosine LR decay: 1e-3 -> 1e-5 over 200 epochs
  - Early stopping: patience=30 epochs on val loss
  - Checkpoint every 25 epochs for later analysis

Receptive field analysis:
  Layer 1 (Mask A, 7x7): RF = 7
  Each ResBlock (2x Mask B 3x3): adds 2 pixels of RF per block
  18 ResBlocks: RF = 7 + 18*2 = 43 pixels > 32 (full grid coverage)
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
PIXELCNN_RETRAIN_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "pixelcnn_retrain.weights.h5")
PIXELCNN_RETRAIN_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "pixelcnn_retrain_history.npz")

# ===== FIREWALL: latent_codes.npy must exist =====
if not os.path.exists(CODES_PATH):
    raise FileNotFoundError(f"[FATAL] {CODES_PATH} not found! Run extract_codes.py first.")
print(f"[FIREWALL] latent_codes.npy found: {os.path.getsize(CODES_PATH)/1024/1024:.1f} MB")

NUM_EMBEDDINGS = 512
LATENT_SIZE = 32
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-3
HIDDEN_CHANNELS = 128
NUM_RES_BLOCKS = 18  # was 12, now 18 for RF=43 > 32

# Frequency-inverse weight clipping bounds
WEIGHT_CLIP_MIN = 1.0
WEIGHT_CLIP_MAX = 5.0

print(f"[CONFIG] Epochs: {EPOCHS}, ResBlocks: {NUM_RES_BLOCKS}, RF: 7+{NUM_RES_BLOCKS}*2={7+NUM_RES_BLOCKS*2}")
print(f"[CONFIG] LR: {LEARNING_RATE} (cosine decay), Batch: {BATCH_SIZE}")


# ============================================================
# MaskedConv2D
# ============================================================
class MaskedConv2D(layers.Layer):
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
        h = self.masked_conv1(x)
        h = self.bn1(h)
        h = tf.nn.relu(h)
        h = self.dropout(h, training=training)
        h = self.masked_conv2(h)
        h = self.bn2(h)
        if x.shape[-1] != self.filters:
            x = self.skip_conv(x)
        return tf.nn.relu(x + h)


def build_pixelcnn(latent_h=32, latent_w=32, num_embeddings=NUM_EMBEDDINGS):
    inputs = keras.Input(shape=(latent_h, latent_w), dtype=tf.int32)
    # Bug Fix 1: Use explicit Embedding layer instead of tf.one_hot.
    # tf.one_hot creates (B,32,32,512) causing MaskedConv2D in_channels=512
    # (param-inefficient and architecturally non-standard).
    # Embedding(512, HIDDEN_CHANNELS) produces (B,32,32,128) — correct pattern.
    x = layers.Embedding(num_embeddings, HIDDEN_CHANNELS, name='code_embedding')(inputs)
    x = MaskedConv2D(HIDDEN_CHANNELS, 7, mask_type='A', name='masked_conv_A')(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    for i in range(NUM_RES_BLOCKS):
        x = ResidualBlock(HIDDEN_CHANNELS, name=f'res_block_{i}')(x)
    outputs = layers.Conv2D(num_embeddings, 1, name='output_logits')(x)
    return keras.Model(inputs, outputs, name='PixelCNN_Retrain')


def load_codes():
    codes = np.load(CODES_PATH).astype(np.int32)
    print(f"[DATA] Loaded codes: {codes.shape}, range [{codes.min()}, {codes.max()}]")
    return codes


def compute_frequency_weights(codes, num_embeddings=NUM_EMBEDDINGS,
                               clip_min=WEIGHT_CLIP_MIN, clip_max=WEIGHT_CLIP_MAX):
    flat = codes.flatten()
    counts = np.bincount(flat, minlength=num_embeddings).astype(np.float32)
    total = counts.sum()
    freqs = counts / total
    weights = 1.0 / (freqs + 1e-5)
    weights = np.clip(weights, clip_min, clip_max)
    weights = weights / weights.mean()
    active = (counts > 0).sum()
    print(f"[WEIGHTS] Active tokens: {active}/{num_embeddings}")
    print(f"[WEIGHTS] Freq range: [{freqs[counts > 0].min():.6f}, "
          f"{freqs[counts > 0].max():.4f}]")
    print(f"[WEIGHTS] Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    return tf.constant(weights, dtype=tf.float32)


def compute_prediction_entropy(model, batch_codes, token_weights):
    """
    Compute per-position prediction entropy on a batch.
    Returns mean entropy (nats).
    """
    logits = model(batch_codes, training=False)
    probs = tf.nn.softmax(logits, axis=-1)
    eps = 1e-9
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + eps), axis=-1)
    return tf.reduce_mean(entropy).numpy()


def main():
    codes = load_codes()
    N = codes.shape[0]
    steps_per_epoch = N // BATCH_SIZE
    np.random.shuffle(codes)
    split = int(N * 0.9)
    train_codes = codes[:split]
    val_codes = codes[split:]
    print(f"[DATA] Train: {train_codes.shape[0]}, Val: {val_codes.shape[0]}")
    print(f"[DATA] Steps per epoch: {steps_per_epoch}, Total epochs: {EPOCHS}\n")

    token_weights = compute_frequency_weights(codes)

    model = build_pixelcnn()

    # Cosine decay LR schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        LEARNING_RATE, EPOCHS * steps_per_epoch, alpha=0.01)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    def weighted_loss_fn(y_true, y_pred):
        pixel_weights = tf.gather(token_weights, y_true)
        per_pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
        return tf.reduce_mean(per_pixel_loss * pixel_weights)

    model.summary()

    train_ds = tf.data.Dataset.from_tensor_slices(train_codes)
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(val_codes)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(batch_indices):
        with tf.GradientTape() as tape:
            logits = model(batch_indices, training=True)
            loss = weighted_loss_fn(batch_indices, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(batch_indices):
        logits = model(batch_indices, training=False)
        loss = weighted_loss_fn(batch_indices, logits)
        return loss

    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 30

    train_losses = []
    val_losses = []
    entropy_log = []

    print(f"\n[INFO] Starting training for {EPOCHS} epochs...")
    print(f"[INFO] Early stopping patience: {PATIENCE} epochs")
    print(f"{'='*70}\n")

    for epoch in range(1, EPOCHS + 1):
        # Train
        epoch_loss = tf.keras.metrics.Mean()
        for step, batch in enumerate(train_ds):
            loss = train_step(batch)
            epoch_loss.update_state(loss)
            if step % 100 == 0:
                print(f"  Epoch {epoch:03d} Step {step:04d} | Loss: {loss.numpy():.4f}")

        train_loss = epoch_loss.result().numpy()
        train_losses.append(train_loss)

        # Validate
        val_epoch_loss = tf.keras.metrics.Mean()
        for batch in val_ds:
            vloss = val_step(batch)
            val_epoch_loss.update_state(vloss)

        val_loss = val_epoch_loss.result().numpy()
        val_losses.append(val_loss)

        # Compute prediction entropy every 10 epochs
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            entropy_vals = []
            for batch in val_ds.take(4):
                ent = compute_prediction_entropy(model, batch, token_weights)
                entropy_vals.append(ent)
            epoch_entropy = np.mean(entropy_vals)
            max_entropy = np.log(NUM_EMBEDDINGS)
            norm_entropy = epoch_entropy / max_entropy
            entropy_log.append((epoch, epoch_entropy, norm_entropy))
            print(f"  [ENTROPY] Epoch {epoch:03d}: mean={epoch_entropy:.4f} nats "
                  f"(normalized={norm_entropy:.2%} of max={max_entropy:.4f})")
        else:
            epoch_entropy = None

        print(f"[OK] Epoch {epoch:03d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(PIXELCNN_RETRAIN_WEIGHTS_PATH)
            patience_counter = 0
            print(f"  --> Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[EARLY STOP] Val loss didn't improve for {PATIENCE} epochs.")
                print(f"[EARLY STOP] Stopping at epoch {epoch}/{EPOCHS}")
                break

        # Save checkpoint every 25 epochs
        if epoch % 25 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"pixelcnn_retrain_ep{epoch:03d}.weights.h5")
            model.save_weights(ckpt_path)
            print(f"  --> Checkpoint saved: {ckpt_path}")

    # ── Final ──
    print(f"\n[DONE] Retraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"[DONE] Weights saved to: {PIXELCNN_RETRAIN_WEIGHTS_PATH}")

    print(f"[STATS] Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    print(f"[STATS] Val loss:   {val_losses[0]:.4f} -> {val_losses[-1]:.4f}")

    # ── Print entropy log ──
    if entropy_log:
        print(f"\n[ENTROPY LOG] Prediction entropy across training:")
        print(f"  {'Epoch':>6} | {'Entropy (nats)':>14} | {'Normalized':>10}")
        print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*10}")
        for ep, ent, norm in entropy_log:
            print(f"  {ep:>6} | {ent:>14.4f} | {norm:>9.2%}")

    # ── Save history ──
    np.savez(PIXELCNN_RETRAIN_HISTORY_PATH,
             train_loss=np.array(train_losses, dtype=np.float32),
             val_loss=np.array(val_losses, dtype=np.float32))
    print(f"[DONE] Training history saved to: {PIXELCNN_RETRAIN_HISTORY_PATH}")


if __name__ == "__main__":
    main()
