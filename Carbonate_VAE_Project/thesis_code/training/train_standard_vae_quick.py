"""
train_standard_vae_quick.py — Quick β-VAE training on real_images subset
=========================================================================
Fast training (10 epochs, ~59 images) just to get a non-collapsed VAE
for Figure 4-2 comparison. Uses direct tf.data pipeline, no 4000+ requirement.
"""
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')


from thesis_code.models.standard_vae import build_standard_encoder, build_standard_decoder

# ── Paths ──
from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Hyperparams ──
BATCH_SIZE = 4
EPOCHS = 15
LATENT_DIM = 128
IMAGE_SIZE = 256

# KL annealing (shorter warmup for faster training)
WARMUP_EPOCHS = 8
BETA_TARGET = 0.005
RECON_WEIGHT = 50.0
MS_SSIM_WEIGHT = 10.0
LEARNING_RATE = 3e-4

ENC_CKPT = os.path.join(CHECKPOINT_DIR, "standard_vae_enc_beta.weights.h5")
DEC_CKPT = os.path.join(CHECKPOINT_DIR, "standard_vae_dec_beta.weights.h5")

# ── Data pipeline ──
def load_image(path):
    img = np.array(Image.open(path), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [256, 256])
    min_v = tf.reduce_min(img_t)
    max_v = tf.reduce_max(img_t)
    img_n = tf.math.divide_no_nan(img_t - min_v, max_v - min_v)
    return img_n

def build_dataset():
    files = sorted([
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    print(f"[DATA] Found {len(files)} images in {DATA_DIR}")
    images = []
    for fp in files:
        images.append(load_image(fp))
    images = tf.stack(images, axis=0)
    ds = tf.data.Dataset.from_tensor_slices((images, images))
    ds = ds.shuffle(200).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds, len(files)


def get_beta(epoch):
    if epoch <= WARMUP_EPOCHS:
        return (epoch / WARMUP_EPOCHS) * BETA_TARGET
    return BETA_TARGET


def reconstruction_loss(images, recon):
    mae = tf.reduce_mean(tf.abs(images - recon))
    ms_ssim = 1.0 - tf.reduce_mean(
        tf.image.ssim_multiscale(images, recon, max_val=1.0,
                                 filter_size=7, filter_sigma=1.5)
    )
    return RECON_WEIGHT * mae + MS_SSIM_WEIGHT * ms_ssim


@tf.function
def train_step(encoder, decoder, images, optimizer, beta):
    with tf.GradientTape() as tape:
        z_mean, z_log_var = encoder(images, training=True)
        eps = tf.random.normal(tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        recon = decoder(z, training=True)
        recon_loss = reconstruction_loss(images, recon)
        kl = tf.reduce_mean(-0.5 * tf.reduce_sum(
            1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total = recon_loss + beta * kl
    grads = tape.gradient(total, encoder.trainable_variables + decoder.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))
    return total, recon_loss, kl


def main():
    print("=" * 65)
    print("  [β-VAE QUICK] Fast training for Figure 4-2")
    print(f"  Latent: {LATENT_DIM} | β={BETA_TARGET} | Warmup: {WARMUP_EPOCHS}ep")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print("=" * 65)

    dataset, n_files = build_dataset()
    steps_per_epoch = n_files // BATCH_SIZE

    encoder = build_standard_encoder(latent_dim=LATENT_DIM)
    decoder = build_standard_decoder(latent_dim=LATENT_DIM)
    _ = encoder(tf.zeros([1, 256, 256, 1]))
    _ = decoder(tf.zeros([1, LATENT_DIM]))

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        LEARNING_RATE, EPOCHS * steps_per_epoch, alpha=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Get a fixed test batch
    all_imgs, _ = next(iter(dataset.unbatch().batch(16).take(1)))
    test_batch = all_imgs[:16]

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        beta = get_beta(epoch)
        avg_loss, avg_recon, avg_kl = 0.0, 0.0, 0.0
        n = 0
        for img_batch, _ in dataset:
            t_loss, r_loss, k_loss = train_step(encoder, decoder, img_batch, optimizer, beta)
            avg_loss += t_loss.numpy()
            avg_recon += r_loss.numpy()
            avg_kl += k_loss.numpy()
            n += 1
        avg_loss /= n
        avg_recon /= n
        avg_kl /= n
        elapsed = time.time() - start
        print(f"[Epoch {epoch:2d}/{EPOCHS}] β={beta:.5f} | Loss={avg_loss:.3f} | "
              f"Recon={avg_recon:.3f} | KL={avg_kl:.4f} | {elapsed:.1f}s")

        # Quick checkpoint every 5
        if epoch % 5 == 0 or epoch == EPOCHS:
            enc_ep = os.path.join(CHECKPOINT_DIR, f"standard_vae_enc_beta_ep{epoch}.weights.h5")
            dec_ep = os.path.join(CHECKPOINT_DIR, f"standard_vae_dec_beta_ep{epoch}.weights.h5")
            encoder.save_weights(enc_ep)
            decoder.save_weights(dec_ep)
            print(f"  [Ckpt] Saved epoch {epoch}")

    # ── Final save ──
    encoder.save_weights(ENC_CKPT)
    decoder.save_weights(DEC_CKPT)
    print(f"\n[DONE] Saved:\n  {ENC_CKPT}\n  {DEC_CKPT}")

    # ── Quick diagnostic ──
    print("\n=== Diagnostic: Test different inputs produce different outputs ===")
    z_means = []
    recons = []
    for i in range(min(6, test_batch.shape[0])):
        inp = test_batch[i:i+1]
        z_mean, _ = encoder(inp, training=False)
        recon = decoder(z_mean, training=False)
        z_means.append(z_mean.numpy())
        recons.append(recon.numpy()[0, :, :, 0])

    for i in range(len(recons)):
        for j in range(i+1, len(recons)):
            diff = np.mean((recons[i] - recons[j]) ** 2)
            print(f"  recon[{i}] vs recon[{j}]: MSE={diff:.6f}")
    print("[DONE] Diagnostic complete.")


if __name__ == "__main__":
    main()
