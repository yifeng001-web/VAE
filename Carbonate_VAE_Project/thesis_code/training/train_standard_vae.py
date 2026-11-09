"""
train_standard_vae.py — β-VAE with KL annealing for carbonate rock CT data
============================================================================
Key anti-collapse measures:
  - KL annealing: β = 0 → 0.005 over first WARMUP epochs (linearly)
  - Target β = 0.005 (very small, just enough to regularize the latent space)
  - MAE + MS-SSIM reconstruction loss
  - Cosine LR schedule

Architecture: standard_vae.py (CBAM, latent_dim=128)
Output: standard_vae_enc_beta.weights.h5 / standard_vae_dec_beta.weights.h5
"""
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from thesis_code.utils.data_loader import create_dataset
from thesis_code.models.standard_vae import build_standard_encoder, build_standard_decoder

# ── Paths ──
from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Hyperparams ──
BATCH_SIZE = 4
EPOCHS = 30
LATENT_DIM = 128
IMAGE_SIZE = 256

# KL annealing
WARMUP_EPOCHS = 15       # β linearly ramps from 0 → BETA_TARGET over these epochs
BETA_TARGET = 0.005       # Final β (very small to prevent collapse)
RECON_WEIGHT = 50.0       # Scale up reconstruction loss relative to KL
MS_SSIM_WEIGHT = 10.0

LEARNING_RATE = 3e-4

ENC_CKPT = os.path.join(CHECKPOINT_DIR, "standard_vae_enc_beta.weights.h5")
DEC_CKPT = os.path.join(CHECKPOINT_DIR, "standard_vae_dec_beta.weights.h5")
HISTORY_PATH = os.path.join(RESULTS_DIR, "standard_vae_history.npz")


def kl_divergence(z_mean, z_log_var):
    """KL(N(z_mean, z_log_var) || N(0, 1)) per sample, summed over latent dim."""
    kl = -0.5 * tf.reduce_sum(
        1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1
    )
    return tf.reduce_mean(kl)


def reconstruction_loss(images, reconstructed):
    """MAE + MS-SSIM composite reconstruction loss."""
    mae = tf.reduce_mean(tf.abs(images - reconstructed))
    ms_ssim = 1.0 - tf.reduce_mean(
        tf.image.ssim_multiscale(images, reconstructed, max_val=1.0,
                                 filter_size=7, filter_sigma=1.5)
    )
    return RECON_WEIGHT * mae + MS_SSIM_WEIGHT * ms_ssim


@tf.function
def train_step(encoder, decoder, images, optimizer, beta):
    """Single training step of β-VAE with reparameterization trick."""
    with tf.GradientTape() as tape:
        z_mean, z_log_var = encoder(images, training=True)
        # Reparameterization
        eps = tf.random.normal(tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        reconstructed = decoder(z, training=True)

        recon_loss = reconstruction_loss(images, reconstructed)
        kl_loss = kl_divergence(z_mean, z_log_var)

        total_loss = recon_loss + beta * kl_loss

    trainable_vars = (encoder.trainable_variables + decoder.trainable_variables)
    grads = tape.gradient(total_loss, trainable_vars)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    return total_loss, recon_loss, kl_loss


def generate_grid(encoder, decoder, epoch, test_images, save_dir):
    """Generate a 4x4 (input + recon) grid at current epoch."""
    z_mean, z_log_var = encoder(test_images, training=False)
    recon = decoder(z_mean, training=False)

    fig, axes = plt.subplots(4, 8, figsize=(16, 8), facecolor='white')

    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            # Input in left half
            ax_in = axes[i, j]
            ax_in.imshow(test_images[idx, :, :, 0].numpy(), cmap='gray',
                         vmin=0.0, vmax=1.0)
            ax_in.set_xticks([])
            ax_in.set_yticks([])
            if j == 0:
                ax_in.set_ylabel(f'Row {i+1}', fontsize=9)
            if i == 0:
                ax_in.set_title('Input', fontsize=8)

            # Reconstruction in right half
            ax_out = axes[i, 4 + j]
            ax_out.imshow(recon[idx, :, :, 0].numpy(), cmap='gray',
                          vmin=0.0, vmax=1.0)
            ax_out.set_xticks([])
            ax_out.set_yticks([])
            if i == 0:
                ax_out.set_title('Recon', fontsize=8)

    mse = np.mean((test_images.numpy()[:, :, :, 0] - recon.numpy()[:, :, :, 0]) ** 2)
    plt.suptitle(f'Epoch {epoch} β-VAE  |  MSE={mse:.5f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'standard_vae_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Grid] Saved epoch {epoch} grid → {save_path}")


def get_beta(epoch):
    """Compute current β value with linear warmup annealing."""
    if epoch <= WARMUP_EPOCHS:
        return (epoch / WARMUP_EPOCHS) * BETA_TARGET
    return BETA_TARGET


def main():
    print("\n" + "=" * 65)
    print("  [β-VAE] Standard VAE Training with KL Annealing")
    print(f"  Latent dim: {LATENT_DIM} | β_target: {BETA_TARGET}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs | Recon weight: {RECON_WEIGHT}")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    print("=" * 65 + "\n")

    # ── Data ──
    dataset, num_files = create_dataset(DATA_DIR, batch_size=BATCH_SIZE)
    steps_per_epoch = num_files // BATCH_SIZE
    print(f" [DATA] Files: {num_files}, Steps/epoch: {steps_per_epoch}\n")

    test_dataset, _ = next(iter(dataset.unbatch().batch(16).take(1)))

    # ── Models ──
    encoder = build_standard_encoder(latent_dim=LATENT_DIM)
    decoder = build_standard_decoder(latent_dim=LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_DIM]))

    total_steps = EPOCHS * steps_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        LEARNING_RATE, total_steps, alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    history = {"loss": [], "recon": [], "kl": [], "beta": []}

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        beta = get_beta(epoch)

        epoch_loss = tf.keras.metrics.Mean()
        epoch_recon = tf.keras.metrics.Mean()
        epoch_kl = tf.keras.metrics.Mean()
        step = 0

        for image_batch, _ in dataset:
            step += 1
            t_loss, r_loss, k_loss = train_step(
                encoder, decoder, image_batch, optimizer, beta
            )
            epoch_loss.update_state(t_loss)
            epoch_recon.update_state(r_loss)
            epoch_kl.update_state(k_loss)

            if step % 100 == 0:
                print(f"  Step {step:04d} | Loss: {t_loss:.3f} | "
                      f"Recon: {r_loss:.3f} | KL: {k_loss:.3f}")

        loss_val = epoch_loss.result().numpy()
        recon_val = epoch_recon.result().numpy()
        kl_val = epoch_kl.result().numpy()

        history["loss"].append(loss_val)
        history["recon"].append(recon_val)
        history["kl"].append(kl_val)
        history["beta"].append(beta)

        elapsed = time.time() - start_time
        print(f"[Epoch {epoch:2d}/{EPOCHS}] β={beta:.5f} | "
              f"Loss: {loss_val:.3f} | Recon: {recon_val:.3f} | "
              f"KL: {kl_val:.3f} | {elapsed:.1f}s")

        # Generate grid every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            generate_grid(
                encoder, decoder, epoch, test_dataset,
                os.path.join(RESULTS_DIR, "generated_images")
            )

        # Save checkpoints every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            enc_ep = os.path.join(CHECKPOINT_DIR, f"standard_vae_enc_beta_ep{epoch}.weights.h5")
            dec_ep = os.path.join(CHECKPOINT_DIR, f"standard_vae_dec_beta_ep{epoch}.weights.h5")
            encoder.save_weights(enc_ep)
            decoder.save_weights(dec_ep)
            print(f"  [Ckpt] Saved epoch {epoch}")

    # ── Save final models (encoder + decoder separately) ──
    print(f"\n[SAVE] Saving final models:")
    print(f"  Encoder → {ENC_CKPT}")
    print(f"  Decoder → {DEC_CKPT}")
    encoder.save_weights(ENC_CKPT)
    decoder.save_weights(DEC_CKPT)

    # ── Save history ──
    np.savez(HISTORY_PATH, **history)
    print(f"[SAVE] History → {HISTORY_PATH}")

    # ── Plot training curves ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
    epochs_range = range(1, EPOCHS + 1)

    axes[0].plot(epochs_range, history["loss"], 'k-', linewidth=1.5)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["recon"], 'b-', linewidth=1.5)
    axes[1].set_title('Reconstruction Loss (MAE + MS-SSIM)')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)

    ax2 = axes[2]
    ax2.plot(epochs_range, history["kl"], 'r-', linewidth=1.5, label='KL div')
    ax2.plot(epochs_range, [b * 1000 for b in history["beta"]], 'g--',
             linewidth=1.5, label='β ×1000')
    ax2.set_title('KL Divergence & β Schedule')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('β-VAE Training Curves (Standard VAE with KL Annealing)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    curve_path = os.path.join(PLOTS_DIR, "standard_vae_training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Training curves → {curve_path}")

    print("\n[DONE] Standard β-VAE training completed!")
    print(f"  Encoder: {ENC_CKPT}")
    print(f"  Decoder: {DEC_CKPT}")


if __name__ == "__main__":
    main()
