"""
train_baseline.py — VQ-VAE Baseline Training (MAE + SSIM ONLY)
==============================================================
Ablation baseline: uses only visual reconstruction losses (MAE + MS-SSIM),
without porosity preservation or edge preservation losses.
All other settings (architecture, hyperparams) identical to train_thesis.py.
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
from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BASELINE_CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "baseline_checkpoints")
IMAGE_DIR = os.path.join(RESULTS_DIR, "generated_images")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [BASELINE_CHECKPOINT_DIR, IMAGE_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 50
LATENT_DIM = 64
NUM_EMBEDDINGS = 512

# Loss weights (baseline: ONLY visual losses)
W_MAE = 10.0
W_SSIM = 5.0


@tf.function
def train_step_baseline(encoder, vq_layer, decoder, images, optimizer):
    """Baseline training step: VQ loss + MAE + MS-SSIM only."""
    with tf.GradientTape() as tape:
        z_e = encoder(images, training=True)
        quantized_z = vq_layer(z_e)
        reconstructed = decoder(quantized_z, training=True)

        vq_loss = sum(vq_layer.losses)
        mae_loss = tf.reduce_mean(tf.abs(images - reconstructed))
        ms_ssim_loss = 1.0 - tf.reduce_mean(
            tf.image.ssim_multiscale(images, reconstructed, max_val=1.0, filter_size=7, filter_sigma=1.5)
        )

        total_loss = vq_loss + (W_MAE * mae_loss) + (W_SSIM * ms_ssim_loss)

    trainable_vars = encoder.trainable_variables + vq_layer.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return total_loss, vq_loss, mae_loss, ms_ssim_loss


def generate_16_grid(encoder, vq_layer, decoder, epoch, test_images, suffix=""):
    z_e = encoder(test_images, training=False)
    quantized_z = vq_layer(z_e)
    gen_imgs = decoder(quantized_z, training=False)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < gen_imgs.shape[0]:
            img = tf.clip_by_value(gen_imgs[i, :, :, 0], 0.0, 1.0)
            ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Epoch {epoch} - Baseline (MAE+SSIM only){suffix}", fontsize=16)
    plt.tight_layout()
    filename = f"baseline_grid_epoch_{epoch:03d}.png"
    plt.savefig(os.path.join(IMAGE_DIR, filename), dpi=150)
    plt.close()


def main():
    print("\n" + "=" * 60)
    print(" [VQ-VAE] Baseline Training — MAE + MS-SSIM ONLY")
    print(" [LOSS]  Total = VQ_Loss + 10*MAE + 5*MS-SSIM")
    print(f" [DATA] Data directory: {DATA_DIR}")
    print("=" * 60 + "\n")

    dataset, num_files = create_dataset(DATA_DIR, batch_size=BATCH_SIZE)
    steps_per_epoch = num_files // BATCH_SIZE

    print(f" [INFO] Total TIF files: {num_files}")
    print(f" [INFO] Steps per epoch: {steps_per_epoch} (batch_size={BATCH_SIZE})")
    print(f" [INFO] Total training steps: {steps_per_epoch * EPOCHS}\n")

    test_batch, _ = next(iter(dataset.unbatch().batch(16).take(1)))

    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, EPOCHS * steps_per_epoch, alpha=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    history = {"loss": [], "vq_loss": [], "mae": [], "ms_ssim": []}

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        print(f"\n[Epoch {epoch}/{EPOCHS}]")
        epoch_metrics = {k: tf.keras.metrics.Mean() for k in history.keys()}
        step = 0

        for image_batch, _ in dataset:
            step += 1
            t_loss, v_loss, m_loss, s_loss = train_step_baseline(
                encoder, vq_layer, decoder, image_batch, optimizer)
            for k, v in zip(history.keys(), [t_loss, v_loss, m_loss, s_loss]):
                epoch_metrics[k].update_state(v)

            if step % 50 == 0:
                print(f"  Step {step:03d} | Loss: {t_loss:.3f} | VQ: {v_loss:.3f} | "
                      f"MAE: {m_loss:.4f} | MS-SSIM: {s_loss:.3f}")

        for k in history.keys():
            history[k].append(epoch_metrics[k].result().numpy())

        print(f"[OK] Epoch {epoch} time {time.time()-start_time:.1f}s | "
              f"Avg Total Loss: {history['loss'][-1]:.3f}")

        if epoch % 5 == 0 or epoch == 1:
            generate_16_grid(encoder, vq_layer, decoder, epoch, test_batch)
            encoder.save_weights(os.path.join(BASELINE_CHECKPOINT_DIR, f"encoder_ep{epoch}.weights.h5"))
            decoder.save_weights(os.path.join(BASELINE_CHECKPOINT_DIR, f"decoder_ep{epoch}.weights.h5"))
            np.save(os.path.join(BASELINE_CHECKPOINT_DIR, f"vq_layer_ep{epoch}.npy"),
                    vq_layer.get_weights()[0])

    # Save final baseline checkpoints
    encoder.save_weights(os.path.join(BASELINE_CHECKPOINT_DIR, "encoder.weights.h5"))
    decoder.save_weights(os.path.join(BASELINE_CHECKPOINT_DIR, "decoder.weights.h5"))
    np.save(os.path.join(BASELINE_CHECKPOINT_DIR, "vq_layer.npy"), vq_layer.get_weights()[0])

    # Draw baseline loss curves
    print("\n[INFO] Generating baseline loss curves...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_to_plot = [
        ("loss", "Total Loss (MAE+SSIM only)", "black"),
        ("vq_loss", "VQ Loss", "magenta"),
        ("mae", "MAE Loss", "blue"),
        ("ms_ssim", "MS-SSIM Loss", "green"),
    ]
    for ax, (key, title, color) in zip(axes.flat, metrics_to_plot):
        if key in history and len(history[key]) > 0:
            ax.plot(history[key], color=color, linewidth=1.5)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_save_path = os.path.join(PLOTS_DIR, "baseline_loss_curves.png")
    plt.savefig(plot_save_path, dpi=150)
    plt.close()
    print(f"[DONE] Baseline loss curves saved to: {plot_save_path}")

    # Save baseline history
    history_path = os.path.join(RESULTS_DIR, "baseline_history.npz")
    np.savez(history_path, **history)
    print(f"[SAVED] Baseline history saved to: {history_path}")


if __name__ == "__main__":
    main()
