"""
[VQ-VAE] Carbonate Rock VQ-VAE Training (Final Edition)
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
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
IMAGE_DIR = os.path.join(RESULTS_DIR, "generated_images")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

for d in [CHECKPOINT_DIR, IMAGE_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 50
LATENT_DIM = 64
NUM_EMBEDDINGS = 512

@tf.function
def train_step(encoder, vq_layer, decoder, images, optimizer):
    with tf.GradientTape() as tape:
        z_e = encoder(images, training=True)
        quantized_z = vq_layer(z_e)
        reconstructed = decoder(quantized_z, training=True)

        vq_loss = sum(vq_layer.losses)
        mae_loss = tf.reduce_mean(tf.abs(images - reconstructed))
        ms_ssim_loss = 1.0 - tf.reduce_mean(
            tf.image.ssim_multiscale(images, reconstructed, max_val=1.0, filter_size=7, filter_sigma=1.5)
        )
        porosity_loss = tf.abs(tf.reduce_mean(images) - tf.reduce_mean(reconstructed))
        edge_loss = tf.reduce_mean(tf.square(tf.image.sobel_edges(images) - tf.image.sobel_edges(reconstructed)))

        total_loss = vq_loss + (10.0 * mae_loss) + (5.0 * ms_ssim_loss) + (20.0 * porosity_loss) + (10.0 * edge_loss)

    trainable_vars = encoder.trainable_variables + vq_layer.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return total_loss, vq_loss, mae_loss, ms_ssim_loss, porosity_loss, edge_loss

def generate_16_grid(encoder, vq_layer, decoder, epoch, test_images):
    z_e = encoder(test_images, training=False)
    quantized_z = vq_layer(z_e)
    gen_imgs = decoder(quantized_z, training=False)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < gen_imgs.shape[0]:
            img = tf.clip_by_value(gen_imgs[i, :, :, 0], 0.0, 1.0)
            ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Epoch {epoch} - VQ-VAE Detailed Reconstruction", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, f"grid_epoch_{epoch:03d}.png"), dpi=150)
    plt.close()

def main():
    print("\n" + "=" * 60)
    print(" [VQ-VAE] Carbonate Rock VQ-VAE Training (Final Edition)")
    print(f" [DATA] Data directory: {DATA_DIR}")
    print("=" * 60 + "\n")

    dataset, num_files = create_dataset(DATA_DIR, batch_size=BATCH_SIZE)
    steps_per_epoch = num_files // BATCH_SIZE

    print(f" [INFO] Total TIF files: {num_files}")
    print(f" [INFO] Steps per epoch: {steps_per_epoch} (batch_size={BATCH_SIZE})")
    print(f" [INFO] Total training steps: {steps_per_epoch * EPOCHS}\n")

    # 提取一个固定的测试批次 (16张图) 用于每轮出对比图
    test_batch, _ = next(iter(dataset.unbatch().batch(16).take(1)))

    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, EPOCHS * steps_per_epoch, alpha=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    history = {"loss": [], "vq_loss": [], "mae": [], "ms_ssim": [], "porosity": [], "edge": []}

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        print(f"\n[Epoch {epoch}/{EPOCHS}]")
        epoch_metrics = {k: tf.keras.metrics.Mean() for k in history.keys()}
        step = 0
        
        for image_batch, _ in dataset:
            step += 1
            t_loss, v_loss, m_loss, s_loss, p_loss, e_loss = train_step(encoder, vq_layer, decoder, image_batch, optimizer)
            for k, v in zip(history.keys(), [t_loss, v_loss, m_loss, s_loss, p_loss, e_loss]):
                epoch_metrics[k].update_state(v)

            if step % 50 == 0:
                print(f"  Step {step:03d} | Loss: {t_loss:.3f} | VQ: {v_loss:.3f} | MS-SSIM: {s_loss:.3f} | Porosity: {p_loss:.3f}")

        for k in history.keys():
            history[k].append(epoch_metrics[k].result().numpy())

        print(f"[OK] Epoch {epoch} time {time.time()-start_time:.1f}s | Avg Total Loss: {history['loss'][-1]:.3f}")

        if epoch % 5 == 0 or epoch == 1:
            generate_16_grid(encoder, vq_layer, decoder, epoch, test_batch)
            encoder.save_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{epoch}.weights.h5"))
            decoder.save_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{epoch}.weights.h5"))
            # 保存 VQ 码本权重（layers.Layer 无 save_weights，用 numpy 直接保存 embedding）
            np.save(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{epoch}.npy"), vq_layer.get_weights()[0])

# 5. 绘制最终 Loss 曲线 (修正版)
    print("\n[INFO] 正在生成 Loss 曲线图...")
    # 确保子图布局与指标数量匹配 (2x3=6个指标)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 🌟 修复点：确保每个元组都是 (Key, Title, Color) 三个元素
    metrics_to_plot = [
        ("loss", "Total Loss", "black"), 
        ("vq_loss", "VQ Loss", "magenta"), 
        ("mae", "MAE Loss", "blue"), 
        ("ms_ssim", "MS-SSIM Loss", "green"), 
        ("porosity", "Porosity Loss", "purple"), 
        ("edge", "MSE Edge Loss", "brown")
    ]
    
    # 这里的解包 (key, title, color) 必须对应上面的三个元素
    for ax, (key, title, color) in zip(axes.flat, metrics_to_plot):
        if key in history and len(history[key]) > 0:
            ax.plot(history[key], color=color, linewidth=1.5)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_save_path = os.path.join(PLOTS_DIR, "final_vqvae_loss_curves.png")
    plt.savefig(plot_save_path, dpi=150)
    plt.close()
    print(f"[DONE] Loss curves saved to: {plot_save_path}")

    # Save history data to disk for later plotting
    history_path = os.path.join(RESULTS_DIR, "training_history.npz")
    np.savez(history_path, **history)
    print(f"[SAVED] History data saved to: {history_path}")

if __name__ == "__main__":
    main()