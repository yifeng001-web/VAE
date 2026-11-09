"""
train_sensitivity.py — 孔隙度权重敏感性分析训练脚本
=====================================================
用法:
    python train_sensitivity.py --lambda_porosity 10.0
    python train_sensitivity.py --lambda_porosity 40.0

保持其他参数固定: lambda_MAE=10.0, lambda_SSIM=5.0, lambda_Edge=10.0
仅改变 lambda_porosity，用于绘制参数敏感性分析图 (图 3-X)。
"""

import os
import sys
import time
import argparse
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
SENSITIVITY_DIR = os.path.join(RESULTS_DIR, "sensitivity_checkpoints")

BATCH_SIZE = 4
EPOCHS = 50
LATENT_DIM = 64
NUM_EMBEDDINGS = 512

# 固定权重
W_MAE = 10.0
W_SSIM = 5.0
W_EDGE = 10.0


def build_train_step(w_porosity):
    """构建带指定孔隙度权重的训练步骤."""
    @tf.function
    def train_step(encoder, vq_layer, decoder, images, optimizer):
        with tf.GradientTape() as tape:
            z_e = encoder(images, training=True)
            quantized_z = vq_layer(z_e)
            reconstructed = decoder(quantized_z, training=True)

            vq_loss = sum(vq_layer.losses)
            mae_loss = tf.reduce_mean(tf.abs(images - reconstructed))
            ms_ssim_loss = 1.0 - tf.reduce_mean(
                tf.image.ssim_multiscale(images, reconstructed, max_val=1.0,
                                         filter_size=7, filter_sigma=1.5)
            )
            porosity_loss = tf.abs(tf.reduce_mean(images) - tf.reduce_mean(reconstructed))
            edge_loss = tf.reduce_mean(
                tf.square(tf.image.sobel_edges(images) - tf.image.sobel_edges(reconstructed))
            )

            total_loss = (vq_loss
                          + (W_MAE * mae_loss)
                          + (W_SSIM * ms_ssim_loss)
                          + (w_porosity * porosity_loss)
                          + (W_EDGE * edge_loss))

        trainable_vars = (encoder.trainable_variables
                          + vq_layer.trainable_variables
                          + decoder.trainable_variables)
        grads = tape.gradient(total_loss, trainable_vars)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return total_loss, vq_loss, mae_loss, ms_ssim_loss, porosity_loss, edge_loss

    return train_step


def generate_16_grid(encoder, vq_layer, decoder, epoch, test_images, ckpt_dir):
    """生成 4x4 重构网格."""
    z_e = encoder(test_images, training=False)
    quantized_z = vq_layer(z_e)
    gen_imgs = decoder(quantized_z, training=False)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < gen_imgs.shape[0]:
            img = tf.clip_by_value(gen_imgs[i, :, :, 0], 0.0, 1.0)
            ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.suptitle(f"lambda_porosity={w_porosity:.1f}  Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    grid_path = os.path.join(ckpt_dir, f"grid_epoch_{epoch:03d}.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_porosity', type=float, required=True,
                        help='孔隙度惩罚权重 (e.g. 10.0, 40.0)')
    args = parser.parse_args()
    global w_porosity
    w_porosity = args.lambda_porosity

    # 输出目录
    ckpt_dir = os.path.join(SENSITIVITY_DIR, f"lam_{w_porosity:.1f}")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  敏感性分析训练: lambda_porosity = {w_porosity:.1f}")
    print(f"  固定权重: MAE={W_MAE}  SSIM={W_SSIM}  Edge={W_EDGE}")
    print(f"  输出目录: {ckpt_dir}")
    print("=" * 60 + "\n")

    # 数据
    dataset, num_files = create_dataset(DATA_DIR, batch_size=BATCH_SIZE)
    steps_per_epoch = num_files // BATCH_SIZE
    print(f"[DATA] Total TIF files: {num_files}, steps/epoch: {steps_per_epoch}")

    # 测试 batch
    test_batch, _ = next(iter(dataset.unbatch().batch(16).take(1)))

    # 模型
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        1e-3, EPOCHS * steps_per_epoch, alpha=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    train_step_fn = build_train_step(w_porosity)

    history = {"loss": [], "vq_loss": [], "mae": [], "ms_ssim": [],
               "porosity": [], "edge": []}

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        epoch_metrics = {k: tf.keras.metrics.Mean() for k in history.keys()}
        step = 0

        for image_batch, _ in dataset:
            step += 1
            t_loss, v_loss, m_loss, s_loss, p_loss, e_loss = train_step_fn(
                encoder, vq_layer, decoder, image_batch, optimizer)
            for k, v in zip(history.keys(),
                            [t_loss, v_loss, m_loss, s_loss, p_loss, e_loss]):
                epoch_metrics[k].update_state(v)

        for k in history.keys():
            history[k].append(epoch_metrics[k].result().numpy())

        elapsed = time.time() - start_time
        print(f"[Epoch {epoch:2d}/{EPOCHS}] loss={history['loss'][-1]:.4f}  "
              f"mae={history['mae'][-1]:.4f}  porosity={history['porosity'][-1]:.4f}  "
              f"time={elapsed:.0f}s")

        # 保存 checkpoint
        if epoch % 5 == 0 or epoch == 1:
            generate_16_grid(encoder, vq_layer, decoder, epoch, test_batch, ckpt_dir)
            encoder.save_weights(os.path.join(ckpt_dir, f"encoder_ep{epoch}.weights.h5"))
            decoder.save_weights(os.path.join(ckpt_dir, f"decoder_ep{epoch}.weights.h5"))
            np.save(os.path.join(ckpt_dir, f"vq_layer_ep{epoch}.npy"),
                    vq_layer.get_weights()[0])

    # 保存最终权重和历史
    encoder.save_weights(os.path.join(ckpt_dir, "encoder.weights.h5"))
    decoder.save_weights(os.path.join(ckpt_dir, "decoder.weights.h5"))
    np.save(os.path.join(ckpt_dir, "vq_layer.npy"), vq_layer.get_weights()[0])
    np.savez(os.path.join(ckpt_dir, "history.npz"), **history)

    print(f"\n[DONE] lambda_porosity={w_porosity:.1f} 训练完成，保存至: {ckpt_dir}")


if __name__ == "__main__":
    main()
