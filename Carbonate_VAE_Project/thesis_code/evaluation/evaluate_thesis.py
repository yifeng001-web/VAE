"""
[VQ-VAE] High-Resolution Reconstruction & Comparison Script
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thesis_code.utils.data_loader import create_dataset
from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50

def main():
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    # 初始化计算图
    dummy_input = tf.zeros([1, 256, 256, 1])
    _ = decoder(vq_layer(encoder(dummy_input)))

    # 加载权重（encoder/decoder 用 Keras API，VQ layer 用 numpy 加载 embedding）
    print(f"[INFO] Loading epoch {EPOCH_TO_LOAD} weights...")
    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    # VQ layer (layers.Layer) 通过 numpy 数组恢复码本权重
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])

    dataset, _ = create_dataset(DATA_DIR, batch_size=8)
    test_images, _ = next(iter(dataset))

    # 推理
    reconstructed = decoder(vq_layer(encoder(test_images, training=False)), training=False)

    # 画对比图
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for i in range(8):
        axes[0, i].imshow(test_images[i, :, :, 0], cmap='gray')
        axes[0, i].axis('off')
        img_recon = tf.clip_by_value(reconstructed[i, :, :, 0], 0.0, 1.0)
        axes[1, i].imshow(img_recon, cmap='gray')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "high_res_comparison.png"), dpi=300)
    print(f"[DONE] High-res comparison saved to: {EVAL_DIR}")

if __name__ == "__main__":
    main()