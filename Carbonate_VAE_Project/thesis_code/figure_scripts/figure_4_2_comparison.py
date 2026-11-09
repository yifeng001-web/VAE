"""
图 4-2: 真实 CT / β-VAE (标准 VAE) / 本研究 VQ-VAE 重构对比矩阵
===========================================================
3 行 × N 列灰度图矩阵，展示同一 CT 切片在三种模型下的重构效果：
  - Row 1: 真实 CT (Ground Truth)
  - Row 2: β-VAE (连续潜空间 + KL 散度, KL 退火训练) → 高斯模糊，细节丢失
  - Row 3: 本研究 VQ-VAE (离散码本 + 复合物理约束) → 锐利细节保留

红框高亮微裂缝/细溶孔位置，突出 VQ-VAE 对细节的保留能力。
蓝色虚线框标注 β-VAE 的模糊区域。
白色背景、SimHei 字体、学术风格。
"""
import os, sys

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import sobel

# ===== 中文字体 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== Paths =====
from thesis_code.config import PROJECT_ROOT
from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer
from thesis_code.models.standard_vae import build_standard_encoder, build_standard_decoder

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
# β-VAE (trained with KL annealing, no collapse) checkpoints
VAE_ENC_CKPT = os.path.join(CHECKPOINT_DIR, "standard_vae_enc_beta.weights.h5")
VAE_DEC_CKPT = os.path.join(CHECKPOINT_DIR, "standard_vae_dec_beta.weights.h5")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
os.makedirs(EVAL_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_4_2_VAE_vs_VQVAE_Comparison.png")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
VAE_LATENT_DIM = 128
IMAGE_SIZE = 256
LATENT_SIZE = 32
VQVAE_EPOCH = 50
N_SAMPLES = 6

# GPU
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected!")
print(f"[GPU] {len(gpus)} device(s)")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_vqvae():
    """Load VQ-VAE composite constraint model (epoch 50)."""
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))
    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{VQVAE_EPOCH}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{VQVAE_EPOCH}.weights.h5"))
    vq_emb = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{VQVAE_EPOCH}.npy"))
    vq_layer.set_weights([vq_emb])
    return encoder, decoder, vq_layer


def load_standard_vae():
    """Load beta-VAE (trained with KL annealing, no posterior collapse)."""
    encoder = build_standard_encoder(latent_dim=VAE_LATENT_DIM)
    decoder = build_standard_decoder(latent_dim=VAE_LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, VAE_LATENT_DIM]))
    encoder.load_weights(VAE_ENC_CKPT)
    decoder.load_weights(VAE_DEC_CKPT)
    return encoder, decoder


def load_one_tif(filepath):
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t)
    max_v = tf.reduce_max(img_t)
    img_n = tf.math.divide_no_nan(img_t - min_v, max_v - min_v)
    return img_n.numpy()


def reconstruct_vqvae(encoder, decoder, vq_layer, img_array):
    inp = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(inp, training=False)
    quantized = vq_layer(z_e, training=False)
    recon = decoder(quantized, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)


def reconstruct_standard_vae(encoder, decoder, img_array):
    inp = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_mean, z_log_var = encoder(inp, training=False)
    recon = decoder(z_mean, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)


def detect_micro_crack_region(img_2d):
    """Detect the most prominent micro-crack region using Sobel gradient."""
    from scipy.ndimage import gaussian_filter
    grad_x = sobel(img_2d, axis=0)
    grad_y = sobel(img_2d, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_smooth = gaussian_filter(grad_mag, sigma=3.0)
    max_idx = np.unravel_index(np.argmax(grad_smooth), grad_smooth.shape)
    cy, cx = max_idx
    half = 20
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    w = min(40, IMAGE_SIZE - x1)
    h = min(40, IMAGE_SIZE - y1)
    return x1, y1, w, h


def main():
    print("=" * 70)
    print("  图 4-2: 真实 CT / β-VAE / 本研究 VQ-VAE 重构对比")
    print("=" * 70)

    # ── Load models ──
    print("\n[INFO] 加载 β-VAE (连续潜空间 + KL 退火, 无后验坍塌)...")
    vae_enc, vae_dec = load_standard_vae()
    print("[INFO] β-VAE 加载完毕。")

    print("\n[INFO] 加载本研究 VQ-VAE (离散码本 + 复合物理约束)...")
    vqvae_enc, vqvae_dec, vqvae_vq = load_vqvae()
    print("[INFO] VQ-VAE 加载完毕。")

    # ── Select N samples ──
    tif_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                        if f.lower().endswith(('.tif', '.tiff'))])
    np.random.seed(42)
    selected = np.random.choice(tif_files, min(N_SAMPLES, len(tif_files)), replace=False)

    real_imgs = []
    vae_recons = []
    vqvae_recons = []
    crack_regions = []

    for i, fp in enumerate(selected):
        fname = os.path.basename(fp)
        print(f"  [{i+1}/{len(selected)}] {fname}")
        img = load_one_tif(fp)
        img_2d = img[:, :, 0]
        real_imgs.append(img_2d)

        # β-VAE reconstruction
        r_vae = reconstruct_standard_vae(vae_enc, vae_dec, img)
        vae_recons.append(r_vae)

        # VQ-VAE reconstruction
        r_vqvae = reconstruct_vqvae(vqvae_enc, vqvae_dec, vqvae_vq, img)
        vqvae_recons.append(r_vqvae)

        # Detect micro-crack region (from VQ-VAE recon)
        region = detect_micro_crack_region(r_vqvae)
        crack_regions.append(region)

        # Stats
        mse_vae = np.mean((img_2d - r_vae) ** 2)
        mse_vqvae = np.mean((img_2d - r_vqvae) ** 2)
        print(f"      β-VAE MSE={mse_vae:.5f}  |  VQ-VAE MSE={mse_vqvae:.5f}")

    # ── Plot: 3×N grid ──
    fig, axes = plt.subplots(3, N_SAMPLES, figsize=(3.5 * N_SAMPLES, 10.5),
                             facecolor='white')

    row_labels = ['(a)', '(b)', '(c)']
    all_images = [real_imgs, vae_recons, vqvae_recons]

    for row in range(3):
        for col in range(N_SAMPLES):
            ax = axes[row, col]
            img_2d = all_images[row][col]
            ax.imshow(img_2d, cmap='gray', vmin=0.0, vmax=1.0,
                      interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])

            # ── (a)/(b)/(c) label on left side of each row ──
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=22, fontweight='bold',
                              color='#222222', labelpad=12, rotation=0,
                              va='center', ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=350, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\n[DONE] 图 4-2 已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
