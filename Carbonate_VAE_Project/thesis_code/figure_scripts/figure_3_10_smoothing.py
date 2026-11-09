"""
figure_3_10_smoothing.py —— 图 3-10: 边界平滑（Soft Mask/Gaussian Filter）效果对比
==================================================================================
左图：VQ-VAE 直接解码的重构图像（可见 8×8 块状马赛克伪影）
右图：经过 5×5 高斯滤波 (σ=1.0) 平滑后的图像
展示 3.4.2 节中"边界平滑"策略对块效应的抑制效果。
输出：Figure_3_10_Smoothing_Effect.jpg（300 DPI）
"""

import os
import sys
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== 必须字体配置 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 路径 =====
from thesis_code.config import PROJECT_ROOT


from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_3_10_Smoothing_Effect.jpg")

# ===== 模型参数 =====
LATENT_DIM = 64
NUM_EMBEDDINGS = 512
IMAGE_SIZE = 256
LATENT_SIZE = 32
EPOCH_TO_LOAD = 50

# ===== 高斯平滑参数 =====
GAUSS_SIGMA = 1.0      # 标准差
GAUSS_KERNEL = 5        # 核尺寸 (5×5)

# ===== 图像选择参数 =====
N_CANDIDATES = 40       # 随机扫描候选图像数
SEED = 42               # 随机种子（可复现）


def load_one_tif(filepath):
    """加载单张 TIF 并 Min-Max 归一化到 [0,1]."""
    from PIL import Image
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    # 统一到 256×256
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t)
    max_v = tf.reduce_max(img_t)
    img_n = (img_t - min_v) / (max_v - min_v + 1e-8)
    return img_n.numpy()


def load_models():
    """加载 VQ-VAE encoder + decoder + VQ layer."""
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    # 初始化计算图
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])

    return encoder, decoder, vq_layer


def reconstruct(encoder, decoder, vq_layer, img_array):
    """VQ-VAE 前向重构."""
    inp = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(inp, training=False)
    quantized = vq_layer(z_e)
    recon = decoder(quantized, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)


def compute_sobel_edge_density(img_2d):
    """计算 Sobel 边缘密度（用于筛选有结构的图像）."""
    t = tf.convert_to_tensor(img_2d[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    edges = tf.image.sobel_edges(t)
    mag = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
    return float(tf.reduce_mean(mag))


def main():
    # ── 收集 TIF 文件 ──
    print("[INFO] Scanning real_images for TIF files...")
    file_list = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.lower().endswith(('.tif', '.tiff')):
            file_list.append(os.path.join(DATA_DIR, fname))
    print(f"    Found {len(file_list)} TIF files")

    if len(file_list) == 0:
        raise FileNotFoundError(f"[FATAL] No TIF files found in {DATA_DIR}")

    # ── 候选扫描：选一张有代表性的 ──
    print(f"[INFO] Scanning {N_CANDIDATES} candidates (seed={SEED})...")
    np.random.seed(SEED)
    indices = np.random.choice(len(file_list), min(N_CANDIDATES, len(file_list)),
                               replace=False)
    candidates = [file_list[i] for i in indices]

    best_path = None
    best_score = -1.0
    for fp in candidates:
        img = load_one_tif(fp)
        img_2d = img[:, :, 0]
        edge_density = compute_sobel_edge_density(img_2d)
        porosity = 1.0 - np.mean(img_2d)
        # 选择中等边缘密度 + 中等孔隙度的（最有代表性）
        # 偏离 0.5 越远越差
        score = 1.0 - abs(edge_density - 0.22) * 3.0 - abs(porosity - 0.15) * 2.0
        if score > best_score:
            best_score = score
            best_path = fp

    print(f"    Selected: {os.path.basename(best_path)} "
          f"(edge_density scoring={best_score:.4f})")

    # ── 加载选中图像 ──
    real_img = load_one_tif(best_path)
    real_2d = real_img[:, :, 0]
    porosity = 1.0 - np.mean(real_2d)
    edge_density = compute_sobel_edge_density(real_2d)
    print(f"    Properties: porosity={porosity:.4f}, edge_density={edge_density:.4f}")

    # ── 加载模型 ──
    print("[INFO] Loading VQ-VAE models...")
    encoder, decoder, vq_layer = load_models()
    print("[INFO] Models loaded.")

    # ── VQ-VAE 重构 ──
    print("[INFO] Reconstructing via VQ-VAE...")
    recon_raw = reconstruct(encoder, decoder, vq_layer, real_img)
    print(f"    Recon range: [{recon_raw.min():.4f}, {recon_raw.max():.4f}]")

    # ── 高斯平滑 ──
    print(f"[INFO] Applying Gaussian filter (sigma={GAUSS_SIGMA}, kernel={GAUSS_KERNEL}×{GAUSS_KERNEL})...")
    recon_smooth = gaussian_filter(recon_raw, sigma=GAUSS_SIGMA)

    # ── 计算差异 ──
    diff = np.abs(recon_smooth - recon_raw)
    print(f"    Smoothing Δ: mean={diff.mean():.6f}, max={diff.max():.6f}")

    # ═══════════════════════════════════════════════════════════
    # 绘图：左右拼图
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='white')

    # 左图：原始重构（含块效应）
    ax_left = axes[0]
    im_left = ax_left.imshow(recon_raw, cmap='gray', vmin=0.0, vmax=1.0,
                              interpolation='nearest')
    ax_left.set_title('VQ-VAE 直接重构\n(可见 8×8 块状伪影)',
                      fontsize=14, fontweight='bold', pad=10)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # 右图：高斯平滑后
    ax_right = axes[1]
    im_right = ax_right.imshow(recon_smooth, cmap='gray', vmin=0.0, vmax=1.0,
                                interpolation='nearest')
    ax_right.set_title(f'高斯平滑后 (σ={GAUSS_SIGMA}, {GAUSS_KERNEL}×{GAUSS_KERNEL})\n(块状伪影被抑制)',
                       fontsize=14, fontweight='bold', pad=10)
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    # 统一边框
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#888888')

    # 差异标注（右侧底部）
    fig.text(0.75, 0.03,
             f'平滑前后平均差异: {diff.mean():.6f} | 最大差异: {diff.max():.6f}',
             fontsize=9, color='#666666', ha='center')

    # 全局标题
    fig.suptitle('边界平滑策略效果对比 — Gaussian Soft Mask',
                 fontsize=16, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"[DONE] 图 3-10 已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
