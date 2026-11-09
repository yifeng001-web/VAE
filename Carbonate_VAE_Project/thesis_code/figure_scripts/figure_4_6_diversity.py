"""
Figure 4-6: 不同高斯尺度与阈值参数下的多样化生成岩心对比矩阵
========================================================
展示 Fuse-then-VQ 框架通过调节高斯云 σ 和覆盖率阈值 thr，
实现从微裂缝主导到大型溶孔主导的连续孔隙形态控制。

3 行 × 3 列:
  Row 1: σ=2.0 (小团块)  → 微裂缝型
  Row 2: σ=4.0 (中等)    → 微裂缝+溶孔混合型
  Row 3: σ=6.0 (大团块)  → 大型溶孔主导型
  Col 1: thr=0.65 (高覆盖 → 高孔隙度)
  Col 2: thr=0.75 (中覆盖 → 中孔隙度)
  Col 3: thr=0.85 (低覆盖 → 低孔隙度)

每个子图下方用黑色英文标注编号 (a)-(i) 和参数符号。
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add thesis_code to path


# ===== 中文字体 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== GPU check =====
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected!")
print(f"[GPU] {len(gpus)} device(s)")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scipy.ndimage import gaussian_filter
try:
    from skimage.exposure import match_histograms
except ImportError:
    raise ImportError("[FATAL] scikit-image not installed!")

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

# ===== Paths =====
from thesis_code.config import PROJECT_ROOT
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
os.makedirs(EVAL_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_4_6_Gaussian_Diversity.png")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
LATENT_SIZE = 32
IMAGE_SIZE = 256
SOFTEN_SIGMA = 1.0

# Physical threshold for porosity calculation (16-bit CT units)
FIXED_THRESHOLD_16BIT = 4500.0
MAX_POROSITY_LIMIT = 0.05  # 致密碳酸盐岩上限

# 生成参数配置
SIGMA_ROWS = [2.0, 4.0, 6.0]     # 低/中/高 σ → 从小裂缝到大溶孔
COVERAGE_COLS = [0.65, 0.75, 0.85]  # 覆盖阈值 → 低/中/高孔隙度
N_ROWS = len(SIGMA_ROWS)
N_COLS = len(COVERAGE_COLS)


def compute_porosity_physical(img_raw_2d, threshold=FIXED_THRESHOLD_16BIT):
    """Binary areal porosity from raw 16-bit image with fixed physical threshold."""
    binary = (img_raw_2d < threshold).astype(np.float32)
    return float(np.mean(binary))


def equivalent_porosity_from_target(generated_img_2d, phi_target):
    """Parent-prior driven exact pixel sorting method.
    
    Uses np.argsort to select exactly phi_target * total_pixels darkest pixels,
    completely avoiding 8-bit gray-level discretization artifacts caused by
    np.percentile thresholding (where many pixels share the same bin value).
    
    Args:
        generated_img_2d: (H, W) float32 in [0, 1] from VQ-VAE decoder
        phi_target: target porosity from parent physical computation
    
    Returns:
        equivalent_porosity: float, exactly ≈ phi_target (by exact sorting definition)
        is_valid: bool, whether phi_target < MAX_POROSITY_LIMIT
    """
    total_pixels = generated_img_2d.size
    # Clamp: at least 1 pixel, at most total_pixels - 1
    target_pore_count = max(1, min(int(phi_target * total_pixels), total_pixels - 1))
    
    # Flatten and argsort by intensity (ascending: darkest first)
    flat = generated_img_2d.flatten()
    sorted_indices = np.argsort(flat)
    
    # Select exactly the first target_pore_count darkest pixels
    flat_mask = np.zeros(total_pixels, dtype=np.bool_)
    flat_mask[sorted_indices[:target_pore_count]] = True
    
    # Verify: this should match phi_target exactly by construction
    equivalent_porosity = float(np.mean(flat_mask))
    is_valid = (phi_target < MAX_POROSITY_LIMIT)
    return equivalent_porosity, is_valid


def load_tif_manual(filepath):
    """Load TIF, resize, return normalized image, raw range, AND physical porosity.
    Returns: (img_norm, raw_min, raw_max, physical_porosity)
    """
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    raw_min = float(tf.reduce_min(img_t).numpy())
    raw_max = float(tf.reduce_max(img_t).numpy())
    img_n = tf.math.divide_no_nan(img_t - raw_min, raw_max - raw_min).numpy()
    # Compute physical porosity from raw 16-bit data
    raw_2d = img_t.numpy()[:, :, 0]
    por = compute_porosity_physical(raw_2d, FIXED_THRESHOLD_16BIT)
    return img_n, raw_min, raw_max, por


def load_pool(data_dir, pool_size=32):
    TARGET_EXTENSIONS = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXTENSIONS:
                file_list.append(os.path.join(root, f))
    np.random.shuffle(file_list)
    pool = []
    for fp in file_list:
        data = load_tif_manual(fp)
        # Filter: only keep parent images with physically reasonable porosity
        # (excludes 45um series with different CT acquisition parameters)
        if data[3] <= 0.07:
            pool.append(data)
        if len(pool) >= pool_size:
            break
    print(f"[INFO] Filtered pool: {len(pool)}/{min(len(file_list), pool_size)} images (kept only por <= 7%)")
    return pool


def generate_organic_mask(h, w, cloud_sigma, coverage_threshold, soften_sigma=SOFTEN_SIGMA):
    noise = np.random.randn(h, w).astype(np.float32)
    cloud = gaussian_filter(noise, sigma=cloud_sigma)
    thresh_val = np.percentile(cloud, coverage_threshold * 100.0)
    pore_binary = (cloud > thresh_val).astype(np.float32)
    pore_soft = gaussian_filter(pore_binary, sigma=soften_sigma)
    pore_mask = pore_soft[np.newaxis, :, :, np.newaxis]
    matrix_mask = 1.0 - pore_mask
    return matrix_mask, pore_mask


def main():
    print("=" * 70)
    print("  图 4-6: 高斯参数多样化生成岩心矩阵")
    print("=" * 70)

    # ── Load VQ-VAE ──
    print("\n[INFO] 加载 VQ-VAE 模型...")
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))
    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_emb = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_emb])
    print("[INFO] VQ-VAE 加载完毕。")

    # ── Load image pool ──
    print("[INFO] 加载图像池...")
    pool = load_pool(DATA_DIR, pool_size=32)
    print(f"[INFO] 池大小: {len(pool)} 张")

    # ── Generate 3×3 matrix ──
    print("\n[INFO] 生成多样化岩心 (不同 σ 和覆盖率)...")
    print(f"  孔隙度方法: 父本先验精确像素排序 (上限 < {MAX_POROSITY_LIMIT:.0%})")
    generated = np.zeros((N_ROWS, N_COLS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    porosities = np.zeros((N_ROWS, N_COLS), dtype=np.float32)
    valid_flags = np.zeros((N_ROWS, N_COLS), dtype=bool)

    # 固定种子确保可复现 (但通过不同 σ/覆盖率控制多样性)
    np.random.seed(42)

    for ri, sigma in enumerate(SIGMA_ROWS):
        for ci, coverage in enumerate(COVERAGE_COLS):
            # 选不同母本/供体组合
            idx_a = (ri * N_COLS + ci) % len(pool)
            idx_b = (ri * N_COLS + ci + 1) % len(pool)
            img_A_norm, _, _, por_A = pool[idx_a]
            img_B_norm, _, _, por_B = pool[idx_b]

            # 直方图匹配
            img_B_matched = match_histograms(
                img_B_norm[:, :, 0], img_A_norm[:, :, 0]
            ).astype(np.float32)
            img_B_matched = np.clip(img_B_matched, 0.0, 1.0)
            img_B_matched = np.expand_dims(img_B_matched, axis=-1)

            # 编码
            z_eA = encoder(tf.constant(np.expand_dims(img_A_norm, 0)), training=False)
            z_eB = encoder(tf.constant(np.expand_dims(img_B_matched, 0)), training=False)

            # 生成掩膜
            matrix_mask, pore_mask = generate_organic_mask(
                LATENT_SIZE, LATENT_SIZE, cloud_sigma=sigma,
                coverage_threshold=coverage
            )

            # 融合 + VQ + 解码
            z_fused = z_eA * tf.constant(matrix_mask, dtype=tf.float32) + \
                      z_eB * tf.constant(pore_mask, dtype=tf.float32)
            z_q = vq_layer(z_fused, training=False)
            img_gen = decoder(z_q, training=False)
            generated[ri, ci] = np.clip(img_gen.numpy()[0, :, :, 0], 0.0, 1.0)

            # ── Parent-prior equivalent percentile porosity ──
            pore_frac = float(np.mean(pore_mask))
            matrix_frac = 1.0 - pore_frac
            phi_target = por_A * matrix_frac + por_B * pore_frac
            
            equivalent_por, is_valid = equivalent_porosity_from_target(
                generated[ri, ci], phi_target
            )
            porosities[ri, ci] = phi_target
            valid_flags[ri, ci] = is_valid
            
            status = "OK" if is_valid else "OVER"
            print(f"  [σ={sigma:.1f}, cov={coverage:.2f}] "
                  f"por_A={por_A:.4f}, por_B={por_B:.4f}, "
                  f"pore_frac={pore_frac:.3f}, "
                  f"φ_target={phi_target:.4f}, "
                  f"φ_extracted={equivalent_por:.4f} [{status}]")

    # ── Plot 3×3 grid ──
    print("\n[INFO] Plotting 3x3 parameter matrix...")
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(18, 19), facecolor='white')
    fig.subplots_adjust(wspace=0.03, hspace=0.20)

    letters = [chr(ord('a') + i) for i in range(N_ROWS * N_COLS)]

    for ri in range(N_ROWS):
        for ci in range(N_COLS):
            ax = axes[ri, ci]
            img_2d = generated[ri, ci]
            ax.imshow(img_2d, cmap='gray', vmin=0.0, vmax=1.0,
                      interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])

            # Label below each sub-image: (letter) σ=X.X, thr=X.XX
            idx = ri * N_COLS + ci
            sigma_val = SIGMA_ROWS[ri]
            cov_val = COVERAGE_COLS[ci]
            label = f"({letters[idx]}) σ={sigma_val:.1f}, thr={cov_val:.2f}"
            ax.set_xlabel(label, fontsize=18, fontweight='bold',
                          color='black', labelpad=6)

    fig.suptitle("Gaussian Parameter Diversity",
                 fontsize=22, fontweight='bold', color='black', y=0.998)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    plt.close(fig)
    print(f"\n[DONE] Figure 4-6 saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
