"""
figure_3_6_residual.py —— 图 3-6: 基础模型 vs 复合约束模型残差与误差直方图
=========================================================================
上半部分：左右两个残差热力图 (灰度colormap，亮度=误差大小)
下半部分：两个模型的绝对误差分布直方图叠加

仅读取已有模型权重和测试数据，不触发训练。
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ===== 必须字体配置 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

# ===== 路径配置 =====
from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
BASELINE_CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "baseline_checkpoints")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_3_6_Residual.png")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
IMAGE_SIZE = 256
LATENT_SIZE = 32
EPOCH_TO_LOAD = 50
N_CANDIDATE_SCAN = 50

# ===== 配色 =====
COLOR_BASELINE = '#D9583B'   # 红色 — 基础模型
COLOR_PROPOSED = '#27AE60'   # 绿色 — 复合约束模型


def gather_all_tif_paths(data_dir):
    TARGET_EXT = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXT:
                file_list.append(os.path.join(root, f))
    return file_list


def load_and_normalize_tif(filepath):
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_val = tf.reduce_min(img_t)
    max_val = tf.reduce_max(img_t)
    return tf.math.divide_no_nan(img_t - min_val, max_val - min_val).numpy()


def load_model(encoder_path, decoder_path, vq_path):
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))
    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    vq_embedding = np.load(vq_path)
    vq_layer.set_weights([vq_embedding])
    return encoder, decoder, vq_layer


def reconstruct(encoder, decoder, vq_layer, img_array):
    input_tensor = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(input_tensor, training=False)
    quantized_z = vq_layer(z_e)
    recon = decoder(quantized_z, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)


def compute_sobel_edge_density(img_2d):
    tensor = tf.convert_to_tensor(img_2d[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    edges = tf.image.sobel_edges(tensor)
    edge_mag = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
    return float(tf.reduce_mean(edge_mag).numpy())


def scan_complex_slice(all_paths):
    """Sobel 边缘密度扫描，选择最复杂切片 (seed=42 可复现)."""
    print(f"[SCAN] 从 {N_CANDIDATE_SCAN} 张候选切片中筛选最复杂切片...")
    np.random.seed(42)
    np.random.shuffle(all_paths)
    candidates = all_paths[:N_CANDIDATE_SCAN]

    best_path, best_edge, best_porosity = None, -1.0, 0.0
    for i, fp in enumerate(candidates):
        img = load_and_normalize_tif(fp)
        img_2d = img[..., 0]
        edge_dens = compute_sobel_edge_density(img_2d)
        porosity = 1.0 - float(img_2d.mean())
        if edge_dens > best_edge:
            best_edge = edge_dens
            best_path = fp
            best_porosity = porosity
    print(f"[SCAN] 选中: {os.path.basename(best_path)}, "
          f"边缘密度={best_edge:.4f}, 孔隙度={best_porosity:.3f}")
    return best_path


def main():
    print("=" * 60)
    print("  图 3-6: 残差热力图 + 误差直方图")
    print("=" * 60)

    # ── 1. 收集数据路径并选择复杂切片 ──
    all_paths = gather_all_tif_paths(DATA_DIR)
    print(f"[DATA] 数据集总计 {len(all_paths)} 张 TIF")
    display_path = scan_complex_slice(all_paths)

    # ── 2. 加载展示切片 ──
    display_img = load_and_normalize_tif(display_path)
    orig_2d = display_img[..., 0]

    # ── 3. 加载两个模型 ──
    print("[MODEL] 加载 复合约束模型 (Proposed)...")
    p_enc, p_dec, p_vq = load_model(
        os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"),
    )
    print("[MODEL] 加载 基础模型 (Baseline)...")
    b_enc, b_dec, b_vq = load_model(
        os.path.join(BASELINE_CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(BASELINE_CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(BASELINE_CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"),
    )

    # ── 4. 重构 ──
    print("[RECON] 执行重构...")
    recon_b = reconstruct(b_enc, b_dec, b_vq, display_img)
    recon_p = reconstruct(p_enc, p_dec, p_vq, display_img)

    # ── 5. 计算残差 ──
    residual_b = np.abs(orig_2d - recon_b)   # 基础模型绝对残差
    residual_p = np.abs(orig_2d - recon_p)   # 复合模型绝对残差

    mae_b = np.mean(residual_b)
    mae_p = np.mean(residual_p)
    print(f"[METRIC] MAE Baseline={mae_b:.6f}  Proposed={mae_p:.6f}")

    # ═══════════════════════════════════════════
    # 绘图 — GridSpec 上下布局
    # ═══════════════════════════════════════════
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0],
                  hspace=0.38, wspace=0.25,
                  left=0.06, right=0.97, top=0.93, bottom=0.07)

    # 共享 colormap 范围
    vmin = 0
    vmax = max(residual_b.max(), residual_p.max())

    # ═══ 上行左: 基础模型残差 ═══
    ax_bl = fig.add_subplot(gs[0, 0])
    ax_bl.set_facecolor('black')
    im1 = ax_bl.imshow(residual_b, cmap='hot', vmin=vmin, vmax=vmax,
                        aspect='equal', interpolation='bilinear')
    ax_bl.set_title('基础模型残差 (Baseline)', fontsize=13, fontweight='bold',
                    color=COLOR_BASELINE, pad=8)
    ax_bl.axis('off')
    plt.colorbar(im1, ax=ax_bl, fraction=0.046, pad=0.04,
                 label='绝对误差', ticks=np.linspace(vmin, vmax, 5))

    # ═══ 上行右: 复合约束模型残差 ═══
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_pr.set_facecolor('black')
    im2 = ax_pr.imshow(residual_p, cmap='hot', vmin=vmin, vmax=vmax,
                        aspect='equal', interpolation='bilinear')
    ax_pr.set_title('复合约束模型残差 (Proposed)', fontsize=13, fontweight='bold',
                    color=COLOR_PROPOSED, pad=8)
    ax_pr.axis('off')
    plt.colorbar(im2, ax=ax_pr, fraction=0.046, pad=0.04,
                 label='绝对误差', ticks=np.linspace(vmin, vmax, 5))

    # ═══ 下行: 误差直方图 (跨两列) ═══
    ax_hist = fig.add_subplot(gs[1, :])
    ax_hist.set_facecolor('white')

    bins = np.linspace(0, vmax, 80)
    ax_hist.hist(residual_b.ravel(), bins=bins, alpha=0.55,
                 color=COLOR_BASELINE, label=f'基础模型 (MAE={mae_b:.4f})',
                 edgecolor='white', linewidth=0.3)
    ax_hist.hist(residual_p.ravel(), bins=bins, alpha=0.55,
                 color=COLOR_PROPOSED, label=f'复合约束模型 (MAE={mae_p:.4f})',
                 edgecolor='white', linewidth=0.3)

    # 标注均值线
    ax_hist.axvline(mae_b, color=COLOR_BASELINE, linestyle='--', linewidth=1.8,
                    alpha=0.8)
    ax_hist.axvline(mae_p, color=COLOR_PROPOSED, linestyle='--', linewidth=1.8,
                    alpha=0.8)

    ax_hist.set_xlabel('绝对误差', fontsize=14, fontweight='bold')
    ax_hist.set_ylabel('像素频数', fontsize=14, fontweight='bold')
    ax_hist.tick_params(axis='both', labelsize=12)
    ax_hist.legend(loc='upper right', fontsize=11, frameon=True,
                   facecolor='white', edgecolor='#cccccc')

    # 隐藏顶/右边框
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['left'].set_color('#555555')
    ax_hist.spines['bottom'].set_color('#555555')
    ax_hist.grid(axis='y', linestyle='--', alpha=0.25, color='#aaaaaa')

    # 全局标题
    fig.suptitle('重建残差分析：基础模型 vs. 复合约束模型',
                 fontsize=15, fontweight='bold', y=0.985)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"[DONE] 图 3-6 已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
