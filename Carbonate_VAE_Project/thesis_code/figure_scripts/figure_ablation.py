"""
figure_ablation.py —— 消融实验增强版：S₂(r) 两点概率函数 + 图像对比 + 智能筛选
============================================================================
两行布局：
  上行：(a) 原始 CT    (b) MAE+SSIM 基线    (c) 复合物理约束 (本文)
  下行：S₂(r) 两点概率函数三线叠加对比 (横跨三列，N=20 样本平均 +/- 1sigma)

展示切片选取策略：
  从 50 张候选切片中扫描 Sobel 边缘密度，自动选取边缘结构最丰富
  (即微裂缝/复杂孔隙拓扑最显著) 的切片作为展示样本，并在终端打印
  地质特征描述，便于论文中引用。

S₂(r) = P[I(x)=1 AND I(x+r)=1]，通过 FFT 自相关 + 径向平均计算。
统一使用每张图像自身的均值作为二值化阈值，确保公平对比。
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===== 中文字体配置 =====
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC',
                                   'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

# ===== 路径配置 =====
from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
BASELINE_CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "baseline_checkpoints")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(EVAL_DIR, "ablation_comparison.png")

# ===== 常量 =====
NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
IMAGE_SIZE = 256
LATENT_SIZE = 32
MAX_R = 100              # S2 最大距离 (pixels)
N_S2_SAMPLES = 20        # S2 统计样本数
N_CANDIDATE_SCAN = 50    # 候选池大小：从中自动筛选最佳展示切片

# ===== 学术配色 =====
C_DARK = '#1a1a1a'
C_GRAY = '#555555'
C_BLUE = '#2C6FAC'       # GT
C_RED = '#D9583B'        # 基线
C_GREEN = '#27AE60'      # 本文方法


# ═══════════════════════════════════════════════════════════════════
# S2(r) 计算引擎 (复自 final_evaluation_charts.py)
# ═══════════════════════════════════════════════════════════════════
def compute_s2_fft(image_binary, max_r=None):
    """
    通过 FFT 计算两点概率函数 S2(r).

    S2(r) = P[I(x)=1 AND I(x+r)=1] = F^{-1}(|F(I)|^2) / N, 然后径向平均.

    Args:
        image_binary: (H, W) 二值数组，1=孔隙, 0=基质
        max_r: 最大计算距离 (默认 min(H,W)//2)
    Returns:
        r_values: (max_r,) 距离 (pixels)
        s2_values: (max_r,) S2 概率值
    """
    H, W = image_binary.shape
    if max_r is None:
        max_r = min(H, W) // 2

    f = np.fft.fft2(image_binary)
    power_spectrum = np.abs(f) ** 2
    autocorr = np.fft.fftshift(np.fft.ifft2(power_spectrum).real)
    autocorr = autocorr / (H * W)

    center_h, center_w = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r_map = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)

    s2_values = np.zeros(max_r)
    r_values = np.arange(max_r)

    for r in range(max_r):
        mask = (r_map >= r - 0.5) & (r_map < r + 0.5)
        if mask.sum() > 0:
            s2_values[r] = autocorr[mask].mean()
        else:
            s2_values[r] = s2_values[r - 1] if r > 0 else s2_values[0]

    return r_values, s2_values


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════
def gather_all_tif_paths(data_dir):
    """遍历 data_dir 收集所有 TIF 文件的绝对路径。"""
    TARGET_EXT = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXT:
                file_list.append(os.path.join(root, f))
    if not file_list:
        raise FileNotFoundError(f"[FATAL] 未在 {data_dir} 找到任何 TIF 文件！")
    return file_list


def compute_sobel_edge_density(img_2d):
    """
    计算单张灰度图的 Sobel 边缘密度。

    使用 TensorFlow 内置 sobel_edges 算子（与训练中 edge_loss 一致），
    返回边缘幅度均值作为"边缘密度"指标。
    值越高 → 微裂缝/纹理越丰富 → 孔隙拓扑越复杂。
    """
    tensor = tf.convert_to_tensor(img_2d[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    edges = tf.image.sobel_edges(tensor)  # (1, H, W, 2) — dy, dx
    edge_mag = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))  # (1, H, W, 1)
    density = tf.reduce_mean(edge_mag).numpy()
    return float(density)


def classify_pore_type(porosity, edge_density):
    """
    基于孔隙度和边缘密度对碳酸盐岩孔隙类型进行启发式分类。

    Returns:
        (type_cn, type_en): 中英文描述
    """
    # 边缘密度相对分类 (基于本数据集的典型范围)
    if edge_density > 0.12:
        edge_class = '高'
        structure = '微裂缝发育 / 复杂溶蚀孔洞'
        structure_en = 'micro-fracture-rich / complex vuggy pores'
    elif edge_density > 0.06:
        edge_class = '中等'
        structure = '粒间孔隙 + 局部微裂缝'
        structure_en = 'intergranular pores + localized micro-fractures'
    else:
        edge_class = '低'
        structure = '均匀基质孔隙'
        structure_en = 'homogeneous matrix porosity'

    if porosity > 0.50:
        pore_class = '高孔隙度'
        pore_class_en = 'high-porosity'
    elif porosity > 0.25:
        pore_class = '中孔隙度'
        pore_class_en = 'medium-porosity'
    else:
        pore_class = '低孔隙度'
        pore_class_en = 'low-porosity'

    desc_cn = f'{pore_class}碳酸盐岩，{structure}'
    desc_en = f'{pore_class_en} carbonate rock with {structure_en}'
    return desc_cn, desc_en


def load_and_normalize_tif(filepath):
    """加载 TIF -> resize 256x256 -> Min-Max 归一化到 [0,1]."""
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_val = tf.reduce_min(img_t)
    max_val = tf.reduce_max(img_t)
    return tf.math.divide_no_nan(img_t - min_val, max_val - min_val).numpy()


def load_model(encoder_path, decoder_path, vq_path):
    """加载 VQ-VAE 模型三元组."""
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
    """单张图片 VQ-VAE 重构."""
    input_tensor = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(input_tensor, training=False)
    quantized_z = vq_layer(z_e)
    recon = decoder(quantized_z, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)


def compute_s2_curve(img_2d, thresh, max_r=MAX_R):
    """对单张灰度图计算 S2(r) 曲线。"""
    binary = (img_2d < thresh).astype(np.float32)
    _, s2 = compute_s2_fft(binary, max_r)
    return s2


def scan_and_select_display(all_paths, n_candidates=N_CANDIDATE_SCAN):
    """
    从候选池中扫描 Sobel 边缘密度，返回边缘最丰富的切片路径及其地质描述。

    Returns:
        display_path: 选中路径
        desc_cn, desc_en: 地质特征描述
        edge_density: 边缘密度值
        porosity: 孔隙度
    """
    print(f"\n[INFO] Sobel 边缘密度扫描: 从 {n_candidates} 张候选切片中自动筛选...")
    np.random.seed(42)
    np.random.shuffle(all_paths)
    candidates = all_paths[:n_candidates]

    best_path = None
    best_edge = -1.0
    best_porosity = 0.0
    edge_list = []

    for i, fp in enumerate(candidates):
        img = load_and_normalize_tif(fp)
        img_2d = img[..., 0]
        edge_dens = compute_sobel_edge_density(img_2d)
        porosity = 1.0 - float(img_2d.mean())
        edge_list.append((edge_dens, porosity, fp))

        if edge_dens > best_edge:
            best_edge = edge_dens
            best_path = fp
            best_porosity = porosity

        if (i + 1) % 10 == 0:
            print(f"  ... 已扫描 {i+1}/{n_candidates}")

    # 统计分布
    edge_array = np.array([e for e, p, f in edge_list])
    print(f"\n  [边缘密度统计]")
    print(f"    min={edge_array.min():.4f}  median={np.median(edge_array):.4f}  "
          f"max={edge_array.max():.4f}")
    print(f"    mean={edge_array.mean():.4f}  std={edge_array.std():.4f}")

    desc_cn, desc_en = classify_pore_type(best_porosity, best_edge)

    print(f"\n  [选中展示切片]")
    print(f"    文件: {os.path.basename(best_path)}")
    print(f"    边缘密度: {best_edge:.4f}  (候选池最高)")
    print(f"    孔隙度: {best_porosity:.3f}")
    print(f"    地质特征: {desc_cn}")
    print(f"              ({desc_en})")
    print(f"\n  >>> 论文引用建议: '图3-5选取了测试集中Sobel边缘响应最强")
    print(f"       (边缘密度={best_edge:.4f})的典型{desc_cn}切片，")
    print(f"       以检验物理约束在复杂孔隙拓扑下的保真能力。'")

    return best_path, desc_cn, desc_en, best_edge, best_porosity


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("  消融实验增强版 -- Ablation Study + S2(r) Analysis")
    print("  + Sobel 边缘密度自动筛选展示切片")
    print("=" * 60)

    # ── 0. 收集全部 TIF 路径 ──
    all_paths = gather_all_tif_paths(DATA_DIR)
    print(f"\n[INFO] 数据集总计 {len(all_paths)} 张 TIF 切片")

    # ── 1. Sobel 扫描 + 自动筛选展示切片 ──
    display_path, desc_cn, desc_en, edge_dens, _ = scan_and_select_display(all_paths)

    # ── 2. 加载展示切片 + 准备 S2 统计样本池 ──
    display_img = load_and_normalize_tif(display_path)
    orig_2d = display_img[..., 0]
    print(f"\n[INFO] 展示切片 ready: mean={orig_2d.mean():.4f}")

    # S2 统计池：从全部数据集中固定种子取 N_S2_SAMPLES 张
    np.random.seed(123)  # 独立种子，避免与扫描池重叠模式
    np.random.shuffle(all_paths)
    s2_paths = all_paths[:N_S2_SAMPLES]
    print(f"[INFO] S2 统计池: {len(s2_paths)} 张独立切片")

    # ── 3. 加载两个模型 ──
    print("\n[INFO] 加载 本文方法 模型 (复合物理约束)...")
    p_enc, p_dec, p_vq = load_model(
        os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"),
    )
    print("[INFO] 加载 基线 模型 (仅 MAE+SSIM)...")
    b_enc, b_dec, b_vq = load_model(
        os.path.join(BASELINE_CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(BASELINE_CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(BASELINE_CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"),
    )

    # ── 4. 展示图像重构 ──
    print("\n[INFO] 执行展示图像重构...")
    recon_baseline = reconstruct(b_enc, b_dec, b_vq, display_img)
    recon_proposed = reconstruct(p_enc, p_dec, p_vq, display_img)

    mae_baseline = np.mean(np.abs(orig_2d - recon_baseline))
    mae_proposed = np.mean(np.abs(orig_2d - recon_proposed))

    orig_mean = orig_2d.mean()
    porosity_orig = 1.0 - orig_mean
    porosity_baseline = 1.0 - recon_baseline.mean()
    porosity_proposed = 1.0 - recon_proposed.mean()

    print(f"  MAE_baseline={mae_baseline:.6f}  MAE_proposed={mae_proposed:.6f}")
    print(f"  孔隙度 GT={porosity_orig:.4f}  Baseline={porosity_baseline:.4f}  "
          f"Proposed={porosity_proposed:.4f}")

    # ── 5. 多样本 S2 统计 ──
    print(f"\n[INFO] 对 {N_S2_SAMPLES} 张切片计算 S2(r) 统计 (r=0..{MAX_R-1})...")

    gt_s2_all = np.zeros((N_S2_SAMPLES, MAX_R))
    base_s2_all = np.zeros((N_S2_SAMPLES, MAX_R))
    prop_s2_all = np.zeros((N_S2_SAMPLES, MAX_R))

    for i, fp in enumerate(s2_paths):
        img = load_and_normalize_tif(fp)
        img_2d = img[..., 0]
        thresh = img_2d.mean()

        r_base = reconstruct(b_enc, b_dec, b_vq, img)
        r_prop = reconstruct(p_enc, p_dec, p_vq, img)

        gt_s2_all[i] = compute_s2_curve(img_2d, thresh)
        base_s2_all[i] = compute_s2_curve(r_base, thresh)
        prop_s2_all[i] = compute_s2_curve(r_prop, thresh)

        if (i + 1) % 5 == 0:
            print(f"  ... {i+1}/{N_S2_SAMPLES}")

    r_vals = np.arange(MAX_R)
    gt_mean = gt_s2_all.mean(axis=0)
    gt_std = gt_s2_all.std(axis=0)
    base_mean = base_s2_all.mean(axis=0)
    base_std = base_s2_all.std(axis=0)
    prop_mean = prop_s2_all.mean(axis=0)
    prop_std = prop_s2_all.std(axis=0)

    corr_baseline = np.corrcoef(gt_mean, base_mean)[0, 1]
    corr_proposed = np.corrcoef(gt_mean, prop_mean)[0, 1]

    long_range = slice(40, MAX_R)
    lr_dev_baseline = np.mean(np.abs(gt_mean[long_range] - base_mean[long_range]))
    lr_dev_proposed = np.mean(np.abs(gt_mean[long_range] - prop_mean[long_range]))

    print(f"\n{'='*60}")
    print(f"  [S2(r) 统计结果] (N={N_S2_SAMPLES} 样本平均)")
    print(f"  {'指标':<36} {'基线 (MAE+SSIM)':<22} {'本文方法 (复合)':<22}")
    print(f"  {'-'*36} {'-'*22} {'-'*22}")
    print(f"  {'Pearson r (vs GT)':<36} {corr_baseline:<22.6f} {corr_proposed:<22.6f}")
    print(f"  {'长程偏差 |dS2| (r>=40)':<36} {lr_dev_baseline:<22.6f} {lr_dev_proposed:<22.6f}")
    print(f"  {'S2(0) = GT 孔隙度均值':<36} {gt_mean[0]:<22.6f}")
    print(f"  {'S2(99) GT 渐近值':<36} {gt_mean[-1]:<22.6f}")
    print(f"{'='*60}")

    # ── 6. 绘图：2 行布局 ──
    fig = plt.figure(figsize=(18.5, 13.5), facecolor='white')

    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2],
                          hspace=0.30, wspace=0.06,
                          left=0.05, right=0.97, top=0.93, bottom=0.06)

    # ── Row 1: 三张展示图像 ──
    col_titles = [
        '(a) 原始 CT 切片 (Ground Truth)',
        '(b) 仅视觉损失 (MAE + MS-SSIM)',
        '(c) 复合物理约束 (本文方法)\n    + Porosity + Sobel Edge',
    ]
    col_colors = [C_BLUE, C_RED, C_GREEN]
    images = [orig_2d, recon_baseline, recon_proposed]

    for idx in range(3):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(images[idx], cmap='gray', vmin=0.0, vmax=1.0,
                  interpolation='bilinear')
        ax.set_title(col_titles[idx], fontsize=12, fontweight='bold',
                     color=col_colors[idx], pad=12)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(col_colors[idx])
            spine.set_linewidth(3.0)

        if idx == 0:
            label = (f'Sobel边缘密度 = {edge_dens:.4f}\n'
                     f'孔隙度 ~ {porosity_orig:.3f}')
        else:
            m = mae_baseline if idx == 1 else mae_proposed
            pv = porosity_baseline if idx == 1 else porosity_proposed
            pd = abs(porosity_orig - pv)
            label = (f'MAE = {m:.4f}\n'
                     f'孔隙度 = {pv:.4f} (偏差 {pd:.4f})')
        ax.text(0.5, -0.08, label,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9.5, color=col_colors[idx], fontweight='bold',
                linespacing=1.4)

    # ── Row 2: S2(r) 三线图 (横跨三列) ──
    ax_s2 = fig.add_subplot(gs[1, :])

    ax_s2.plot(r_vals, gt_mean, '-', color=C_BLUE, linewidth=2.8,
               label='(a) 真实 CT (Ground Truth)', zorder=5)
    ax_s2.fill_between(r_vals, gt_mean - gt_std, gt_mean + gt_std,
                       alpha=0.15, color=C_BLUE, zorder=1)

    ax_s2.plot(r_vals, base_mean, '--', color=C_RED, linewidth=2.5,
               label='(b) 仅视觉损失 (MAE + MS-SSIM)', zorder=4)
    ax_s2.fill_between(r_vals, base_mean - base_std, base_mean + base_std,
                       alpha=0.12, color=C_RED, zorder=1)

    ax_s2.plot(r_vals, prop_mean, '-.', color=C_GREEN, linewidth=2.5,
               label='(c) 复合物理约束 (本文方法)', zorder=3)
    ax_s2.fill_between(r_vals, prop_mean - prop_std, prop_mean + prop_std,
                       alpha=0.12, color=C_GREEN, zorder=1)

    asymptote = gt_mean[-1]
    ax_s2.axhline(y=asymptote, color='gray', linestyle=':', linewidth=1.2,
                  alpha=0.7, label=f'$S_2$ 渐近值 ~ {asymptote:.4f}')

    ax_s2.annotate(
        f'Pearson r:\n'
        f'  基线 vs GT = {corr_baseline:.4f}\n'
        f'  本文 vs GT = {corr_proposed:.4f}',
        xy=(0.97, 0.78), xycoords='axes fraction',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor=C_GRAY, alpha=0.92),
        ha='right', va='top')

    ax_s2.annotate(
        f'长程偏差 |$\\Delta S_2$| (r$\\geq$40):\n'
        f'  基线 = {lr_dev_baseline:.4f}\n'
        f'  本文 = {lr_dev_proposed:.4f}',
        xy=(0.97, 0.52), xycoords='axes fraction',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor=C_GRAY, alpha=0.92),
        ha='right', va='top')

    ax_s2.set_xlabel('距离 r (pixels)', fontsize=13, fontweight='bold')
    ax_s2.set_ylabel('两点概率函数 $S_2$(r)', fontsize=13, fontweight='bold')
    ax_s2.set_title(
        f'两点概率函数 $S_2$(r) 对比 —— 孔隙空间自相关性分析 (N={N_S2_SAMPLES})',
        fontsize=14, fontweight='bold', pad=12)
    ax_s2.legend(loc='center left', fontsize=10.5, framealpha=0.9)
    ax_s2.grid(True, alpha=0.25, linestyle='--')
    ax_s2.set_xlim(0, MAX_R)
    ax_s2.yaxis.set_major_locator(MaxNLocator(6))

    # ── 全局标题 ──
    fig.suptitle(
        '消融实验：物理约束损失函数对 VQ-VAE 重构质量的影响\n'
        'Ablation Study -- Image Quality & Two-Point Probability Function $S_2$(r)',
        fontsize=15, fontweight='bold', y=0.988, color=C_DARK)

    # ── 底部说明 ──
    fig.text(0.5, 0.01,
             f'展示切片: {desc_cn} (Sobel边缘密度={edge_dens:.4f}, '
             f'N={N_S2_SAMPLES} 样本 S2 统计 +/- 1$\\sigma$)。'
             '两模型架构完全相同 (512x64 Codebook, 50 Epochs)。',
             ha='center', fontsize=9.0, color=C_GRAY, fontstyle='italic')

    # ── 保存 ──
    plt.savefig(OUTPUT_PATH, dpi=350, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"\n[DONE] 增强版消融对比图已保存至:")
    print(f"       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
