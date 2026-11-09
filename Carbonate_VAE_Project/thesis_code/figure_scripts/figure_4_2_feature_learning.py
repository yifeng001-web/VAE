"""
figure_4_2_feature_learning.py —— 图 4-2: VQ-VAE 测试集特征表达对比矩阵
========================================================================
加载第三章最优复合约束 VQ-VAE 模型 (lambda3=20.0, lambda4=10.0, Epoch 50)
从测试集 (data/real_images/) 盲抽 8 张真实岩心切片，涵盖致密基质、微裂缝、大溶孔。
上行：真实岩心切片 (Ground Truth)
下行：离散特征表达输出 (Learned Output)
用彩色方框轻微高亮对比明显的细节区域（微裂缝、溶孔边界），保持纯净学术感。
"""

import os
import sys
import numpy as np
import tensorflow as tf
from scipy.ndimage import label as connected_label
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ===== 字体配置 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 路径 =====
from thesis_code.config import PROJECT_ROOT

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_4_2_Feature_Learning.png")

# ===== 模型参数 =====
LATENT_DIM = 64
NUM_EMBEDDINGS = 512
IMAGE_SIZE = 256
LATENT_SIZE = 32
EPOCH_TO_LOAD = 50

# ===== 采样参数 =====
N_TOTAL = 8                  # 最终展示 8 张
N_SCAN = 456                 # 扫描全部测试集
SEED = 42                    # 可复现
TARGET_PER_CAT = [3, 2, 3]   # [dense, crack, dissolved] 每类目标数量

# ===== 高亮框配色 =====
HIGHLIGHT_COLORS = {
    'dense':     '#00AA00',   # 绿色 — 致密基质对比区域
    'crack':     '#E8384F',   # 红色 — 微裂缝
    'dissolved': '#2979FF',   # 蓝色 — 大溶孔边界
}


def load_one_tif(filepath):
    """加载单张 TIF 并 Min-Max 归一化到 [0,1]."""
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t)
    max_v = tf.reduce_max(img_t)
    img_n = (img_t - min_v) / (max_v - min_v + 1e-8)
    return img_n.numpy()


def compute_sobel_edge_density(img_2d):
    """计算 Sobel 边缘密度."""
    t = tf.convert_to_tensor(img_2d[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    edges = tf.image.sobel_edges(t)
    mag = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
    return float(tf.reduce_mean(mag))


def classify_sample(porosity, edge_density, por_p25, por_p75, edge_median):
    """
    基于动态百分位阈值的三分类：
      'dense'     — 致密基质 (低孔隙度)
      'crack'     — 微裂缝 (中低孔隙 + 高边缘密度)
      'dissolved' — 大溶孔 (高孔隙度)
    """
    # 微裂缝：边缘密度显著高于中位数 且 孔隙度不是最高档
    if edge_density > edge_median * 1.35 and porosity < por_p75:
        return 'crack'
    # 溶孔：孔隙度在上四分位以上
    elif porosity > por_p75:
        return 'dissolved'
    # 致密基质：孔隙度在下四分位以下
    elif porosity < por_p25:
        return 'dense'
    # 中间地带：按孔隙度距两端距离判断
    else:
        dist_low = abs(porosity - por_p25)
        dist_high = abs(porosity - por_p75)
        if dist_low < dist_high:
            return 'dense'
        else:
            return 'dissolved'


def load_models():
    """加载复合约束 VQ-VAE (lambda3=20.0, lambda4=10.0) Epoch 50."""
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_emb = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_emb])

    return encoder, decoder, vq_layer


def reconstruct(encoder, decoder, vq_layer, img_array):
    """VQ-VAE 前向重构."""
    inp = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(inp, training=False)
    quantized = vq_layer(z_e)
    recon = decoder(quantized, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)


def find_highlight_boxes(img_2d, category, n_boxes=1):
    """
    为不同类型的岩心自动定位高亮框。
    - crack: 在边缘密度最高的局部区域画框
    - dissolved: 在最大但不过大的连通暗区 (孔隙) 周围画框（限制框尺寸）
    - dense: 在局部对比度最高的随机区域画框（随机偏移防重复）
    返回: list of (y1, x1, y2, x2) 归一化到 0..1
    """
    H, W = img_2d.shape
    boxes = []
    MAX_BOX_FRAC = 0.50  # 框最大不超过图像的 50%

    if category == 'crack':
        # 找 Sobel 边缘幅度最高的局部区域
        t = tf.convert_to_tensor(img_2d[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        edges = tf.image.sobel_edges(t)
        mag = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))[0, :, :, 0].numpy()
        win = 48
        best_sum = -1.0
        best_rc = (0, 0)
        for r in range(0, H - win, win // 3):
            for c in range(0, W - win, win // 3):
                s = mag[r:r+win, c:c+win].sum()
                if s > best_sum:
                    best_sum = s
                    best_rc = (r, c)
        r0, c0 = best_rc
        boxes.append((r0 / H, c0 / W, (r0 + win) / H, (c0 + win) / W))

    elif category == 'dissolved':
        # 找最大的连通暗区 (孔隙)，但使用更严格的阈值避免框过大
        # 使用 P10 而非 P25，确保只捕获真正深的孔隙
        thresh = np.percentile(img_2d, 15)
        binary = (img_2d < thresh).astype(np.int32)
        labeled, n_features = connected_label(binary)
        if n_features > 0:
            # 收集所有连通域，按面积排序，选面积第2大的（最大往往是背景融合区）
            regions = []
            for lbl in range(1, n_features + 1):
                area = (labeled == lbl).sum()
                ys, xs = np.where(labeled == lbl)
                h_box = ys.max() - ys.min()
                w_box = xs.max() - xs.min()
                # 过滤：面积太小忽略，框太大（>50%图像）忽略
                if area < 20 or h_box > H * MAX_BOX_FRAC or w_box > W * MAX_BOX_FRAC:
                    continue
                regions.append((area, ys.min(), xs.min(), ys.max(), xs.max()))

            if regions:
                # 选面积居中的区域（不要最大也不要最小），更可能是有代表性的孔隙
                regions.sort(key=lambda x: x[0], reverse=True)
                # 如果只有1个，用它；否则取第2个（跳过最大的可能是背景）
                idx = 1 if len(regions) >= 2 else 0
                _, y1, x1, y2, x2 = regions[min(idx, len(regions) - 1)]
                pad = 6
                y1 = max(0, y1 - pad)
                x1 = max(0, x1 - pad)
                y2 = min(H, y2 + pad)
                x2 = min(W, x2 + pad)
                boxes.append((y1 / H, x1 / W, y2 / H, x2 / W))
            else:
                # fallback: 在局部暗区中心画固定大小框
                blur = np.mean(img_2d.reshape(H // 16, 16, W // 16, 16), axis=(1, 3))
                br, bc = np.unravel_index(np.argmin(blur), blur.shape)
                cy, cx = int((br + 0.5) * 16), int((bc + 0.5) * 16)
                half = 30
                y1, y2 = max(0, cy - half), min(H, cy + half)
                x1, x2 = max(0, cx - half), min(W, cx + half)
                boxes.append((y1 / H, x1 / W, y2 / H, x2 / W))

    elif category == 'dense':
        # 找局部标准差最大的区域，分区搜索防重复
        win = 44
        # 将图像分成 3x3 网格区域，每区域独立搜索最佳
        grid_r, grid_c = 3, 3
        cell_h, cell_w = H // grid_r, W // grid_c
        all_candidates = []
        for gr in range(grid_r):
            for gc in range(grid_c):
                r_start = gr * cell_h
                c_start = gc * cell_w
                r_end = min(r_start + cell_h, H - win)
                c_end = min(c_start + cell_w, W - win)
                best_std = -1.0
                best_pos = (r_start, c_start)
                for r in range(r_start, r_end, win // 3):
                    for c in range(c_start, c_end, win // 3):
                        s = float(np.std(img_2d[r:r+win, c:c+win]))
                        if s > best_std:
                            best_std = s
                            best_pos = (r, c)
                all_candidates.append((best_std, best_pos[0], best_pos[1]))

        # 按标准差从高到低排序，选第 2-3 名的（避免总是第一）
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        # 确保不同类别取不同位置的框：用 hash 做偏移
        import hashlib
        img_hash = int(hashlib.md5(img_2d.tobytes()).hexdigest()[:8], 16)
        rank = img_hash % min(3, len(all_candidates))
        _, r0, c0 = all_candidates[rank]
        boxes.append((r0 / H, c0 / W, (r0 + win) / H, (c0 + win) / W))

    return boxes


def draw_highlight_box(ax, y1, x1, y2, x2, color, lw=2.5):
    """在 ax 上绘制彩色高亮矩形框."""
    H, W = 256, 256
    rect = mpatches.Rectangle(
        (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
        linewidth=lw, edgecolor=color, facecolor='none',
        linestyle='-', zorder=10
    )
    ax.add_patch(rect)


def main():
    # ============================================================
    # 阶段 1: 收集测试集，先全量扫描并计算动态分类阈值
    # ============================================================
    print("[STAGE 1] Scanning test set (data/real_images/) ...")
    file_list = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.lower().endswith(('.tif', '.tiff')):
            file_list.append(os.path.join(DATA_DIR, fname))
    n_total = len(file_list)
    print(f"  Test set size: {n_total} TIF files")

    if n_total == 0:
        raise FileNotFoundError(f"[FATAL] No TIF files in {DATA_DIR}")

    # 全量扫描，收集孔隙度和边缘密度
    np.random.seed(SEED)
    np.random.shuffle(file_list)
    scan_n = min(N_SCAN, len(file_list))
    candidates = file_list[:scan_n]

    all_porosity = []
    all_edge = []
    records = []  # (fp, img, porosity, edge_density)

    print(f"  Scanning {scan_n} candidates to compute dynamic thresholds...")
    for i, fp in enumerate(candidates):
        img = load_one_tif(fp)
        img_2d = img[:, :, 0]
        porosity = 1.0 - float(np.mean(img_2d))
        edge_density = compute_sobel_edge_density(img_2d)
        all_porosity.append(porosity)
        all_edge.append(edge_density)
        records.append((fp, img, porosity, edge_density))
        if (i + 1) % 100 == 0:
            print(f"    ... {i+1}/{scan_n}")

    all_porosity = np.array(all_porosity)
    all_edge = np.array(all_edge)

    por_p25 = float(np.percentile(all_porosity, 25))
    por_p50 = float(np.percentile(all_porosity, 50))
    por_p75 = float(np.percentile(all_porosity, 75))
    edge_median = float(np.median(all_edge))

    print(f"\n  Dynamic thresholds:")
    print(f"    Porosity:    P25={por_p25:.4f}  P50={por_p50:.4f}  P75={por_p75:.4f}")
    print(f"    Edge density: median={edge_median:.4f}")
    print(f"    Porosity range: [{all_porosity.min():.4f}, {all_porosity.max():.4f}]")
    print(f"    Edge range:     [{all_edge.min():.4f}, {all_edge.max():.4f}]")

    # 分类
    cat_pool = {'dense': [], 'crack': [], 'dissolved': []}
    for fp, img, por, edge in records:
        cat = classify_sample(por, edge, por_p25, por_p75, edge_median)
        cat_pool[cat].append((fp, img, por, edge))

    for cat in ['dense', 'crack', 'dissolved']:
        print(f"  [{cat:10s}] pool size: {len(cat_pool[cat])}")

    # ============================================================
    # 阶段 2: 从每类盲抽目标数量
    # ============================================================
    print(f"\n[STAGE 2] Drawing {N_TOTAL} samples: {TARGET_PER_CAT}")
    selected = []
    cat_names = ['dense', 'crack', 'dissolved']

    for cat, target_n in zip(cat_names, TARGET_PER_CAT):
        pool = cat_pool[cat]
        np.random.shuffle(pool)
        actual_n = min(target_n, len(pool))
        for fp, img, por, edge in pool[:actual_n]:
            selected.append({
                'path': fp,
                'img': img,
                'porosity': por,
                'edge_density': edge,
                'category': cat,
            })
        print(f"  [{cat:10s}] drawn: {actual_n}/{target_n} (pool {len(pool)})")

    # 如果不足 8 张，从剩余池补充
    if len(selected) < N_TOTAL:
        print(f"  Insufficient samples ({len(selected)}/{N_TOTAL}), filling from remaining pool...")
        # 收集未被选中的记录
        selected_paths = {s['path'] for s in selected}
        remaining = [(fp, img, por, edge, cat)
                     for cat in cat_names
                     for fp, img, por, edge in cat_pool[cat]
                     if fp not in selected_paths]
        np.random.shuffle(remaining)
        needed = N_TOTAL - len(selected)
        for fp, img, por, edge, cat in remaining[:needed]:
            selected.append({
                'path': fp, 'img': img,
                'porosity': por, 'edge_density': edge, 'category': cat,
            })

    # 打乱顺序（避免同类聚集）
    np.random.shuffle(selected)
    selected = selected[:N_TOTAL]

    print(f"\n  Final {len(selected)} selected:")
    for i, s in enumerate(selected):
        fname = os.path.basename(s['path'])
        print(f"    [{i+1}] {fname} | cat={s['category']:10s} | "
              f"por={s['porosity']:.4f} | edge={s['edge_density']:.4f}")

    # ============================================================
    # 阶段 3: 加载模型
    # ============================================================
    print("\n[STAGE 3] Loading composite-constraint VQ-VAE model "
          "(lambda3=20.0, lambda4=10.0, Epoch 50)...")
    encoder, decoder, vq_layer = load_models()
    print(f"  Codebook shape: {vq_layer.embeddings.shape}")
    print(f"  Latent dim: {LATENT_DIM}")

    # ============================================================
    # 阶段 4: VQ-VAE 重构
    # ============================================================
    print("\n[STAGE 4] VQ-VAE forward reconstruction (discrete feature learning)...")
    for i, s in enumerate(selected):
        recon = reconstruct(encoder, decoder, vq_layer, s['img'])
        mae = float(np.mean(np.abs(s['img'][:, :, 0] - recon)))
        s['recon'] = recon
        s['mae'] = mae
        print(f"  [{i+1}] {os.path.basename(s['path'])}: MAE={mae:.4f}")

    # ============================================================
    # 阶段 5: 定位高亮框
    # ============================================================
    print("\n[STAGE 5] Locating highlight boxes...")
    for i, s in enumerate(selected):
        boxes = find_highlight_boxes(s['img'][:, :, 0], s['category'], n_boxes=1)
        s['boxes'] = boxes
        box_str = ", ".join(
            [f"({b[0]:.2f},{b[1]:.2f})-({b[2]:.2f},{b[3]:.2f})" for b in boxes])
        print(f"  [{i+1}] {s['category']:10s}: {len(boxes)} box(es) -> {box_str}")

    # ============================================================
    # 阶段 6: 绘图 — 2 行 x 8 列矩阵
    # ============================================================
    print("\n[STAGE 6] Rendering 2x8 comparison matrix...")
    n_cols = N_TOTAL
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.6, 6.0),
                             facecolor='white')

    cat_to_color = {
        'dense': HIGHLIGHT_COLORS['dense'],
        'crack': HIGHLIGHT_COLORS['crack'],
        'dissolved': HIGHLIGHT_COLORS['dissolved'],
    }

    for col_idx, s in enumerate(selected):
        # ---- 上行：真实岩心切片 ----
        ax_real = axes[0, col_idx]
        real_2d = s['img'][:, :, 0]
        ax_real.imshow(real_2d, cmap='gray', vmin=0.0, vmax=1.0,
                       interpolation='bilinear')
        ax_real.set_xticks([])
        ax_real.set_yticks([])

        color = cat_to_color.get(s['category'], '#FF6600')
        for box in s['boxes']:
            draw_highlight_box(ax_real, *box, color=color, lw=2.2)

        # 边框：顶部色条指示类别
        for spine in ax_real.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('#cccccc')
        ax_real.spines['top'].set_linewidth(3.0)
        ax_real.spines['top'].set_color(color)

        # ---- 下行：特征表达输出 ----
        ax_recon = axes[1, col_idx]
        recon_2d = s['recon']
        ax_recon.imshow(recon_2d, cmap='gray', vmin=0.0, vmax=1.0,
                        interpolation='bilinear')
        ax_recon.set_xticks([])
        ax_recon.set_yticks([])

        for box in s['boxes']:
            draw_highlight_box(ax_recon, *box, color=color, lw=2.2)

        for spine in ax_recon.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('#cccccc')
        ax_recon.spines['top'].set_linewidth(3.0)
        ax_recon.spines['top'].set_color(color)

    # ---- 左侧垂直居中标签 ----
    fig.text(0.015, 0.755, '真实样本\n(Ground Truth)',
             fontsize=13, fontweight='bold', ha='center', va='center',
             rotation=90, color='#333333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#AAAAAA', linewidth=0.8))
    fig.text(0.015, 0.265, '特征表达\n(Learned Output)',
             fontsize=13, fontweight='bold', ha='center', va='center',
             rotation=90, color='#333333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#AAAAAA', linewidth=0.8))

    # ---- 图例 (底部) ----
    legend_elements = [
        mpatches.Patch(edgecolor=HIGHLIGHT_COLORS['crack'],
                       facecolor='none', linewidth=2.5,
                       label='微裂缝 (Micro-crack)'),
        mpatches.Patch(edgecolor=HIGHLIGHT_COLORS['dissolved'],
                       facecolor='none', linewidth=2.5,
                       label='溶孔边界 (Dissolved pore)'),
        mpatches.Patch(edgecolor=HIGHLIGHT_COLORS['dense'],
                       facecolor='none', linewidth=2.5,
                       label='致密基质 (Dense matrix)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    # ---- 标题 ----
    fig.suptitle('VQ-VAE 测试集特征表达对比矩阵',
                 fontsize=16, fontweight='bold', y=0.995)

    # ---- 保存 ----
    fig.tight_layout(rect=[0.055, 0.04, 1, 0.94], pad=0.8, w_pad=0.3, h_pad=0.5)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    plt.close(fig)
    print(f"\n[DONE] Figure 4-2 saved to: {OUTPUT_PATH}")

    # ---- 统计报告 ----
    avg_mae = np.mean([s['mae'] for s in selected])
    print(f"\n[STATS] Average MAE across 8 test samples: {avg_mae:.5f}")
    for cat in cat_names:
        cat_samples = [s for s in selected if s['category'] == cat]
        if cat_samples:
            cat_mae = np.mean([s['mae'] for s in cat_samples])
            print(f"  [{cat:10s}] n={len(cat_samples)}, avg MAE={cat_mae:.5f}")


if __name__ == "__main__":
    main()
