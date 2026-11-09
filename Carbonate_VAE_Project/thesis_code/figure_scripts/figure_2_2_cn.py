"""
figure_2_2_cn.py — 图 2-2: 自适应百分位对比度拉伸（最终版）
============================================================================
全程 Python/matplotlib，直方图使用 plt.bar() 离散柱状图（MATLAB 风格），
字体、字号、科学计数法完全可控。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tifffile
import os

# --- 中文字体配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ===== 路径配置 =====
from thesis_code.config import PROJECT_ROOT

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
image_path = os.path.join(DATA_DIR, "1633-1663-1203-50um0000.tif")

if not os.path.exists(image_path):
    image_path = os.path.join(PROJECT_ROOT, "data", "raw", "CT", "8-gs101-2",
                              "volume tif-1813-1854-1469-45um",
                              "1813-1854-1469-45um0203.tif")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Figure_2_2_MinMaxScaling_CN.png")

# ===== 共用参数 =====
N_BINS = 80
BIN_RANGE = (0.0, 1.0)
XLIM = (-0.02, 1.02)
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 11
BAR_EDGE_WIDTH = 0.5
BAR_ALPHA = 0.85

HIST_COLOR_C = '#5c6bc0'       # 蓝紫色
HIST_EDGE_C = '#2a3a6e'        # 深蓝边框
HIST_COLOR_D = '#2e7d32'       # 绿色
HIST_EDGE_D = '#1a4a1e'        # 深绿边框


def adaptive_contrast_stretching(image_array, lower_percent=2.0, upper_percent=98.0):
    """自适应百分位截断对比度拉伸。"""
    img_float = image_array.astype(np.float32)
    p_low = np.percentile(img_float, lower_percent)
    p_high = np.percentile(img_float, upper_percent)
    img_clipped = np.clip(img_float, p_low, p_high)
    epsilon = 1e-8
    stretched_float = (img_clipped - p_low) / (p_high - p_low + epsilon)
    return img_float, stretched_float, p_low, p_high


def draw_matlab_style_hist(ax, data, bins, range_tuple, color, edgecolor,
                           title_text, xlabel_text, ylabel_text):
    """
    MATLAB 风格离散柱状直方图。
    使用 np.histogram + plt.bar，每个 bin 独立柱子 + 细边框。
    """
    counts, bin_edges = np.histogram(data.ravel(), bins=bins,
                                     range=range_tuple, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = bin_edges[1] - bin_edges[0]

    ax.bar(bin_centers, counts, width=bar_width,
           color=color, edgecolor=edgecolor,
           linewidth=BAR_EDGE_WIDTH, alpha=BAR_ALPHA)

    ax.set_title(title_text, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel_text, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel_text, fontsize=LABEL_FONTSIZE)

    # Y 轴科学计数法
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # 调整科学计数法偏移文本的字体
    ax.yaxis.get_offset_text().set_fontsize(TICK_FONTSIZE)

    ax.set_xlim(XLIM)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    # MATLAB 风格网格：只有水平虚线
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # 去掉顶部和右侧边框（MATLAB 默认风格）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    return bin_centers, counts, bin_edges


def main():
    # 1. 读取原始 16-bit 图像
    if not os.path.exists(image_path):
        print(f"[ERROR] 图片路径不存在: {image_path}")
        return

    img_raw = tifffile.imread(image_path)
    print(f"[INFO] 图片读取成功: {image_path}")
    print(f"       dtype={img_raw.dtype}, shape={img_raw.shape}, "
          f"min={img_raw.min()}, max={img_raw.max()}, mean={img_raw.mean():.1f}")

    # 2. 百分位自适应拉伸
    img_float, img_stretched_float, p_low, p_high = \
        adaptive_contrast_stretching(img_raw, lower_percent=2.0, upper_percent=98.0)

    # 原始图像 min-max 归一化到 [0,1]，统一横坐标
    img_min = img_float.min()
    img_max = img_float.max()
    img_normalized = (img_float - img_min) / (img_max - img_min + 1e-8)

    p_low_norm = (p_low - img_min) / (img_max - img_min + 1e-8)
    p_high_norm = (p_high - img_min) / (img_max - img_min + 1e-8)

    print(f"[INFO] 百分位截断: P2={p_low:.1f} (归一化={p_low_norm:.4f}), "
          f"P98={p_high:.1f} (归一化={p_high_norm:.4f})")
    print(f"[INFO] 拉伸完成: float范围 [{img_stretched_float.min():.6f}, "
          f"{img_stretched_float.max():.6f}]")

    # 3. 绘制 2×2 学术对比图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    # (a) 左上：原始切片
    global_max = np.max(img_float)
    axs[0, 0].imshow(img_raw, cmap='gray', vmin=0, vmax=global_max)
    axs[0, 0].set_title('(a) 原始切片图像', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=8)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    for spine in axs[0, 0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    # (b) 右上：拉伸后切片
    axs[0, 1].imshow(img_stretched_float, cmap='gray', vmin=0.0, vmax=1.0)
    axs[0, 1].set_title('(b) 自适应对比度拉伸后切片', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=8)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    for spine in axs[0, 1].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    # (c) 左下：原始图像直方图（MATLAB 风格，统一归一化横坐标）
    counts_c, bin_edges_c = np.histogram(img_normalized.ravel(), bins=N_BINS,
                                         range=BIN_RANGE, density=True)
    bin_centers_c = (bin_edges_c[:-1] + bin_edges_c[1:]) / 2
    bar_width = bin_edges_c[1] - bin_edges_c[0]

    axs[1, 0].bar(bin_centers_c, counts_c, width=bar_width,
                  color=HIST_COLOR_C, edgecolor=HIST_EDGE_C,
                  linewidth=BAR_EDGE_WIDTH, alpha=BAR_ALPHA)

    # P2/P98 截断线
    y_max_c = np.max(counts_c) * 1.1
    axs[1, 0].axvline(x=p_low_norm, color='red', linestyle='--', linewidth=1.5)
    axs[1, 0].axvline(x=p_high_norm, color='red', linestyle='--', linewidth=1.5)
    axs[1, 0].text(p_low_norm + 0.02, y_max_c * 0.90,
                   f'P2={p_low_norm:.3f}', color='red',
                   fontsize=TICK_FONTSIZE, fontweight='bold')
    axs[1, 0].text(p_high_norm + 0.02, y_max_c * 0.90,
                   f'P98={p_high_norm:.3f}', color='red',
                   fontsize=TICK_FONTSIZE, fontweight='bold')

    axs[1, 0].set_title('(c) 原始图像直方图', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=6)
    axs[1, 0].set_xlabel('归一化像素值', fontsize=LABEL_FONTSIZE)
    axs[1, 0].set_ylabel('概率密度', fontsize=LABEL_FONTSIZE)
    axs[1, 0].set_xlim(XLIM)
    axs[1, 0].set_ylim(0, y_max_c * 1.05)
    axs[1, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axs[1, 0].yaxis.get_offset_text().set_fontsize(TICK_FONTSIZE)
    axs[1, 0].tick_params(labelsize=TICK_FONTSIZE)
    axs[1, 0].grid(True, linestyle='--', alpha=0.3, axis='y')
    axs[1, 0].set_axisbelow(True)
    axs[1, 0].spines['top'].set_visible(False)
    axs[1, 0].spines['right'].set_visible(False)

    # (d) 右下：拉伸后图像直方图（MATLAB 风格）
    counts_d, bin_edges_d = np.histogram(img_stretched_float.ravel(), bins=N_BINS,
                                         range=BIN_RANGE, density=True)
    bin_centers_d = (bin_edges_d[:-1] + bin_edges_d[1:]) / 2

    axs[1, 1].bar(bin_centers_d, counts_d, width=bar_width,
                  color=HIST_COLOR_D, edgecolor=HIST_EDGE_D,
                  linewidth=BAR_EDGE_WIDTH, alpha=BAR_ALPHA)

    axs[1, 1].set_title('(d) 自适应对比度拉伸后直方图', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=6)
    axs[1, 1].set_xlabel('归一化像素值', fontsize=LABEL_FONTSIZE)
    axs[1, 1].set_ylabel('概率密度', fontsize=LABEL_FONTSIZE)
    axs[1, 1].set_xlim(XLIM)
    axs[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axs[1, 1].yaxis.get_offset_text().set_fontsize(TICK_FONTSIZE)
    axs[1, 1].tick_params(labelsize=TICK_FONTSIZE)
    axs[1, 1].grid(True, linestyle='--', alpha=0.3, axis='y')
    axs[1, 1].set_axisbelow(True)
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)

    # 收紧版面
    plt.subplots_adjust(hspace=0.30, wspace=0.08)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[DONE] 图表已保存至: {OUTPUT_PATH}")
    print(f"       直方图: MATLAB 风格 plt.bar() 离散柱状, 统一 [0,1] 横坐标")
    print(f"       字体: SimHei 中文, 标题={TITLE_FONTSIZE}pt, 轴标签={LABEL_FONTSIZE}pt")
    print(f"       Y轴: 科学计数法, P2/P98 红色虚线标注")


if __name__ == "__main__":
    main()
