"""
plot_training_curves.py — 符合中文核心期刊学术标准的 VQ-VAE 训练收敛曲线图
===============================================================================
读取 training_history.npz 的 6 组真实训练指标数据（50 个 epoch），
绘制一张干净、极简的 Loss 收敛曲线图。
特点：白底，去顶/右 spines，浅灰虚线网格，中英双语标注，>300 DPI。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== 中文字体支持 (防乱码) =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# ===== 路径 =====
from thesis_code.config import PROJECT_ROOT
HISTORY_PATH = os.path.join(PROJECT_ROOT, "results", "training_history.npz")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "training_convergence_curve.png")


def main():
    # ── 1. 加载数据 ──
    data = np.load(HISTORY_PATH)
    epochs = np.arange(1, len(data['loss']) + 1)

    print(f"[DATA] Loaded {len(epochs)} epochs from: {HISTORY_PATH}")
    for k in data.keys():
        print(f"  {k:10s}: {data[k][0]:.4f} → {data[k][-1]:.4f}")

    # ── 2. 创建图像 ──
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), facecolor='white')
    ax.set_facecolor('white')

    # 去掉顶部和右侧 spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['bottom'].set_linewidth(0.8)

    # 浅灰虚线网格
    ax.grid(True, linestyle='--', alpha=0.30, color='#aaaaaa', linewidth=0.6)
    ax.set_axisbelow(True)  # 网格线在数据下方

    # ── 3. 绘制曲线 ──
    # 主曲线：总损失 — 深蓝色粗线
    ax.plot(epochs, data['loss'],
            color='#1f77b4', linewidth=2.5, label='总损失 (Total Loss)',
            zorder=4)

    # 子损失曲线 — 细线
    sub_curves = [
        ('vq_loss',   '#E67E22', 1.4, 'VQ 损失 (VQ Loss)'),
        ('mae',       '#2ECC71', 1.2, 'MAE 重建损失 (MAE Loss)'),
        ('ms_ssim',   '#3498DB', 1.2, 'MS-SSIM 损失 (MS-SSIM Loss)'),
        ('porosity',  '#8E44AD', 1.2, '孔隙度损失 (Porosity Loss)'),
        ('edge',      '#E74C3C', 1.2, 'Sobel 边缘损失 (Edge Loss)'),
    ]

    for key, color, lw, label in sub_curves:
        if key in data and len(data[key]) > 0:
            ax.plot(epochs, data[key],
                    color=color, linewidth=lw, label=label,
                    alpha=0.75, zorder=3)

    # ── 4. 坐标轴标签 ──
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontweight='bold', color='#222222')
    ax.set_ylabel('损失值 (Loss)',  fontsize=14, fontweight='bold', color='#222222')

    # X 轴刻度
    ax.set_xlim(1, len(epochs))
    ax.set_xticks([1] + list(range(5, len(epochs) + 1, 5)))
    ax.tick_params(axis='x', labelsize=11, colors='#333333')
    ax.tick_params(axis='y', labelsize=11, colors='#333333')

    # ── 5. 图例 (右上角) ──
    legend = ax.legend(
        loc='upper right',
        fontsize=10,
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc',
        framealpha=0.9,
        ncol=1,
    )

    # ── 6. 标注起始值和终止值 ──
    # 在总损失曲线末端标注
    ax.annotate(
        f'{data["loss"][-1]:.3f}',
        xy=(epochs[-1], data['loss'][-1]),
        xytext=(epochs[-1] + 1.2, data['loss'][-1] + 0.05),
        fontsize=9, color='#1f77b4', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
        ha='left', va='center',
    )

    # ── 7. 保存 ──
    plt.tight_layout(pad=1.2)
    fig.savefig(OUTPUT_PATH, dpi=350, bbox_inches='tight', facecolor='white',
                edgecolor='none')
    plt.close(fig)
    print(f"\n[DONE] Convergence curve saved to: {OUTPUT_PATH}")
    print(f"[INFO] Resolution: 350 DPI, Size: {fig.get_size_inches()[0]:.1f}x{fig.get_size_inches()[1]:.1f} inches")


if __name__ == "__main__":
    main()
