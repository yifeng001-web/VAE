"""
图 4-1: VQ-VAE 第一阶段训练损失收敛曲线
==========================================
读取 training_history.npz 的 6 组指标（50 epoch），绘制三曲线双Y轴图：
  - 左Y轴: 总损失 (Total Loss) — 黑色实线
  - 右Y轴: MAE (钢蓝色)、孔隙度损失 (砖红色)
  - Grid lines, Epoch 45-50 收敛标注 "收敛于 ~0.005"
  - 白色背景、SimHei 字体、学术风格
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== 中文字体 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# ===== 路径 =====
from thesis_code.config import PROJECT_ROOT
HISTORY_PATH = os.path.join(PROJECT_ROOT, "results", "training_history.npz")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_4_1_Training_Loss.png")

# ===== 学术配色 =====
C_BLACK = '#1a1a1a'       # Total Loss
C_BLUE = '#2C6FAC'        # MAE
C_RED = '#B8403A'         # Porosity (砖红)
C_GRAY = '#888888'


def main():
    # ── 1. 加载数据 ──
    data = np.load(HISTORY_PATH)
    epochs = np.arange(1, len(data['loss']) + 1)

    print(f"[DATA] Loaded {len(epochs)} epochs")
    for k in data.keys():
        print(f"  {k:10s}: {data[k][0]:.6f} → {data[k][-1]:.6f}")

    # ── 2. 创建图像 (双Y轴) ──
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6.5), facecolor='white')
    ax1.set_facecolor('white')

    # 去掉顶/右 spines (ax1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color('#333333')
    ax1.spines['left'].set_linewidth(0.8)
    ax1.spines['bottom'].set_color('#333333')
    ax1.spines['bottom'].set_linewidth(0.8)

    # 右轴
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('#333333')
    ax2.spines['right'].set_linewidth(0.8)

    # 网格线
    ax1.grid(True, linestyle='--', alpha=0.25, color='#aaaaaa', linewidth=0.5)
    ax1.set_axisbelow(True)

    # ── 3. 绘制三条曲线 ──
    # 左Y轴: 总损失 (黑色实线, 粗)
    line1, = ax1.plot(epochs, data['loss'],
                      color=C_BLACK, linewidth=2.5, label='Total Loss (总损失)',
                      zorder=5)

    # 右Y轴: MAE (钢蓝色, 虚线)
    line2, = ax2.plot(epochs, data['mae'],
                      color=C_BLUE, linewidth=2.0, linestyle='-', alpha=0.85,
                      label='MAE (重建损失)', zorder=4)

    # 右Y轴: Porosity (砖红色, 点划线)
    line3, = ax2.plot(epochs, data['porosity'],
                      color=C_RED, linewidth=2.0, linestyle='-', alpha=0.85,
                      label='Porosity Loss (孔隙度损失)', zorder=3)

    # ── 4. 坐标轴标签 ──
    ax1.set_xlabel('Epoch (训练轮次)', fontsize=13, fontweight='bold', color='#222222')
    ax1.set_ylabel('Total Loss (总损失)', fontsize=13, fontweight='bold', color=C_BLACK)
    ax2.set_ylabel('Component Loss (分量损失)', fontsize=13, fontweight='bold', color='#555555')

    # Y轴颜色与曲线对应
    ax1.tick_params(axis='y', labelcolor=C_BLACK, colors='#333333')
    ax2.tick_params(axis='y', labelcolor='#555555', colors='#333333')
    ax1.tick_params(axis='x', labelsize=11, colors='#333333')

    # X轴刻度
    ax1.set_xlim(1, len(epochs))
    ax1.set_xticks([1] + list(range(5, len(epochs) + 1, 5)))

    # ── 5. 合并图例 ──
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='upper right', fontsize=10,
                        frameon=True, facecolor='white', edgecolor='#cccccc',
                        framealpha=0.92, ncol=1)
    legend.set_zorder(10)

    # ── 6. 收敛区间标注 (Epoch 45-50) ──
    # 灰色半透明竖带
    ax1.axvspan(45, 50, alpha=0.08, color=C_GRAY, zorder=1)

    # 标注文字和箭头
    ax1.annotate(
        '收敛于 ~0.005',
        xy=(47.5, data['loss'][46]),
        xytext=(43, data['loss'][46] * 1.8),
        fontsize=12, fontweight='bold', color='#333333',
        ha='center', va='bottom',
        arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.5,
                        connectionstyle='arc3,rad=0.15'),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C_GRAY, alpha=0.92),
        zorder=10
    )

    # ── 7. 标题 ──
    ax1.set_title(
        '图 4-1  VQ-VAE 第一阶段训练损失收敛曲线',
        fontsize=14, fontweight='bold', color='#222222', pad=15
    )

    # ── 8. 末期数值标注 ──
    ax1.annotate(
        f'{data["loss"][-1]:.3f}',
        xy=(epochs[-1], data['loss'][-1]),
        xytext=(epochs[-1] + 1.8, data['loss'][-1] + 0.08),
        fontsize=8, color=C_BLACK, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C_BLACK, lw=0.8),
        ha='left', va='center'
    )

    # MAE 和 Porosity 末值 (在右轴)
    for key, color, y_offset in [('mae', C_BLUE, 0.003), ('porosity', C_RED, 0.0002)]:
        ax2.annotate(
            f'{data[key][-1]:.4f}',
            xy=(epochs[-1], data[key][-1]),
            xytext=(epochs[-1] + 1.8, data[key][-1] + y_offset),
            fontsize=7.5, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=0.6),
            ha='left', va='center'
        )

    # ── 9. 保存 ──
    fig.tight_layout(pad=1.2)
    fig.savefig(OUTPUT_PATH, dpi=350, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\n[DONE] 图 4-1 已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
