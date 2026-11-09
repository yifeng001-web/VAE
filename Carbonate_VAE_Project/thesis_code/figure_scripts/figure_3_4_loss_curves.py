"""
figure_3_4_loss_curves.py —— 图 3-4: Baseline vs Proposed 训练收敛曲线
======================================================================
从真实 training_history.npz 和 baseline_history.npz 读取 50 个 epoch 的训练数据。
绘制双模型总损失 (Total Loss) 对比 + 各自损失分量分解。

重要说明：两个模型的 Total Loss 不可直接对比绝对值（损失项数量不同），
因此也提供 (b) 共享子损失（MAE、MS-SSIM）的对比子图。
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ===== 中文字体 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# ===== 路径 =====
from thesis_code.config import PROJECT_ROOT
PROP_HISTORY = os.path.join(PROJECT_ROOT, "results", "training_history.npz")
BASE_HISTORY = os.path.join(PROJECT_ROOT, "results", "baseline_history.npz")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "loss_curves_comparison.png")

# 学术配色
C_DARK = '#1a1a1a'
C_PROPOSED = '#27AE60'    # 本文方法 — 绿色
C_BASELINE = '#D9583B'    # 基线 — 橙红
C_MAE = '#2C6FAC'         # MAE — 蓝色
C_SSIM = '#8E44AD'        # MS-SSIM — 紫色
C_POROSITY = '#E74C3C'    # Porosity — 红色
C_EDGE = '#F39C12'        # Edge — 橙色
C_VQ = '#7F8C8D'          # VQ — 灰色


def main():
    # ── 1. 加载真实数据 ──
    prop = np.load(PROP_HISTORY)
    base = np.load(BASE_HISTORY)
    epochs_p = np.arange(1, len(prop['loss']) + 1)
    epochs_b = np.arange(1, len(base['loss']) + 1)

    assert len(epochs_p) == 50 and len(epochs_b) == 50, \
        f"Epoch 数量不匹配: prop={len(epochs_p)}, base={len(epochs_b)}"

    print(f"[DATA] Proposed  总损失: {prop['loss'][0]:.4f} -> {prop['loss'][-1]:.4f}")
    print(f"[DATA] Baseline  总损失: {base['loss'][0]:.4f} -> {base['loss'][-1]:.4f}")
    print(f"[DATA] 共享子损失 (Epoch 50):")
    print(f"       MAE    — Baseline={base['mae'][-1]:.4f}  Proposed={prop['mae'][-1]:.4f}")
    print(f"       MS-SSIM — Baseline={base['ms_ssim'][-1]:.4f}  Proposed={prop['ms_ssim'][-1]:.4f}")

    # ── 2. 创建 1x3 布局 ──
    fig = plt.figure(figsize=(18, 6.2), facecolor='white')

    gs = fig.add_gridspec(1, 3, wspace=0.32,
                          left=0.06, right=0.98, top=0.88, bottom=0.14)

    # ═══════════════════════════════════════════
    # (a) 双模型 Total Loss 对比
    # ═══════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('white')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.spines['left'].set_color('#555555')
    ax_a.spines['bottom'].set_color('#555555')
    ax_a.grid(True, linestyle='--', alpha=0.30, color='#aaaaaa', linewidth=0.6)
    ax_a.set_axisbelow(True)

    ax_a.plot(epochs_b, base['loss'], '-', color=C_BASELINE, linewidth=2.5,
              label='基础模型 (Baseline)', zorder=5)
    ax_a.plot(epochs_p, prop['loss'], '-', color=C_PROPOSED, linewidth=2.5,
              label='复合约束模型 (Proposed)', zorder=4)

    # 标注终点值
    ax_a.annotate(f'{base["loss"][-1]:.3f}',
                  xy=(epochs_b[-1], base['loss'][-1]),
                  xytext=(epochs_b[-1] + 2.5, base['loss'][-1] + 0.15),
                  fontsize=9, color=C_BASELINE, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_BASELINE, lw=1.2))
    ax_a.annotate(f'{prop["loss"][-1]:.3f}',
                  xy=(epochs_p[-1], prop['loss'][-1]),
                  xytext=(epochs_p[-1] + 2.5, prop['loss'][-1] - 0.25),
                  fontsize=9, color=C_PROPOSED, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_PROPOSED, lw=1.2))

    ax_a.set_xlabel('训练轮数 (Epoch)', fontsize=12, fontweight='bold', color=C_DARK)
    ax_a.set_ylabel('总损失 (Total Loss)', fontsize=12, fontweight='bold', color=C_DARK)
    ax_a.set_title('(a) 总损失收敛曲线', fontsize=13, fontweight='bold', color=C_DARK, pad=10)
    ax_a.set_xlim(1, 50)
    ax_a.set_xticks([1, 10, 20, 30, 40, 50])
    ax_a.legend(loc='upper right', fontsize=9.5, frameon=True,
                facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax_a.yaxis.set_major_locator(MaxNLocator(6))

    # 关键注释：不可直接对比
    ax_a.text(0.5, -0.22,
              '注意: 两模型损失函数构成不同，\n总值不可直接对比，详见 (b)(c)。',
              transform=ax_a.transAxes, ha='center', fontsize=8.5,
              color='#888888', fontstyle='italic')

    # ═══════════════════════════════════════════
    # (b) 共享子损失对比: MAE
    # ═══════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('white')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.spines['left'].set_color('#555555')
    ax_b.spines['bottom'].set_color('#555555')
    ax_b.grid(True, linestyle='--', alpha=0.30, color='#aaaaaa', linewidth=0.6)
    ax_b.set_axisbelow(True)

    ax_b.plot(epochs_b, base['mae'], '-', color=C_BASELINE, linewidth=2.5,
              label='基础模型 MAE', zorder=5)
    ax_b.plot(epochs_p, prop['mae'], '-', color=C_PROPOSED, linewidth=2.5,
              label='复合约束模型 MAE', zorder=4)

    ax_b.annotate(f'{base["mae"][-1]:.4f}',
                  xy=(epochs_b[-1], base['mae'][-1]),
                  xytext=(epochs_b[-1] + 3, base['mae'][-1] - 0.003),
                  fontsize=8.5, color=C_BASELINE, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_BASELINE, lw=1.0))
    ax_b.annotate(f'{prop["mae"][-1]:.4f}',
                  xy=(epochs_p[-1], prop['mae'][-1]),
                  xytext=(epochs_p[-1] + 3, prop['mae'][-1] + 0.005),
                  fontsize=8.5, color=C_PROPOSED, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_PROPOSED, lw=1.0))

    ax_b.set_xlabel('训练轮数 (Epoch)', fontsize=12, fontweight='bold', color=C_DARK)
    ax_b.set_ylabel('MAE 损失', fontsize=12, fontweight='bold', color=C_DARK)
    ax_b.set_title('(b) MAE 重建损失对比 (可对比)', fontsize=13, fontweight='bold',
                   color=C_DARK, pad=10)
    ax_b.set_xlim(1, 50)
    ax_b.set_xticks([1, 10, 20, 30, 40, 50])
    ax_b.legend(loc='upper right', fontsize=9.5, frameon=True,
                facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax_b.yaxis.set_major_locator(MaxNLocator(6))

    # ═══════════════════════════════════════════
    # (c) 共享子损失对比: MS-SSIM
    # ═══════════════════════════════════════════
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor('white')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.spines['left'].set_color('#555555')
    ax_c.spines['bottom'].set_color('#555555')
    ax_c.grid(True, linestyle='--', alpha=0.30, color='#aaaaaa', linewidth=0.6)
    ax_c.set_axisbelow(True)

    ax_c.plot(epochs_b, base['ms_ssim'], '-', color=C_BASELINE, linewidth=2.5,
              label='基础模型 MS-SSIM', zorder=5)
    ax_c.plot(epochs_p, prop['ms_ssim'], '-', color=C_PROPOSED, linewidth=2.5,
              label='复合约束模型 MS-SSIM', zorder=4)

    ax_c.annotate(f'{base["ms_ssim"][-1]:.4f}',
                  xy=(epochs_b[-1], base['ms_ssim'][-1]),
                  xytext=(epochs_b[-1] + 3, base['ms_ssim'][-1] + 0.005),
                  fontsize=8.5, color=C_BASELINE, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_BASELINE, lw=1.0))
    ax_c.annotate(f'{prop["ms_ssim"][-1]:.4f}',
                  xy=(epochs_p[-1], prop['ms_ssim'][-1]),
                  xytext=(epochs_p[-1] + 3, prop['ms_ssim'][-1] - 0.008),
                  fontsize=8.5, color=C_PROPOSED, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=C_PROPOSED, lw=1.0))

    ax_c.set_xlabel('训练轮数 (Epoch)', fontsize=12, fontweight='bold', color=C_DARK)
    ax_c.set_ylabel('MS-SSIM 损失', fontsize=12, fontweight='bold', color=C_DARK)
    ax_c.set_title('(c) MS-SSIM 损失对比 (可对比)', fontsize=13, fontweight='bold',
                   color=C_DARK, pad=10)
    ax_c.set_xlim(1, 50)
    ax_c.set_xticks([1, 10, 20, 30, 40, 50])
    ax_c.legend(loc='upper right', fontsize=9.5, frameon=True,
                facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax_c.yaxis.set_major_locator(MaxNLocator(6))

    # ── 全局标题 ──
    fig.suptitle(
        '训练收敛曲线: 基础模型 (仅视觉损失) vs. 复合约束模型 (本文方法)\n'
        'Training Convergence: Baseline (Visual-Only) vs. Proposed (Physics-Constrained)',
        fontsize=13.5, fontweight='bold', y=0.998, color=C_DARK)

    # ── 保存 ──
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"\n[DONE] Loss 收敛曲线对比图已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
