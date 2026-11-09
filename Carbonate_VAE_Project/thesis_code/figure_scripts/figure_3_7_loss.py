"""
figure_3_7_loss.py —— 图 3-7: 训练 Loss 收敛曲线 (Baseline vs Proposed)
=====================================================================
仅读取已有的 training_history.npz 和 baseline_history.npz，不触发训练。
X 轴：训练轮数 (Epoch)，Y 轴：总损失 (Total Loss)
基础模型 = 紫色，复合约束模型 = 绿色
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== 必须字体配置 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 路径 =====
from thesis_code.config import PROJECT_ROOT
PROP_HISTORY = os.path.join(PROJECT_ROOT, "results", "training_history.npz")
BASE_HISTORY = os.path.join(PROJECT_ROOT, "results", "baseline_history.npz")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_3_7_Loss_Curves.png")

# ===== 配色 =====
COLOR_BASELINE = '#8E44AD'   # 紫色 — 基础模型
COLOR_PROPOSED = '#27AE60'   # 绿色 — 复合约束模型


def main():
    # ── 加载数据 ──
    prop = np.load(PROP_HISTORY)
    base = np.load(BASE_HISTORY)
    epochs_p = np.arange(1, len(prop['loss']) + 1)
    epochs_b = np.arange(1, len(base['loss']) + 1)

    print(f"[DATA] Proposed Total Loss: {prop['loss'][0]:.4f} -> {prop['loss'][-1]:.4f}")
    print(f"[DATA] Baseline Total Loss: {base['loss'][0]:.4f} -> {base['loss'][-1]:.4f}")

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.set_facecolor('white')

    ax.plot(epochs_b, base['loss'], '-', color=COLOR_BASELINE, linewidth=2.2,
            label='基础模型 (Baseline)', zorder=5)
    ax.plot(epochs_p, prop['loss'], '-', color=COLOR_PROPOSED, linewidth=2.2,
            label='复合约束模型 (Proposed)', zorder=4)

    # 标注终点数值
    ax.annotate(f'{base["loss"][-1]:.4f}',
                xy=(epochs_b[-1], base['loss'][-1]),
                xytext=(epochs_b[-1] + 3, base['loss'][-1] + 0.10),
                fontsize=10, color=COLOR_BASELINE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_BASELINE, lw=1.2))
    ax.annotate(f'{prop["loss"][-1]:.4f}',
                xy=(epochs_p[-1], prop['loss'][-1]),
                xytext=(epochs_p[-1] + 3, prop['loss'][-1] - 0.18),
                fontsize=10, color=COLOR_PROPOSED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_PROPOSED, lw=1.2))

    ax.set_xlabel('训练轮数 (Epoch)', fontsize=14, fontweight='bold')
    ax.set_ylabel('总损失 (Total Loss)', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 50)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.tick_params(axis='both', labelsize=12)

    # 留缓冲空间
    all_loss = np.concatenate([base['loss'], prop['loss']])
    y_min, y_max = all_loss.min(), all_loss.max()
    y_pad = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - y_pad, y_max + y_pad * 1.5)

    ax.grid(True, linestyle='--', alpha=0.30, color='#aaaaaa', linewidth=0.6)
    ax.set_axisbelow(True)

    # 隐藏顶/右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')

    # 图例
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                       facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    for text, color in zip(legend.get_texts(), [COLOR_BASELINE, COLOR_PROPOSED]):
        text.set_color(color)
        text.set_fontweight('bold')

    # 标题
    ax.set_title('训练收敛曲线：基础模型 vs. 复合约束模型',
                 fontsize=14, fontweight='bold', pad=12)

    fig.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"[DONE] 图 3-7 已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
