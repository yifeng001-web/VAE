"""
figure_3_9_pixelcnn_loss.py —— 图 3-9: PixelCNN 训练 Loss 曲线
==============================================================
读取 PixelCNN 训练历史 (pixelcnn_history.npz)，绘制 25 个 Epoch 的
Train/Val 交叉熵损失收敛曲线。
X 轴：训练轮数 (Epoch)，Y 轴：交叉熵损失 (Cross-Entropy Loss)
曲线平滑清晰，证明模型正常收敛。
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
HISTORY_PATH = os.path.join(PROJECT_ROOT, "results", "checkpoints", "pixelcnn_history.npz")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_3_9_PixelCNN_Loss.png")

# ===== 配色 =====
COLOR_TRAIN = '#1A73E8'    # 蓝色 — 训练损失
COLOR_VAL   = '#E8710A'    # 橙色 — 验证损失


def main():
    # ── 加载数据 ──
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"[FATAL] PixelCNN history not found: {HISTORY_PATH}")
    data = np.load(HISTORY_PATH)
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    epochs = np.arange(1, len(train_loss) + 1)

    print(f"[DATA] PixelCNN Train Loss: {train_loss[0]:.4f} → {train_loss[-1]:.4f}")
    print(f"[DATA] PixelCNN Val Loss:   {val_loss[0]:.4f} → {val_loss[-1]:.4f}")
    best_epoch = int(np.argmin(val_loss)) + 1
    best_val = float(np.min(val_loss))
    print(f"[DATA] Best Val Loss: {best_val:.4f} @ Epoch {best_epoch}")

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='white')
    ax.set_facecolor('white')

    # 主曲线
    ax.plot(epochs, train_loss, 'o-', color=COLOR_TRAIN, linewidth=2.0,
            markersize=5, markerfacecolor='white', markeredgewidth=1.5,
            label='训练损失 (Train Loss)', zorder=5)
    ax.plot(epochs, val_loss, 's-', color=COLOR_VAL, linewidth=2.0,
            markersize=5, markerfacecolor='white', markeredgewidth=1.5,
            label='验证损失 (Val Loss)', zorder=4)

    # 标注最佳验证损失
    ax.annotate(f'最优: {best_val:.4f}',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 1.5, best_val + 0.04),
                fontsize=10.5, color=COLOR_VAL, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLOR_VAL, lw=1.3))

    # 标注起点和终点
    ax.annotate(f'{train_loss[0]:.4f}',
                xy=(1, train_loss[0]),
                xytext=(0.2, train_loss[0] + 0.08),
                fontsize=9, color=COLOR_TRAIN, fontweight='bold')
    ax.annotate(f'{train_loss[-1]:.4f}',
                xy=(25, train_loss[-1]),
                xytext=(22.5, train_loss[-1] - 0.10),
                fontsize=9, color=COLOR_TRAIN, fontweight='bold')
    ax.annotate(f'{val_loss[-1]:.4f}',
                xy=(25, val_loss[-1]),
                xytext=(22.5, val_loss[-1] + 0.12),
                fontsize=9, color=COLOR_VAL, fontweight='bold')

    # 轴标签
    ax.set_xlabel('训练轮数 (Epoch)', fontsize=14, fontweight='bold')
    ax.set_ylabel('交叉熵损失 (Cross-Entropy Loss)', fontsize=14, fontweight='bold')

    # 轴范围
    ax.set_xlim(0.5, 25.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.tick_params(axis='both', labelsize=12)

    # Y 轴缓冲
    all_loss = np.concatenate([train_loss, val_loss])
    y_min, y_max = all_loss.min(), all_loss.max()
    y_pad = (y_max - y_min) * 0.20
    ax.set_ylim(y_min - y_pad * 0.3, y_max + y_pad * 1.2)

    # 网格
    ax.grid(True, linestyle='--', alpha=0.30, color='#aaaaaa', linewidth=0.6)
    ax.set_axisbelow(True)

    # 边框处理
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')

    # 图例
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                       facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    for text, color in zip(legend.get_texts(), [COLOR_TRAIN, COLOR_VAL]):
        text.set_color(color)
        text.set_fontweight('bold')

    # 标题
    ax.set_title('PixelCNN 先验训练收敛曲线',
                 fontsize=15, fontweight='bold', pad=14)

    fig.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"[DONE] 图 3-9 已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
