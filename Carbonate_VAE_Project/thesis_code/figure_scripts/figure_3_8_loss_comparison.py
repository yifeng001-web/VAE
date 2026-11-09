"""
图 3-8: PixelCNN vs NanoGPT 训练损失对比折线图
===============================================================
Academic paper-level dual-curve loss comparison chart.
Data sources:
  - PixelCNN (retrained, 18 ResBlocks): results/checkpoints/pixelcnn_retrain_history.npz
  - NanoGPT V2 (Micro-GPT, 4L/4H/256D): results/checkpoints/gpt_prior_v2_history.npz

Style: Nature-inspired seaborn-whitegrid, blue/red palette, 600 DPI.
X-axis: Epoch 0-25. Y-axis: Cross-Entropy Loss (Weighted).
Four curves: PixelCNN Train (blue solid), PixelCNN Val (blue dashed),
             NanoGPT Train (red solid), NanoGPT Val (red dashed).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('seaborn-v0_8-whitegrid')

# 强制指定使用 Windows 系统自带的黑体 (SimHei) 来显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴上的负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ── Paths ──
from thesis_code.config import PROJECT_ROOT
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

PIXELCNN_HISTORY = os.path.join(CHECKPOINT_DIR, "pixelcnn_retrain_history.npz")
GPT_HISTORY = os.path.join(CHECKPOINT_DIR, "gpt_prior_v2_history.npz")
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_3_8_Loss_Comparison.png")

# ── Load data ──
pixelcnn = np.load(PIXELCNN_HISTORY)
gpt = np.load(GPT_HISTORY)

pcnn_train = pixelcnn['train_loss']
pcnn_val   = pixelcnn['val_loss']
gpt_train  = gpt['train']
gpt_val    = gpt['val']

MAX_EPOCH = 25  # User-specified truncation

# Truncate to first MAX_EPOCH epochs (epoch index 0-based)
pcnn_train = pcnn_train[:MAX_EPOCH]
pcnn_val   = pcnn_val[:MAX_EPOCH]
gpt_train  = gpt_train[:MAX_EPOCH]
gpt_val    = gpt_val[:MAX_EPOCH]

epochs_pcnn = np.arange(1, len(pcnn_train) + 1)
epochs_gpt  = np.arange(1, len(gpt_train) + 1)

# ── Plot ──
fig, ax = plt.subplots(figsize=(8, 5.5))

# Nature-inspired color palette
BLUE_SOLID  = '#2166AC'   # deep blue
BLUE_DASHED = '#67A9CF'   # light blue
RED_SOLID   = '#B2182B'   # deep red
RED_DASHED  = '#EF8A62'   # light red

LW = 2.0
MS = 5  # marker size

# PixelCNN curves (blue family)
ax.plot(epochs_pcnn, pcnn_train, color=BLUE_SOLID, linewidth=LW,
        marker='o', markersize=MS, markevery=2, markerfacecolor='white',
        markeredgewidth=1.2, markeredgecolor=BLUE_SOLID,
        label='PixelCNN Train Loss', zorder=5)
ax.plot(epochs_pcnn, pcnn_val, color=BLUE_DASHED, linewidth=LW,
        linestyle='--', marker='s', markersize=MS, markevery=2,
        markerfacecolor='white', markeredgewidth=1.2,
        markeredgecolor=BLUE_DASHED,
        label='PixelCNN Val Loss', zorder=4)

# NanoGPT curves (red family)
ax.plot(epochs_gpt, gpt_train, color=RED_SOLID, linewidth=LW,
        marker='o', markersize=MS, markevery=2, markerfacecolor='white',
        markeredgewidth=1.2, markeredgecolor=RED_SOLID,
        label='NanoGPT Train Loss', zorder=5)
ax.plot(epochs_gpt, gpt_val, color=RED_DASHED, linewidth=LW,
        linestyle='--', marker='s', markersize=MS, markevery=2,
        markerfacecolor='white', markeredgewidth=1.2,
        markeredgecolor=RED_DASHED,
        label='NanoGPT Val Loss', zorder=4)

# ── Axis labels and styling ──
ax.set_xlabel('Epoch', fontsize=13, fontweight='medium', labelpad=8)
ax.set_ylabel('Cross-Entropy Loss (Weighted)', fontsize=13, fontweight='medium', labelpad=8)

# X-axis: integer ticks, 0..25
ax.set_xlim(0, MAX_EPOCH + 0.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_xticks([0, 5, 10, 15, 20, 25])

# Y-axis: include ticks near 1.40
y_min = min(pcnn_train.min(), pcnn_val.min(), gpt_train.min(), gpt_val.min())
y_max = max(pcnn_train.max(), pcnn_val.max(), gpt_train.max(), gpt_val.max())
y_pad = 0.15
ax.set_ylim(y_min - y_pad, y_max + y_pad)
# Ensure 1.40 tick is visible
yticks = ax.get_yticks()
if not any(abs(t - 1.40) < 0.02 for t in yticks):
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

# Tick font sizes (≈ Word 五号字)
ax.tick_params(axis='both', which='major', labelsize=10.5)
ax.tick_params(axis='both', which='minor', labelsize=9)

# ── Legend (upper right) ──
legend = ax.legend(loc='upper right', fontsize=10, frameon=True,
                   framealpha=0.92, edgecolor='#CCCCCC',
                   ncol=1, handlelength=2.5, handletextpad=0.6)
legend.get_frame().set_linewidth(0.8)

# ── Grid styling ──
ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.35, color='#888888')
ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.2, color='#AAAAAA')

# ── Despine (scientific minimalism) ──
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(0.8)
    ax.spines[spine].set_color('#555555')

# ── Title ──
ax.set_title('Training Loss: PixelCNN vs. NanoGPT (VQ-VAE Prior)',
             fontsize=14, fontweight='bold', pad=14,
             color='#333333')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"[DONE] Figure 3-8 saved to:\n  {OUTPUT_PATH}")
print(f"  PixelCNN: {len(pcnn_train)} epochs, train [{pcnn_train[0]:.4f} → {pcnn_train[-1]:.4f}]")
print(f"  PixelCNN: val   [{pcnn_val[0]:.4f} → {pcnn_val[-1]:.4f}]")
print(f"  NanoGPT:  {len(gpt_train)} epochs, train [{gpt_train[0]:.4f} → {gpt_train[-1]:.4f}]")
print(f"  NanoGPT:  val   [{gpt_val[0]:.4f} → {gpt_val[-1]:.4f}]")
