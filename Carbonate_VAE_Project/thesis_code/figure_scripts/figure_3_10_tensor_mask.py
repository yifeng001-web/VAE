"""
图 3-10: 连续特征张量提取与掩膜矩阵(M)可视化
================================================================
严谨工科风格重绘：去除所有多余修饰，教科书级几何示意图。
左侧: 3D 线框长方体 (学术灰蓝单色线条) + 数学符号标注
中间: 平实箭头 + "特征加权混合" + 融合算式
右侧: 高斯软掩膜矩阵 M 灰度热力图 + 标准 Colorbar (0.0~1.0)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# 强制指定使用 Windows 系统自带的黑体 (SimHei) 来显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴上的负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Paths ──
from thesis_code.config import PROJECT_ROOT
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_3_10_Tensor_Mask.png")

# ── 全局配色与字号 ──
WIRE_COLOR = '#5B7A9A'        # 学术灰蓝（克制单色）
TEXT_COLOR = '#222222'         # 正文黑
LABEL_COLOR = '#444444'        # 标注灰
ARROW_COLOR = '#666666'        # 箭头灰
FONT_SIZE_BASE = 10.5          # ≈ 五号字
FONT_SIZE_LABEL = 11
FONT_SIZE_TITLE = 12

fig = plt.figure(figsize=(16, 6), facecolor='white')

# ================================================================
# LEFT: 3D 张量线框长方体 — 教科书几何示意图风格
# ================================================================
ax_left = fig.add_subplot(1, 3, 1, projection='3d', facecolor='white')

# 符号维度
H, W, D = 4, 4, 8   # H'=32, W'=32, D=64 的示意缩放

# ── 绘制所有网格线 (克制的单色线框，无填充) ──
lw_grid = 0.5
alpha_grid = 0.45

# Z 方向竖线
for xi in range(H + 1):
    for yi in range(W + 1):
        ax_left.plot([xi, xi], [yi, yi], [0, D],
                     color=WIRE_COLOR, linewidth=lw_grid, alpha=alpha_grid)

# Y 方向横线 (平行于 W 轴)
for xi in range(H + 1):
    for zi in range(D + 1):
        ax_left.plot([xi, xi], [0, W], [zi, zi],
                     color=WIRE_COLOR, linewidth=lw_grid, alpha=alpha_grid)

# X 方向横线 (平行于 H 轴)
for yi in range(W + 1):
    for zi in range(D + 1):
        ax_left.plot([0, H], [yi, yi], [zi, zi],
                     color=WIRE_COLOR, linewidth=lw_grid, alpha=alpha_grid)

# ── 外框加粗 (可见轮廓线) ──
lw_outer = 1.2
alpha_outer = 0.75
# 底面四条边
for (x1, y1, x2, y2) in [(0, 0, H, 0), (H, 0, H, W),
                           (H, W, 0, W), (0, W, 0, 0)]:
    ax_left.plot([x1, x2], [y1, y2], [0, 0],
                 color=WIRE_COLOR, linewidth=lw_outer, alpha=alpha_outer)
# 顶面四条边
for (x1, y1, x2, y2) in [(0, 0, H, 0), (H, 0, H, W),
                           (H, W, 0, W), (0, W, 0, 0)]:
    ax_left.plot([x1, x2], [y1, y2], [D, D],
                 color=WIRE_COLOR, linewidth=lw_outer, alpha=alpha_outer)
# 四条竖棱
for (x, y) in [(0, 0), (H, 0), (H, W), (0, W)]:
    ax_left.plot([x, x], [y, y], [0, D],
                 color=WIRE_COLOR, linewidth=lw_outer, alpha=alpha_outer)

# ── 轴标注 ──
ax_left.set_xlabel("H' = 32", fontsize=FONT_SIZE_LABEL,
                   fontweight='medium', color=LABEL_COLOR, labelpad=6)
ax_left.set_ylabel("W' = 32", fontsize=FONT_SIZE_LABEL,
                   fontweight='medium', color=LABEL_COLOR, labelpad=6)
ax_left.set_zlabel("D = 64", fontsize=FONT_SIZE_LABEL,
                   fontweight='medium', color=LABEL_COLOR, labelpad=6)

ax_left.set_xticks([0, H])
ax_left.set_yticks([0, W])
ax_left.set_zticks([0, D])
ax_left.set_xlim(0, H)
ax_left.set_ylim(0, W)
ax_left.set_zlim(0, D)
ax_left.tick_params(labelsize=9, colors=LABEL_COLOR)

ax_left.view_init(elev=22, azim=-55)
ax_left.set_box_aspect([H, W, D])
ax_left.set_title('', fontsize=1)

# ── 数学符号标注: Z_e ∈ R^(32×32×64) ──
ax_left.text2D(0.50, 0.92,
               r'$\mathbf{Z}_e \in \mathbb{R}^{32 \times 32 \times 64}$',
               transform=ax_left.transAxes,
               fontsize=13, fontweight='bold', ha='center',
               color=TEXT_COLOR)
ax_left.text2D(0.50, 0.85,
               r'(Encoder 输出的连续潜特征张量)',
               transform=ax_left.transAxes,
               fontsize=9.5, ha='center', color='#777777')

# ================================================================
# MIDDLE: 融合箭头与公式
# ================================================================
ax_mid = fig.add_subplot(1, 3, 2, facecolor='white')
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.axis('off')

# 平实箭头
ax_mid.annotate('', xy=(0.95, 0.55), xytext=(0.05, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2.0,
                                color=ARROW_COLOR))

# 箭头上方: "特征加权混合"
ax_mid.text(0.5, 0.78, '特征加权混合',
            fontsize=FONT_SIZE_LABEL, fontweight='medium',
            ha='center', va='center', color=TEXT_COLOR)

# 箭头下方: 融合公式
ax_mid.text(0.5, 0.22,
            r'$\mathbf{Z}_{\mathrm{fused}} = \mathbf{M} \odot '
            r'\mathbf{Z}_{eB} + (1-\mathbf{M}) \odot \mathbf{Z}_{eA}$',
            fontsize=10.5, ha='center', va='center',
            color=TEXT_COLOR)

# ================================================================
# RIGHT: 高斯软掩膜矩阵 M 灰度热力图
# ================================================================
ax_right = fig.add_subplot(1, 3, 3, facecolor='white')

# 生成 2D 高斯软掩膜 (与代码中实际使用的公式一致)
size = 200
xs = np.linspace(-2.8, 2.8, size)
ys = np.linspace(-2.8, 2.8, size)
X, Y = np.meshgrid(xs, ys)
sigma = 1.2
M = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

im = ax_right.imshow(M, cmap='gray', vmin=0.0, vmax=1.0,
                     extent=[0, size, 0, size], origin='lower',
                     interpolation='bilinear')

# ── 标准 Colorbar (无多余标注) ──
cbar = plt.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04,
                    ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_label('权重', fontsize=FONT_SIZE_LABEL,
               fontweight='medium', color=TEXT_COLOR, labelpad=6)
cbar.ax.tick_params(labelsize=9, colors=TEXT_COLOR)

# ── 热力图标题 ──
ax_right.set_title('软掩膜矩阵 M',
                   fontsize=FONT_SIZE_TITLE, fontweight='bold',
                   pad=10, color=TEXT_COLOR)

# 去除坐标轴刻度
ax_right.set_xticks([])
ax_right.set_yticks([])

# 细边框
for spine in ax_right.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(0.6)

# ================================================================
# 整体标题与保存
# ================================================================
fig.suptitle('图 3-10  连续特征张量提取与掩膜矩阵(M)可视化',
             fontsize=14, fontweight='bold', y=1.01, color=TEXT_COLOR)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_PATH, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"[DONE] 图 3-10 已重绘并保存至:\n  {OUTPUT_PATH}")
