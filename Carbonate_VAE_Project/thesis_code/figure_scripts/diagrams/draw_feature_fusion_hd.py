
from thesis_code.config import DIAGRAM_DIR
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布 - 左侧进一步加宽
fig = plt.figure(figsize=(30, 11), dpi=300)
gs = GridSpec(1, 3, width_ratios=[1.6, 0.8, 1])

# ================= 1. 左侧：3D 特征张量 (等轴测模拟) =================
ax0 = fig.add_subplot(gs[0])
# 坐标范围加大，匹配更大立方体
ax0.set_xlim(-3, 18)
ax0.set_ylim(-3, 18)
ax0.axis('off')

def draw_tensor_3d(ax, ox, oy, w, h, d, color='#4a86e8'):
    """
    等轴测投影绘制 3D 立方体线框 + 内部网格
    ox, oy: 前左下角原点投影坐标
    w, h, d: 宽度(W'), 高度(H'), 深度(D)
    """
    alpha = 0.5  # 等轴测斜率 (dx/dz = dy/dz = alpha)

    def proj(x, y, z):
        return ox + x + z * alpha, oy + y + z * alpha

    # ---- 骨架线框 ----
    corners_3d = [
        (0,0,0), (w,0,0), (w,h,0), (0,h,0),
        (0,0,d), (w,0,d), (w,h,d), (0,h,d),
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),   # 前面
        (4,5), (5,6), (6,7), (7,4),   # 后面
        (0,4), (1,5), (2,6), (3,7),   # 竖棱
    ]
    for i, j in edges:
        x1, y1 = proj(*corners_3d[i])
        x2, y2 = proj(*corners_3d[j])
        ax.plot([x1, x2], [y1, y2], color=color, lw=2.2, alpha=0.75)

    # ---- 内部网格 (深度方向细线) ----
    for i in np.linspace(0, w, 5):
        for j in np.linspace(0, h, 5):
            x1, y1 = proj(i, j, 0)
            x2, y2 = proj(i, j, d)
            ax.plot([x1, x2], [y1, y2], color=color, lw=0.5, alpha=0.3)

    return proj  # 返回投影函数供标注使用


# 立方体：放大到 5×5×10
W = 5.0   # W'
H = 5.0   # H'
D = 10.0  # D
OX = 2.5  # 原点 x 偏移
OY = 2.5  # 原点 y 偏移

proj = draw_tensor_3d(ax0, OX, OY, W, H, D)

# 计算线框实际投影范围
all_px, all_py = [], []
for cx, cy, cz in [(0,0,0), (W,0,0), (W,H,0), (0,H,0),
                    (0,0,D), (W,0,D), (W,H,D), (0,H,D)]:
    px, py = proj(cx, cy, cz)
    all_px.append(px)
    all_py.append(py)
bx_min, bx_max = min(all_px), max(all_px)
by_min, by_max = min(all_py), max(all_py)

# ========== 标注：全部放置在线框投影范围之外 ==========
margin = 3.2  # 安全间距（随字体增大而增加）

# W' 标注 —— 线框正下方
wx, wy = (bx_min + bx_max) / 2, by_min - margin
ax0.text(wx, wy, "$W' = 32$", ha='center', va='top', fontsize=34,
         color='#1f4e79', fontweight='bold')

# H' 标注 —— 线框正左侧
hx, hy = bx_min - margin, (by_min + by_max) / 2
ax0.text(hx, hy, "$H' = 32$", va='center', ha='right', fontsize=34,
         color='#1f4e79', fontweight='bold', rotation=90)

# D 标注 —— 线框正右侧（沿深度方向）
dx, dy = bx_max + margin, (by_min + by_max) / 2
ax0.text(dx, dy, "$D = 64$", va='center', ha='left', fontsize=34,
         color='#1f4e79', fontweight='bold')

# 标题 —— 线框正上方
title_margin = 4.0
tx, ty = (bx_min + bx_max) / 2, by_max + title_margin
ax0.text(tx, ty, "$Z_{i} \\in \\mathbb{R}^{32 \\times 32 \\times 64}$",
         ha='center', va='bottom', fontsize=36, fontweight='bold',
         color='#1f4e79')
ax0.text(tx, ty + 1.8, "(Encoder 抽取的连续特征张量)",
         ha='center', va='bottom', fontsize=26, color='#555555')

# ================= 2. 中间：融合公式与箭头 =================
ax1 = fig.add_subplot(gs[1])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')

ax1.annotate('', xy=(9, 5), xytext=(1, 5),
             arrowprops=dict(facecolor='#333333', edgecolor='none',
                           width=8, headwidth=20))
ax1.text(5, 5.8, "特征加权融合", ha='center', fontsize=28, fontweight='bold')

formula = r"$Z_{fused} = M \odot Z_{eB} + (1 - M) \odot Z_{eA}$"
ax1.text(5, 3.5, formula, ha='center', fontsize=32, color='#c00000',
         bbox=dict(facecolor='#fdf2f2', edgecolor='#f4b084',
                  boxstyle='round,pad=0.5'))

# ================= 3. 右侧：掩膜矩阵可视化 =================
ax2 = fig.add_subplot(gs[2])

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
M = np.exp(-(X**2 + Y**2) / 2.0)

im = ax2.imshow(M, cmap='gray', extent=[-1, 1, -1, 1])
ax2.set_title("软掩膜矩阵 $M$", fontsize=28, fontweight='bold', pad=20)

cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('权重值', fontsize=24)
cbar.ax.tick_params(labelsize=20)

ax2.set_xticks([])
ax2.set_yticks([])

# 整体布局调整
plt.tight_layout()
plt.savefig(str(DIAGRAM_DIR / 'Feature_Fusion_HD.png'), dpi=600, bbox_inches='tight')
print("高清原理示意图已生成！")
