"""
figure_2_1_latent_space.py — 图 2-1: 传统 VAE 与 VQ-VAE 潜在空间对比图
========================================================================
左: 传统 VAE 连续高斯概率分布空间 (continuous latent space)
右: VQ-VAE 离散码本特征空间 (discrete codebook space)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 毕业论文标准设置：支持中文字体，解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

# ================== 左侧：传统 VAE ==================
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)/2)

# 画出平滑的连续高斯云团
ax1.contourf(X, Y, Z, levels=12, cmap='Blues', alpha=0.5)
# 均值中心
ax1.scatter(0, 0, color='red', s=50, zorder=5)
ax1.text(0.1, 0.1, r'$\mu$', fontsize=14, color='red')
# 采样点
ax1.scatter(1.2, 1.2, color='darkblue', s=30, zorder=5)
ax1.annotate(r'$z$', xy=(1.2, 1.2), xytext=(2, 2),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontsize=14)

ax1.text(0, -3.5, "连续概率分布空间", ha='center', fontsize=13)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.axis('off')  # 去除刻度边框，保持学术简洁

# ================== 右侧：VQ-VAE ==================
# 画出离散码本网格点
grid_x, grid_y = np.meshgrid(np.linspace(-2.5, 2.5, 5), np.linspace(-2.5, 2.5, 5))
ax2.scatter(grid_x, grid_y, color='lightgray', s=40, edgecolors='gray', zorder=2)

# 连续特征点 Ze(x) 和最近的码本点 e_k
ze_x, ze_y = 0.8, 1.3
ek_x, ek_y = 1.25, 1.25

ax2.scatter(ze_x, ze_y, color='red', marker='*', s=150, zorder=5)
ax2.scatter(ek_x, ek_y, color='darkblue', s=60, zorder=5)

# 绘制虚线最近邻匹配箭头
ax2.annotate('', xy=(ek_x, ek_y), xytext=(ze_x, ze_y),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, ls='--'))

ax2.text(ze_x-0.8, ze_y-0.2, r'$Z_e(x)$', fontsize=12, color='red')
ax2.text(ek_x+0.2, ek_y+0.2, r'$e_k$', fontsize=12, color='darkblue')

ax2.text(0, -3.5, "离散码本特征空间", ha='center', fontsize=13)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.axis('off')

# 自动紧凑排版并保存
from thesis_code.config import PROJECT_ROOT
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Fig_2_1_LatentSpace.png")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"[DONE] Saved to: {OUTPUT_PATH}")
