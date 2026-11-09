"""
figure_3_3_sensitivity.py — λ_porosity 敏感性分析 (双 Y 轴折线图)
=================================================================
左 Y 轴：孔隙度绝对偏差 (实线 + 圆点标记)
右 Y 轴：MAE 重建误差 (虚线 + 方块标记)
合并图例置于图表上方，300 DPI 学术排版标准。

数据来源：eval_one_model.py 对 200 张固定测试集 (seed=999) 的真实评估结果
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== 必须字体配置 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════════════
# 真实评估数据 (200 张固定测试集, seed=999)
# ═══════════════════════════════════════════════════════════
LAMBDA_VALS  = np.array([10.0, 20.0, 40.0])
POR_DEV_MEAN = np.array([0.005194, 0.002198, 0.074343])
POR_DEV_STD  = np.array([0.003662, 0.002046, 0.055178])
MAE_MEAN     = np.array([0.035420, 0.032749, 0.168098])
MAE_STD      = np.array([0.006738, 0.006153, 0.053558])

X_LABELS = [r'$\lambda_\phi = 10$',
            r'$\lambda_\phi = 20$',
            r'$\lambda_\phi = 40$']

# ═══════════════════════════════════════════════════════════
# 配色
# ═══════════════════════════════════════════════════════════
COLOR_POR = '#2C6FAC'   # 蓝色 — 孔隙度偏差 (左轴)
COLOR_MAE = '#D9583B'   # 橙红色 — MAE (右轴)

# ═══════════════════════════════════════════════════════════
# 绘图 — 8×6 宽敞画布
# ═══════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8, 6), facecolor='white')
ax1.set_facecolor('white')

x_pos = np.arange(len(LAMBDA_VALS))  # 数值坐标 0, 1, 2

# ── 左 Y 轴：孔隙度绝对偏差 (实线 + 圆点) ──
line1, = ax1.plot(x_pos, POR_DEV_MEAN,
                  marker='o', linestyle='-',
                  color=COLOR_POR, linewidth=2.2, markersize=10,
                  markerfacecolor='white', markeredgewidth=2.2,
                  zorder=5, label='孔隙度绝对偏差 (左轴)')

# 左轴数值标注 (在点上方)
for xp, val in zip(x_pos, POR_DEV_MEAN):
    ax1.text(xp, val + 0.006,
             f'{val:.4f}', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color=COLOR_POR)

ax1.set_xlabel(r'$\lambda_\phi$ (孔隙度惩罚权重)', fontsize=14, fontweight='bold')
ax1.set_ylabel('孔隙度绝对偏差 |$\phi_{pred} - \phi_{real}$|',
               fontsize=14, fontweight='bold', color=COLOR_POR)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(X_LABELS, fontsize=12)
ax1.tick_params(axis='y', labelcolor=COLOR_POR, colors=COLOR_POR, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

# 左轴留出缓冲空间
por_min, por_max = POR_DEV_MEAN.min(), POR_DEV_MEAN.max()
por_pad = (por_max - por_min) * 0.30 + 0.005
ax1.set_ylim(max(0, por_min - por_pad), por_max + por_pad * 2.0)

ax1.grid(axis='y', linestyle='--', alpha=0.30, color='#aaaaaa', linewidth=0.6)
ax1.set_axisbelow(True)

# ── 右 Y 轴：MAE (虚线 + 方块) ──
ax2 = ax1.twinx()
line2, = ax2.plot(x_pos, MAE_MEAN,
                  marker='s', linestyle='--',
                  color=COLOR_MAE, linewidth=2.2, markersize=10,
                  markerfacecolor='white', markeredgewidth=2.2,
                  zorder=4, label='MAE 重建误差 (右轴)')

# 右轴数值标注 (在点上方)
for xp, val in zip(x_pos, MAE_MEAN):
    ax2.text(xp, val + 0.013,
             f'{val:.4f}', ha='center', va='bottom',
             fontsize=11, fontweight='bold', color=COLOR_MAE)

ax2.set_ylabel('MAE 重建误差', fontsize=14, fontweight='bold', color=COLOR_MAE)
ax2.tick_params(axis='y', labelcolor=COLOR_MAE, colors=COLOR_MAE, labelsize=12)

# 右轴留出缓冲空间
mae_min, mae_max = MAE_MEAN.min(), MAE_MEAN.max()
mae_pad = (mae_max - mae_min) * 0.30 + 0.01
ax2.set_ylim(max(0, mae_min - mae_pad), mae_max + mae_pad * 2.0)

# ── 隐藏顶部 / 右侧多余轴线 ──
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['left'].set_color(COLOR_POR)
ax1.spines['bottom'].set_color('#555555')
ax2.spines['right'].set_color(COLOR_MAE)

# ── 合并图例 (置于图表上方) ──
legend = fig.legend(handles=[line1, line2],
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.06),
                    ncol=2,
                    frameon=False,
                    fontsize=12)

# 图例文字颜色与对应曲线一致
for text, color in zip(legend.get_texts(), [COLOR_POR, COLOR_MAE]):
    text.set_color(color)
    text.set_fontweight('bold')

# ── 标题 ──
ax1.set_title(r'$\lambda_\phi$ 敏感性分析：孔隙度偏差与重建误差',
              fontsize=14, fontweight='bold', pad=18)

# ═══════════════════════════════════════════════════════════
# 保存 (tight_layout + 300 DPI)
# ═══════════════════════════════════════════════════════════
from thesis_code.config import PROJECT_ROOT
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_3_3_Sensitivity.png")

fig.tight_layout(rect=[0, 0, 1, 0.95])  # 为顶部图例留空间
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.3)
plt.close(fig)

print(f"[DONE] 敏感性分析图已保存至: {OUTPUT_PATH}")
print(f"  数据点: λ=10, 20, 40")
print(f"  PorDev: {POR_DEV_MEAN}")
print(f"  MAE:    {MAE_MEAN}")
