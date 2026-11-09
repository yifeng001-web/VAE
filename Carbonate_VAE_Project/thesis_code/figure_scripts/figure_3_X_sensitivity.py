"""
figure_3_X_sensitivity.py —— 参数敏感性分析: 孔隙度权重 lambda_phi 对 MAE 和孔隙度偏差的影响
============================================================================================
四组实验数据 (N=200 测试切片):
  lambda= 0.0: MAE=0.035302, PorDev=0.010992  (Baseline, 仅视觉损失)
  lambda=10.0: MAE=0.035420, PorDev=0.005194  (中等物理约束)
  lambda=20.0: MAE=0.032749, PorDev=0.002198  (Proposed, 最佳权衡点)
  lambda=40.0: MAE=0.168098, PorDev=0.074343  (过度约束, 双重恶化)

双 Y 轴图:
  左轴 (蓝色柱状图): 孔隙度绝对偏差 |phi_GT - phi_recon|
  右轴 (红色虚线): 像素重构误差 MAE
"""

import os
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
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(EVAL_DIR, "figure_3_X_sensitivity.png")

# ===== 真实实验数据 =====
LAMBDA_VALS = np.array([0.0, 10.0, 20.0, 40.0], dtype=np.float32)

# MAE (像素重构误差) — mean +/- std
MAE_MEAN = np.array([0.035302, 0.035420, 0.032749, 0.168098], dtype=np.float32)
MAE_STD  = np.array([0.006400, 0.006738, 0.006153, 0.053558], dtype=np.float32)

# 孔隙度绝对偏差 — mean +/- std
POR_DEV_MEAN = np.array([0.010992, 0.005194, 0.002198, 0.074343], dtype=np.float32)
POR_DEV_STD  = np.array([0.006905, 0.003662, 0.002046, 0.055178], dtype=np.float32)

# 学术配色
C_DARK = '#1a1a1a'
C_POROSITY = '#2C6FAC'    # 孔隙度偏差 — 蓝色
C_MAE = '#D9583B'         # MAE — 红色
C_OPTIMAL = '#27AE60'     # 最优 — 绿色


def main():
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6.5), facecolor='white')
    ax1.set_facecolor('white')

    # 去掉顶部 spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color('#555555')
    ax1.spines['bottom'].set_color('#555555')
    ax1.spines['right'].set_color(C_MAE)
    ax1.spines['right'].set_linewidth(1.5)

    # 网格
    ax1.grid(True, linestyle='--', alpha=0.30, color='#aaaaaa', linewidth=0.6, axis='y')
    ax1.set_axisbelow(True)

    # ═════════════════════════════════════════
    # 左轴: 孔隙度绝对偏差 (柱状图)
    # ═════════════════════════════════════════
    bar_width = 3.0
    x_positions = LAMBDA_VALS.copy()

    bars = ax1.bar(x_positions, POR_DEV_MEAN, width=bar_width,
                   color=C_POROSITY, alpha=0.75, edgecolor='white',
                   linewidth=1.2, zorder=3, label='孔隙度绝对偏差')

    # 误差线
    ax1.errorbar(x_positions, POR_DEV_MEAN, yerr=POR_DEV_STD,
                 fmt='none', ecolor='#1a3a5c', capsize=6, capthick=1.5,
                 linewidth=1.5, zorder=5)

    # 柱状图顶部标注数值
    for x, val, std in zip(x_positions, POR_DEV_MEAN, POR_DEV_STD):
        offset = 0.004 if val < 0.05 else 0.012
        ax1.text(x, val + std + offset, f'{val:.4f}',
                 ha='center', va='bottom', fontsize=10,
                 fontweight='bold', color=C_POROSITY)

    ax1.set_xlabel('孔隙度惩罚权重 $\\lambda_{\\phi}$', fontsize=13,
                   fontweight='bold', color=C_DARK)
    ax1.set_ylabel('孔隙度绝对偏差 $|\\phi_{GT} - \\phi_{recon}|$',
                   fontsize=12, fontweight='bold', color=C_POROSITY)
    ax1.tick_params(axis='y', labelcolor=C_POROSITY, colors=C_POROSITY)
    ax1.set_ylim(0, POR_DEV_MEAN.max() * 1.3)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'{v:.0f}' for v in x_positions])

    # ═════════════════════════════════════════
    # 右轴: MAE (红色虚线 + 标记)
    # ═════════════════════════════════════════
    ax2 = ax1.twinx()
    ax2.spines['right'].set_color(C_MAE)
    ax2.spines['right'].set_linewidth(1.5)

    ax2.plot(x_positions, MAE_MEAN, 'o-', color=C_MAE, linewidth=2.8,
             markersize=9, markerfacecolor='white', markeredgewidth=2.0,
             zorder=6, label='像素重构误差 (MAE)')

    # MAE 误差阴影
    ax2.fill_between(x_positions,
                     MAE_MEAN - MAE_STD,
                     MAE_MEAN + MAE_STD,
                     alpha=0.15, color=C_MAE, zorder=2)

    # MAE 标注 (偏移到右上方避免与柱状图重叠)
    for x, val, std in zip(x_positions, MAE_MEAN, MAE_STD):
        offset_y = 0.008 if val < 0.1 else 0.02
        ax2.annotate(f'{val:.3f}',
                     xy=(x, val),
                     xytext=(x + 0.6, val + std + offset_y),
                     fontsize=9.5, fontweight='bold', color=C_MAE,
                     arrowprops=dict(arrowstyle='->', color=C_MAE,
                                     lw=1.2, alpha=0.8),
                     zorder=10)

    ax2.set_ylabel('像素重构误差 (MAE)', fontsize=12,
                   fontweight='bold', color=C_MAE)
    ax2.tick_params(axis='y', labelcolor=C_MAE, colors=C_MAE)
    ax2.yaxis.set_major_locator(MaxNLocator(6))

    # ═════════════════════════════════════════
    # 标注最优权衡点 (λ=20)
    # ═════════════════════════════════════════
    optimal_idx = 2  # λ=20
    ax1.annotate(
        '最优权衡点\n(λ=20, 本文采用)',
        xy=(x_positions[optimal_idx], POR_DEV_MEAN[optimal_idx]),
        xytext=(x_positions[optimal_idx] + 6, POR_DEV_MEAN[optimal_idx] + 0.008),
        fontsize=10, fontweight='bold', color=C_OPTIMAL,
        arrowprops=dict(arrowstyle='->', color=C_OPTIMAL, lw=2.0,
                        connectionstyle='arc3,rad=0.3'),
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                  edgecolor=C_OPTIMAL, alpha=0.9),
        zorder=10)

    # 标注过正则化区域 (λ=40)
    ax1.annotate(
        '过正则化\n(MAE急剧恶化)',
        xy=(x_positions[3], POR_DEV_MEAN[3]),
        xytext=(x_positions[3] - 5, POR_DEV_MEAN[3] - 0.025),
        fontsize=9.5, fontweight='bold', color='#C0392B',
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.8,
                        connectionstyle='arc3,rad=-0.3'),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC',
                  edgecolor='#C0392B', alpha=0.9),
        zorder=10)

    # ═════════════════════════════════════════
    # 图例
    # ═════════════════════════════════════════
    from matplotlib.lines import Line2D
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=C_POROSITY, alpha=0.75,
                      edgecolor='white', linewidth=1.2,
                      label='孔隙度绝对偏差 (左轴)'),
        Line2D([0], [0], color=C_MAE, linewidth=2.8, marker='o',
               markerfacecolor='white', markeredgewidth=2.0, markersize=9,
               label='像素重构误差 MAE (右轴)'),
    ]
    legend = ax1.legend(handles=legend_elements, loc='upper left',
                        fontsize=10.5, frameon=True,
                        facecolor='white', edgecolor='#cccccc',
                        framealpha=0.95)
    legend.set_zorder(20)

    # ═════════════════════════════════════════
    # 标题
    # ═════════════════════════════════════════
    ax1.set_title(
        '参数敏感性分析: 孔隙度惩罚权重 $\\lambda_{\\phi}$ 对重构质量的影响\n'
        'Parameter Sensitivity — Trade-off Between Porosity Fidelity and Pixel Accuracy',
        fontsize=13.5, fontweight='bold', color=C_DARK, pad=15)

    # 底部说明
    fig.text(0.5, 0.01,
             '固定参数: $\\lambda_{MAE}$=10.0, $\\lambda_{SSIM}$=5.0, '
             '$\\lambda_{Edge}$=10.0。测试集 N=200, 50 Epochs。'
             '误差线表示 ±1 标准差。',
             ha='center', fontsize=8.5, color='#888888', fontstyle='italic')

    plt.tight_layout(pad=1.5)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)

    # ── 打印数据表 ──
    print("\n" + "=" * 70)
    print("  参数敏感性分析 — 四组实验数据汇总")
    print("=" * 70)
    print(f"  {'lambda_phi':>10}  {'MAE':>12}  {'+- std':>10}  {'PorDev':>10}  {'+- std':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for lam, mae_m, mae_s, por_m, por_s in zip(LAMBDA_VALS, MAE_MEAN, MAE_STD,
                                                 POR_DEV_MEAN, POR_DEV_STD):
        print(f"  {lam:10.1f}  {mae_m:12.6f}  {mae_s:10.6f}  {por_m:10.6f}  {por_s:10.6f}")
    print("=" * 70)
    print(f"\n[DONE] 敏感性分析图已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
