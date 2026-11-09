
from thesis_code.config import DIAGRAM_DIR
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(55, 16), dpi=300)
ax.set_xlim(0, 68)
ax.set_ylim(-1, 13)
ax.axis('off')

FONT = 26
LW = 3
AW = 3.5

def draw_box(x, y, w, h, text, fc, ec, ls='-', text_color='black', fontsize=FONT):
    rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15,rounding_size=0.3",
                                  facecolor=fc, edgecolor=ec, lw=LW, linestyle=ls)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=text_color)
    return (x+w/2, y), (x-w/2, y), (x, y+h/2), (x, y-h/2)

def draw_trapezoid(x, y, w, h_left, h_right, text, fc, ec):
    pts = [[x-w/2, y-h_left/2], [x+w/2, y-h_right/2], [x+w/2, y+h_right/2], [x-w/2, y+h_left/2]]
    poly = patches.Polygon(pts, facecolor=fc, edgecolor=ec, lw=LW)
    ax.add_patch(poly)
    ax.text(x, y, text, ha='center', va='center', fontsize=FONT)
    return (x+w/2, y), (x-w/2, y)

def draw_arrow(p1, p2, color='#333333', ls='-'):
    ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color=color, lw=AW, ls=ls, shrinkA=0, shrinkB=0))

def draw_curved_arrow(p1, p2, color, rad=0.4):
    arrow = patches.FancyArrowPatch(
        p1, p2, arrowstyle=patches.ArrowStyle('->', head_length=12, head_width=8),
        color=color, lw=AW, connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(arrow)

# ================= 1. 上半部分：致密岩心样本 A =================
y_top = 8.8
rA, _, _, _ = draw_box(3.5, y_top, 5.0, 2.6, "致密岩心样本 A\n$x_A$", '#f4f4f4', '#333333')
rEA, lEA = draw_trapezoid(12.5, y_top, 5.5, 3.8, 2.4, "编码器\nEncoder", '#e2f0d9', '#548235')
rZA, lZA, _, bZA = draw_box(20.5, y_top, 4.5, 2.6, "连续潜特征\n$z_{eA}$", '#f2e6f2', '#9b59b6')

draw_arrow(rA, lEA)
draw_arrow(rEA, lZA)

# ================= 2. 下半部分：高孔隙样本 B =================
y_bot = 1.5
rB, _, _, _ = draw_box(3.5, y_bot, 5.0, 2.6, "高孔隙样本 B\n$x_B$", '#f4f4f4', '#333333')
rHist, lHist, _, _ = draw_box(12, y_bot, 4.5, 2.6, "直方图匹配\n(基线平齐)", '#fff2cc', '#d6b656')
rEB, lEB = draw_trapezoid(19.5, y_bot, 5.5, 3.8, 2.4, "编码器\nEncoder", '#e2f0d9', '#548235')
rZB, lZB, tZB, _ = draw_box(26.5, y_bot, 4.5, 2.6, "连续潜特征\n$z_{eB}$", '#e2efda', '#548235')

draw_arrow(rB, lHist)
draw_arrow(rHist, lEB)
draw_arrow(rEB, lZB)

# ================= 3. 融合核心 =================
y_mid = 5
fusion_text = "连续潜空间特征掩膜融合\n\n$Z_{fused} = M \odot z_{eB} + (1 - M) \odot z_{eA}$"
rF, lF, tF, _ = draw_box(34.5, y_mid, 9.5, 3.4, fusion_text, '#fdfdfd', '#5b9bd5', ls='--')

# 软掩膜模块 (在融合框正上方，远离 z_eA)
_, _, _, bMask = draw_box(34.5, 11.5, 4.0, 2.2, "平滑高斯软掩膜\n(Soft Mask)\n$M$", '#cfe2f3', '#3d85c6', text_color='#0b5394')
draw_arrow(bMask, tF, color='#3d85c6')

# 曲线箭头：z_eA → 融合, z_eB → 融合
draw_curved_arrow((rZA[0], y_top), (lF[0], y_mid + 0.9), color='#9b59b6', rad=-0.35)
draw_curved_arrow((rZB[0], y_bot), (lF[0], y_mid - 0.9), color='#548235', rad=0.35)

# ================= 4. 融合特征 -> VQ -> 解码 -> 输出 =================
rZF, lZF, _, _ = draw_box(41.5, y_mid, 3.6, 2.6, "融合潜特征\n$Z_{fused}$", '#fce4d6', '#ed7d31')

rVQ, lVQ, _, _ = draw_box(47.5, y_mid, 4.2, 3.6, "矢量量化匹配\n(VQ 层)\n\n$Quantize(Z_{fused})$", '#f8cecc', '#cc0000', text_color='#990000')

rDec, lDec = draw_trapezoid(54, y_mid, 5.5, 2.4, 4.4, "冻结的解码器\nFrozen Decoder", '#e2efda', '#548235')
_, lOut, _, _ = draw_box(61.5, y_mid, 5.0, 2.6, "高保真融合数字岩心\n$\hat{x}_{new}$", '#f4f4f4', '#333333')

draw_arrow(rF, lZF)
draw_arrow(rZF, lVQ)
draw_arrow(rVQ, lDec)
draw_arrow(rDec, lOut)

plt.tight_layout()
plt.savefig(str(DIAGRAM_DIR / 'Figure_Original_Style_Pipeline_HD.png'), dpi=600, bbox_inches='tight')
print("原图风格的高清流水线架构图已成功生成！(已包含关键的 VQ 量化层修正)")
