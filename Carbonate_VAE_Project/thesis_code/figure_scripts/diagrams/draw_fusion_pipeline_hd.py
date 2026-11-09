"""
Figure 3.2 — 连续特征融合 + VQ量化层 超高清流程图
=============================================================
核心逻辑：
  1. 两张致密岩心切片 A/B → 冻结编码器 → 连续潜特征 Z_eA / Z_eB
  2. 有机掩膜 M 在连续流形域做凸组合 → Z_fused (连续)
  3. Z_fused 必须经过 VQ 层重新离散化 → Z_q_fused (离散物理基元)
  4. 冻结解码器 → 无伪影新岩心

关键设计原则：
  - "融合在前，量化在后" (Fuse-then-VQ)
  - VQ 层是理论闭环中不可跳过的环节
  - 600 DPI, 无重叠, 大字体
  - SimHei 原生粗黑体，不加 fontweight='bold'（避免 bold 变体缺中文字形）
"""

from thesis_code.config import DIAGRAM_DIR
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# SimHei 是原生粗黑体，无需额外指定 bold
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Canvas setup
# ============================================================
fig, ax = plt.subplots(figsize=(22, 11), dpi=300)
ax.set_xlim(0, 24)
ax.set_ylim(0, 12)
ax.axis('off')

# Color palette
C_INPUT      = '#fde4c8'
C_INPUT_EC   = '#e67e22'
C_ENCODER    = '#d4e6f1'
C_ENCODER_EC = '#2980b9'
C_LATENT     = '#d5f5e3'
C_LATENT_EC  = '#27ae60'
C_FUSION     = '#fef9e7'
C_FUSION_EC  = '#f1c40f'
C_VQ         = '#fadbd8'
C_VQ_EC      = '#e74c3c'
C_DECODER    = '#d4e6f1'
C_DECODER_EC = '#2980b9'
C_OUTPUT     = '#d5f5e3'
C_OUTPUT_EC  = '#1e8449'
C_MASK       = '#ebdef0'
C_MASK_EC    = '#8e44ad'

FONT_BOX    = 15
FONT_SMALL  = 13
FONT_TITLE  = 18
FONT_ANNO   = 14
FONT_VQ     = 14
FONT_FUSION = 12

BOX_W = 3.0
BOX_H = 1.5

# ============================================================
# Helper functions
# ============================================================
def draw_box(x, y, w, h, text, fc, ec, fontsize=FONT_BOX, fontcolor='black',
             pad=0.3, lw=2.5):
    """Draw a rounded box, return (right_center, left_center, top_center, bottom_center)."""
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad={pad},rounding_size=0.25",
        facecolor=fc, edgecolor=ec, lw=lw, zorder=3
    )
    ax.add_patch(rect)
    # 不加 fontweight='bold' —— SimHei 本身就是粗黑体
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=fontcolor, zorder=4)
    return ((x + w/2, y), (x - w/2, y), (x, y + h/2), (x, y - h/2))


def draw_arrow(p1, p2, color='#555555', lw=2.5, style='-|>', zorder=2):
    """Draw a styled arrow from p1 to p2."""
    ax.annotate('', xy=p2, xytext=p1, zorder=zorder,
                arrowprops=dict(arrowstyle=style, facecolor=color,
                                edgecolor='none', lw=lw))


# ============================================================
# Layout coordinates
# ============================================================
X_INPUT   = 2.0
X_ENCODER = 6.0
X_LATENT  = 10.0
X_FUSION  = 14.5
X_VQ      = 18.5
X_DECODER = 18.5
X_OUTPUT  = 18.5

Y_A       = 9.5
Y_B       = 3.0
Y_FUSION  = 6.0
Y_MASK    = 7.5
Y_DECODER = 1.0

# ============================================================
# Draw blocks
# ============================================================

# 1. Input
r_inA, _, _, _ = draw_box(X_INPUT, Y_A, BOX_W, BOX_H,
    "致密岩心切片 A", C_INPUT, C_INPUT_EC)
r_inB, _, _, _ = draw_box(X_INPUT, Y_B, BOX_W, BOX_H,
    "致密岩心切片 B", C_INPUT, C_INPUT_EC)

# 2. Encoder (frozen)
r_encA, l_encA, _, _ = draw_box(X_ENCODER, Y_A, BOX_W, BOX_H + 0.2,
    "CNN 编码器\n(冻结权重)", C_ENCODER, C_ENCODER_EC, fontsize=FONT_SMALL)
r_encB, l_encB, _, _ = draw_box(X_ENCODER, Y_B, BOX_W, BOX_H + 0.2,
    "CNN 编码器\n(冻结权重)", C_ENCODER, C_ENCODER_EC, fontsize=FONT_SMALL)

# 3. Continuous latent features
r_zA, l_zA, _, b_zA = draw_box(X_LATENT, Y_A, BOX_W, BOX_H + 0.1,
    "连续潜特征\n$Z_{eA} \\in \\mathbb{R}^{64}$",
    C_LATENT, C_LATENT_EC, fontsize=FONT_SMALL)
r_zB, l_zB, t_zB, _ = draw_box(X_LATENT, Y_B, BOX_W, BOX_H + 0.1,
    "连续潜特征\n$Z_{eB} \\in \\mathbb{R}^{64}$",
    C_LATENT, C_LATENT_EC, fontsize=FONT_SMALL)

# 4. Organic mask source
_, _, _, b_mask = draw_box(X_LATENT, Y_MASK, 2.2, 0.9,
    "有机掩膜 $M$\n(GRF 生成)",
    C_MASK, C_MASK_EC, fontsize=FONT_SMALL)

# 5. Continuous manifold fusion (core innovation)
fusion_text = ("$\\mathbf{Z_{fused}} = M \\odot Z_{eA} + (1-M) \\odot Z_{eB}$"
               "\n64维连续向量逐位置凸组合\n"
               "═══════════════════")
r_fus, l_fus, _, b_fus = draw_box(X_FUSION, Y_FUSION, 4.2, 2.2, fusion_text,
    C_FUSION, C_FUSION_EC, fontsize=FONT_FUSION, fontcolor='#7d6608', lw=3.0)

# 6. VQ quantization layer (critical theoretical closure)
#    Note: \mathbf{} in LaTeX cannot render CJK — split Chinese out of math mode
vq_text = ("$\\mathbf{VQ}$ 离散量化\n"
           "最近邻码本匹配\n"
           "$\\mathbf{Z_{q\\_fused}}$\n"
           "← 512个物理基元")
r_vq, l_vq, t_vq, b_vq = draw_box(X_VQ, Y_FUSION, 3.0, 2.8, vq_text,
    C_VQ, C_VQ_EC, fontsize=FONT_VQ, fontcolor='#922b21', lw=3.5)

# 7. Decoder (frozen)
_, _, t_dec, b_dec = draw_box(X_DECODER, Y_DECODER + 0.3, BOX_W, BOX_H,
    "CNN 解码器\n(冻结权重)",
    C_DECODER, C_DECODER_EC, fontsize=FONT_SMALL)

# 8. Output
r_out, _, _, _ = draw_box(X_OUTPUT, Y_DECODER - 1.5, 3.2, 1.2,
    "生成新岩心\n(无拼接伪影)",
    C_OUTPUT, C_OUTPUT_EC, fontsize=FONT_BOX, fontcolor='#0e5c2e', lw=3.0)

# ============================================================
# Draw arrows
# ============================================================

# Input → Encoder
draw_arrow(r_inA, l_encA)
draw_arrow(r_inB, l_encB)

# Encoder → Latent features
draw_arrow(r_encA, l_zA)
draw_arrow(r_encB, l_zB)

# Z_eA → Fusion (from bottom of Z_eA)
draw_arrow((X_LATENT, Y_A - BOX_H/2 - 0.05), (X_FUSION - 2.1, Y_FUSION + 1.1))

# Z_eB → Fusion (from top of Z_eB)
draw_arrow((X_LATENT, Y_B + BOX_H/2 + 0.05), (X_FUSION - 2.1, Y_FUSION - 1.1))

# Mask → Fusion (vertical from mask bottom to near fusion top area)
draw_arrow(b_mask, (X_LATENT, Y_FUSION + 0.5))

# Fusion → VQ
draw_arrow(r_fus, l_vq)

# VQ → Decoder
draw_arrow(b_vq, t_dec)

# Decoder → Output
draw_arrow(b_dec, (X_OUTPUT, Y_DECODER - 0.75))

# ============================================================
# Dashed region: Continuous Manifold Domain
# ============================================================
rect_cont = patches.Rectangle(
    (X_LATENT - 2.0, Y_FUSION - 1.8),
    6.5, 3.8,
    linewidth=2.5, edgecolor='#999999',
    facecolor='#f8f8f8', linestyle=(0, (6, 4)), zorder=0, alpha=0.6
)
ax.add_patch(rect_cont)
ax.text(X_LATENT + 1.25, Y_FUSION + 2.1, "连续流形域 (拦截区)",
        fontsize=FONT_ANNO, color='#666666',
        ha='center', va='bottom')

# ============================================================
# Red annotation: VQ theoretical closure
# ============================================================
ax.text(X_VQ + 2.2, Y_FUSION, "★ 理论闭环\n必须经过\nVQ 重新量化",
        fontsize=FONT_SMALL, color='#c0392b',
        ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdecea',
                  edgecolor='#e74c3c', lw=2))

# ============================================================
# Phase labels at top
# ============================================================
for x_pos, label in [
    (X_INPUT,    "① 输入"),
    (X_ENCODER,  "② 编码"),
    (X_LATENT,   "③ 连续潜空间"),
    (X_FUSION,   "④ 连续融合"),
    (X_VQ,       "⑤ VQ 离散化"),
    (X_DECODER,  "⑥ 解码重建"),
]:
    ax.text(x_pos, 11.2, label,
            fontsize=FONT_ANNO, color='#444444',
            ha='center', va='center')

# ============================================================
# Title
# ============================================================
ax.text(10.25, 11.8,
        "图 3.2  连续潜特征融合 + VQ 离散量化完整流程图 (Fuse-then-VQ)",
        fontsize=FONT_TITLE, color='#1a1a1a',
        ha='center', va='center')

# Subtitle
ax.text(10.25, 11.35,
        "核心创新：在 VQ 量化前于连续流形域完成掩膜融合，再统一经码本映射到离散物理基元域",
        fontsize=FONT_SMALL, color='#555555',
        ha='center', va='center', style='italic')

# ============================================================
# Save
# ============================================================
output_path = str(DIAGRAM_DIR / 'Figure_3_2_Fusion_Pipeline_HD.png')
plt.tight_layout(pad=0.5)
fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"[DONE] Saved -> {output_path}")
