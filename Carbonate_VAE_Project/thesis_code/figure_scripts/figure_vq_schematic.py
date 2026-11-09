"""
figure_vq_schematic.py — Refined VQ-VAE Vector Quantization Mapping Principle
=============================================================================
Academic flat-design schematic: strictly left-to-right horizontal flow.
Components:  2D z_e(x) grid → Comparison & Distance Calculation →
             Learnable Codebook (E) → VQ Logic (argmin) →
             2D z_q(x) grid → Decoder (G) → Reconstructed Image.
All coordinates, colors, and formulas are precisely aligned.
v3: Chinese + English bilingual labels, fixed bottom circle clipping.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import os

# ===== Academic style (Chinese-capable) =====
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'

from thesis_code.config import PROJECT_ROOT
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "vq_schematic.png")

# ===== Color palette =====
C_DARK   = '#222222'
C_GRAY   = '#555555'
C_LIGHT  = '#888888'
C_BLUE_GRID = '#D6EAF8'       # light blue for z_e cells
C_BLUE_EDGE = '#2980B9'
C_BLOCK_BG  = '#F8F8F8'
C_BLOCK_EDGE= '#444444'

# Codebook index colors (5 distinct)
CB_COLORS = {
    1: '#E74C3C',  # red
    2: '#2ECC71',  # green
    3: '#3498DB',  # blue
    4: '#F39C12',  # orange
    5: '#9B59B6',  # purple
}

# Grid cell → codebook index mapping (deterministic, visually diverse)
# 4×4 grid, (row, col) 0-indexed → codebook index
GRID_MAP = [
    [1, 2, 3, 4],
    [2, 5, 1, 3],
    [4, 3, 5, 2],
    [3, 1, 2, 5],
]

# ===== Coordinate system: 16 × 12 =====
FIG_W, FIG_H = 16, 12


def rounded_box(ax, x, y, w, h, text="", sub_text="", facecolor=C_BLOCK_BG,
                edgecolor=C_BLOCK_EDGE, lw=1.8, fontsize=12, text_color=C_DARK,
                sub_color=C_GRAY, sub_size=10, bold=True, pad=0.08):
    """Draw a FancyBboxPatch with centered text and optional subtitle."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={pad}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=3)
    ax.add_patch(box)
    if text:
        cy = y + h / 2
        if sub_text:
            cy += 0.18
            ax.text(x + w/2, y + h/2 - 0.18, sub_text,
                    ha='center', va='center', fontsize=sub_size, color=sub_color,
                    fontstyle='italic', zorder=5)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, cy, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=5)


def arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=2.0, style='-|>', zorder=4):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0'),
                zorder=zorder)


def draw_grid(ax, gx, gy, cell_w, cell_h, rows, cols, cell_colors,
              edge_color, labels=None, fontsize=10, label_color=C_DARK):
    """
    Draw a 2D grid of colored cells.
    cell_colors: 2D list of (r,g,b) or color strings per cell.
    labels: 2D list of strings per cell.
    """
    for r in range(rows):
        for c in range(cols):
            cx = gx + c * cell_w
            cy = gy + (rows - 1 - r) * cell_h  # top-to-bottom
            color = cell_colors[r][c] if isinstance(cell_colors, list) else cell_colors
            rect = Rectangle((cx, cy), cell_w, cell_h,
                             facecolor=color, edgecolor=edge_color,
                             linewidth=1.5, zorder=3)
            ax.add_patch(rect)
            if labels:
                lbl = labels[r][c] if isinstance(labels, list) else labels
                ax.text(cx + cell_w/2, cy + cell_h/2, str(lbl),
                        ha='center', va='center', fontsize=fontsize,
                        color=label_color, fontweight='bold', zorder=5)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect('equal')
    ax.axis('off')

    # ===== Title (bilingual) =====
    ax.text(8, 11.4, 'Refined VQ-VAE 矢量量化映射原理图',
            ha='center', va='center', fontsize=22, fontweight='bold', color='#111111')
    ax.text(8, 10.85, '(Vector Quantization Mapping Principle)',
            ha='center', va='center', fontsize=14, color=C_GRAY, fontstyle='italic')

    # ================================================================
    # Common grid params
    # ================================================================
    GRID_ROWS, GRID_COLS = 4, 4
    CELL_W, CELL_H = 0.90, 0.80
    GRID_TOTAL_W = GRID_COLS * CELL_W   # 3.6
    GRID_TOTAL_H = GRID_ROWS * CELL_H   # 3.2

    # ================================================================
    # LEFT: Continuous Feature Map  z_e(x)  — 4×4 blue grid
    # ================================================================
    ZE_GX, ZE_GY = 1.0, 4.5

    # Label above grid (bilingual)
    ax.text(ZE_GX + GRID_TOTAL_W/2, ZE_GY + GRID_TOTAL_H + 0.6,
            'z_e(x) : 连续特征图\n(Continuous Feature Map)',
            ha='center', va='center', fontsize=14, fontweight='bold', color=C_BLUE_EDGE,
            linespacing=1.3)

    # Draw blue cells
    ze_colors = [[C_BLUE_GRID] * GRID_COLS for _ in range(GRID_ROWS)]
    ze_labels = [[f'z_e({i+1},{j+1})' for j in range(GRID_COLS)]
                  for i in range(GRID_ROWS)]
    draw_grid(ax, ZE_GX, ZE_GY, CELL_W, CELL_H, GRID_ROWS, GRID_COLS,
              ze_colors, C_BLUE_EDGE, labels=ze_labels, fontsize=10,
              label_color='#1a5276')

    # Axis labels
    ax.text(ZE_GX + GRID_TOTAL_W/2, ZE_GY - 0.35,
            '空间位置 (Spatial Position j) →', ha='center', va='center',
            fontsize=11, color=C_GRAY, fontstyle='italic')
    ax.text(ZE_GX - 0.5, ZE_GY + GRID_TOTAL_H/2,
            '← (i)', ha='center', va='center', rotation=90,
            fontsize=11, color=C_GRAY, fontstyle='italic')

    # ================================================================
    # TOP-CENTER: Comparison & Distance Calculation Block
    # ================================================================
    COMP_X, COMP_Y = 5.0, 9.1
    COMP_W, COMP_H = 4.2, 2.0
    rounded_box(ax, COMP_X, COMP_Y, COMP_W, COMP_H,
                text='距离比较与计算\n(Comparison & Distance Calculation)',
                sub_text='',
                fontsize=12, text_color=C_DARK)

    # Formula inside
    ax.text(COMP_X + COMP_W/2, COMP_Y + 0.65,
            'd(z_e(i,j), e_k) = || z_e(i,j) - e_k ||_2',
            ha='center', va='center', fontsize=12.5, color=C_DARK, zorder=5)

    # Arrow: z_e grid → Comparison block
    arrow(ax, ZE_GX + GRID_TOTAL_W/2, ZE_GY + GRID_TOTAL_H + 0.15,
          ZE_GX + GRID_TOTAL_W/2, COMP_Y - 0.08, C_GRAY, 2.0)

    # ================================================================
    # MIDDLE: Learnable Vectors Codebook (E) — vertical bars
    # ================================================================
    CB_X, CB_Y = 5.2, 4.5
    CB_BAR_W, CB_BAR_H = 3.2, 0.58
    CB_GAP = 0.14
    K = 5

    # Codebook bounding box
    cb_total_h = K * CB_BAR_H + (K - 1) * CB_GAP
    cb_box = FancyBboxPatch(
        (CB_X - 0.25, CB_Y - 0.3), CB_BAR_W + 0.5, cb_total_h + 0.6,
        boxstyle="round,pad=0.08", facecolor='#FAFAFA',
        edgecolor='#666666', linewidth=1.8, linestyle=(0, (6, 2)), zorder=1
    )
    ax.add_patch(cb_box)

    # Title on top (bilingual)
    ax.text(CB_X + CB_BAR_W/2, CB_Y + cb_total_h + 0.5,
            '可学习码本 E\n(Learnable Codebook)',
            ha='center', va='center', fontsize=13, fontweight='bold', color=C_DARK,
            linespacing=1.3)

    # Draw each codebook vector
    for i in range(K):
        by = CB_Y + (K - 1 - i) * (CB_BAR_H + CB_GAP)
        idx = i + 1
        color = CB_COLORS[idx]
        # Bar
        bar = FancyBboxPatch((CB_X, by), CB_BAR_W, CB_BAR_H,
                             boxstyle="round,pad=0.04",
                             facecolor=color, edgecolor='#333333',
                             linewidth=1.5, zorder=3)
        ax.add_patch(bar)
        # Label inside
        ax.text(CB_X + 0.25, by + CB_BAR_H/2,
                f'e_{idx}',
                ha='left', va='center', fontsize=12, fontweight='bold',
                color='white', zorder=5)
        # Dimension annotation
        ax.text(CB_X + CB_BAR_W - 0.18, by + CB_BAR_H/2,
                'R^D',
                ha='right', va='center', fontsize=10, color='white',
                fontstyle='italic', zorder=5)

    # Arrow: Comparison block → Codebook
    arrow(ax, COMP_X + COMP_W/2, COMP_Y - 0.08,
          CB_X + CB_BAR_W/2, CB_Y + cb_total_h + 0.15, C_GRAY, 2.0)

    # ================================================================
    # BOTTOM-CENTER: Vector Quantization Logic (argmin)
    # ================================================================
    VQ_X, VQ_Y = 5.0, 1.3
    VQ_W, VQ_H = 4.2, 1.8

    rounded_box(ax, VQ_X, VQ_Y, VQ_W, VQ_H,
                text='矢量量化逻辑 (argmin)\n(Vector Quantization Logic)',
                fontsize=12, text_color=C_DARK)

    # Formula inside
    ax.text(VQ_X + VQ_W/2, VQ_Y + 0.55,
            'k* = argmin  || z_e(i,j) - e_k ||_2\n'
            '       k ∈ {1,…,K}',
            ha='center', va='center', fontsize=12.5, color=C_DARK, zorder=5,
            linespacing=1.4)

    # Arrow: Codebook → VQ Logic
    arrow(ax, CB_X + CB_BAR_W/2, CB_Y - 0.08,
          CB_X + CB_BAR_W/2, VQ_Y + VQ_H + 0.08, C_GRAY, 2.0)

    # ================================================================
    # OUTPUT INDICATOR: Selected Codebook Vector
    # ================================================================
    SEL_X, SEL_Y = 9.8, 5.5
    SEL_W, SEL_H = 2.0, 1.5

    rounded_box(ax, SEL_X, SEL_Y, SEL_W, SEL_H,
                text='选中码本向量\n(Selected Vector)',
                sub_text='e_k* = z_q(i,j)',
                fontsize=11, text_color=C_DARK, sub_color='#2ECC71',
                sub_size=12, facecolor='#e8f8f0',
                edgecolor='#2ECC71')

    # Arrow: Codebook → Selected (from e2 position)
    e2_y = CB_Y + 3 * (CB_BAR_H + CB_GAP) + CB_BAR_H / 2  # middle of e₂ (index 2, 0-based pos 1)
    arrow(ax, CB_X + CB_BAR_W + 0.05, e2_y,
          SEL_X - 0.05, SEL_Y + SEL_H/2, '#2ECC71', 2.5)

    # ================================================================
    # RIGHT: Quantized Feature Map  z_q(x) — 4×4 colored grid
    # ================================================================
    ZQ_GX, ZQ_GY = 12.1, 4.5

    # Label above grid (bilingual)
    ax.text(ZQ_GX + GRID_TOTAL_W/2, ZQ_GY + GRID_TOTAL_H + 0.6,
            'z_q(x) : 量化特征图\n(Quantized Feature Map)',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#2ECC71',
            linespacing=1.3)

    # Colored cells based on GRID_MAP
    zq_colors = [[CB_COLORS[GRID_MAP[r][c]] for c in range(GRID_COLS)]
                  for r in range(GRID_ROWS)]
    zq_labels = [[str(GRID_MAP[r][c]) for c in range(GRID_COLS)]
                  for r in range(GRID_ROWS)]
    draw_grid(ax, ZQ_GX, ZQ_GY, CELL_W, CELL_H, GRID_ROWS, GRID_COLS,
              zq_colors, '#333333', labels=zq_labels, fontsize=12,
              label_color='white')

    # Highlight cell (2,1) — 1-based: row=2, col=1 → 0-based: row=1, col=0
    hl_r, hl_c = 1, 0  # 0-indexed
    hl_x = ZQ_GX + hl_c * CELL_W
    hl_y = ZQ_GY + (GRID_ROWS - 1 - hl_r) * CELL_H
    hl_ring = FancyBboxPatch((hl_x - 0.05, hl_y - 0.05), CELL_W + 0.10, CELL_H + 0.10,
                              boxstyle="round,pad=0.02", facecolor='none',
                              edgecolor='#111111', linewidth=3.0, zorder=6)
    ax.add_patch(hl_ring)

    # Annotation for cell (2,1)
    ax.annotate('z_q(2,1) = e_2',
                xy=(hl_x + CELL_W/2, hl_y + CELL_H/2),
                xytext=(hl_x + CELL_W/2 - 0.8, hl_y + CELL_H/2 + 1.5),
                fontsize=11, color='#2ECC71', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=2.0),
                ha='center', va='center', zorder=7)

    # Axis labels
    ax.text(ZQ_GX + GRID_TOTAL_W/2, ZQ_GY - 0.35,
            '空间位置 (Spatial Position j) →', ha='center', va='center',
            fontsize=11, color=C_GRAY, fontstyle='italic')
    ax.text(ZQ_GX - 0.5, ZQ_GY + GRID_TOTAL_H/2,
            '← (i)', ha='center', va='center', rotation=90,
            fontsize=11, color=C_GRAY, fontstyle='italic')

    # Arrow: Selected → z_q grid
    arrow(ax, SEL_X + SEL_W, SEL_Y + SEL_H/2,
          ZQ_GX - 0.05, ZQ_GY + GRID_TOTAL_H/2, C_GRAY, 2.0)

    # Arrow: VQ Logic → Selected
    arrow(ax, VQ_X + VQ_W, VQ_Y + VQ_H/2,
          SEL_X - 0.05, SEL_Y + SEL_H/2, C_GRAY, 1.8)

    # ================================================================
    # BOTTOM: Decoder (G) → Reconstructed Image
    # (moved up to avoid circle clipping below y=0)
    # ================================================================
    DEC_X, DEC_Y = 12.1, 2.2
    DEC_W, DEC_H = 3.6, 0.9

    rounded_box(ax, DEC_X, DEC_Y, DEC_W, DEC_H,
                text='解码器 (Decoder G)',
                fontsize=14, text_color=C_DARK,
                facecolor='#f0f0f0', edgecolor='#333333')

    # Reconstructed Image (circle) — center at y=1.2, radius 0.6 → bottom at y=0.6, safe!
    CIRC_CX, CIRC_CY = 13.9, 1.2
    CIRC_R = 0.6
    circ = Circle((CIRC_CX, CIRC_CY), CIRC_R, facecolor='#E8E8E8',
                  edgecolor='#333333', linewidth=2.5, zorder=3)
    ax.add_patch(circ)
    ax.text(CIRC_CX, CIRC_CY, r'$\mathbf{\hat{x}}$',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=C_DARK, zorder=5)
    ax.text(CIRC_CX, CIRC_CY - 0.95, '重建图像\n(Reconstructed Image)',
            ha='center', va='top', fontsize=11, color=C_GRAY, fontstyle='italic',
            linespacing=1.3)

    # Arrow: z_q grid → Decoder
    arrow(ax, ZQ_GX + GRID_TOTAL_W/2, ZQ_GY - 0.08,
          DEC_X + DEC_W/2, DEC_Y + DEC_H + 0.08, C_GRAY, 2.0)

    # Arrow: Decoder → Reconstructed Image
    arrow(ax, DEC_X + DEC_W/2, DEC_Y - 0.05,
          CIRC_CX, CIRC_CY + CIRC_R + 0.05, C_GRAY, 2.0)

    # ================================================================
    # LEGEND: color-codebook mapping (bottom-left)
    # ================================================================
    LG_X, LG_Y = 1.0, 0.8
    ax.text(LG_X, LG_Y + 1.8, '码本索引 → 颜色映射 (Codebook Index → Color Mapping)',
            fontsize=11, fontweight='bold', color=C_DARK)
    for i in range(K):
        idx = i + 1
        dx = i * 2.2
        sw = Rectangle((LG_X + dx, LG_Y), 0.4, 0.35,
                       facecolor=CB_COLORS[idx], edgecolor='#333333',
                       linewidth=1.0, zorder=3)
        ax.add_patch(sw)
        ax.text(LG_X + dx + 0.55, LG_Y + 0.18,
                f'e_{idx}  ({idx})',
                fontsize=11, va='center', color=C_DARK)

    # ===== Save =====
    plt.tight_layout(pad=0.5)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[DONE] Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
