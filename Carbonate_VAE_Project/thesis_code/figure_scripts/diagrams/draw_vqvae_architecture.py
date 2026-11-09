
from thesis_code.config import DIAGRAM_DIR
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体，防止中文显示为方块 (支持 Windows 和 Mac)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(18, 8), dpi=200)
ax.set_xlim(-1, 23)
ax.set_ylim(-4, 5)
ax.axis('off')

# ================= 辅助函数：绘制圆角矩形 =================
def draw_box(center_x, center_y, width, height, text, facecolor, edgecolor, 
             linestyle='-', text_color='black', fontsize=12, fontweight='normal', zorder=1):
    rect = patches.FancyBboxPatch(
        (center_x - width/2, center_y - height/2), width, height,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
        linestyle=linestyle, zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(center_x, center_y, text, ha='center', va='center', 
            color=text_color, fontsize=fontsize, fontweight=fontweight, zorder=zorder+1)
    return center_x - width/2, center_x + width/2, center_y + height/2, center_y - height/2

# ================= 1. 绘制主体流水线 (水平方向) =================
y_main = 0
boxes = {
    'input': {'x': 1, 'w': 2, 'h': 1.5, 'text': '输入岩心\n$x$', 'fc': '#f2f2f2', 'ec': '#666666'},
    'enc':   {'x': 4, 'w': 2, 'h': 1.5, 'text': '编码器\nEncoder', 'fc': '#d9e8f5', 'ec': '#4f81bd'},
    'ze':    {'x': 7, 'w': 2, 'h': 1.5, 'text': '连续特征\n$z_e(x)$', 'fc': '#ebdcf2', 'ec': '#8064a2'},
    'zq':    {'x': 15, 'w': 2, 'h': 1.5, 'text': '离散特征\n$z_q(x)$', 'fc': '#e2f0d9', 'ec': '#76933c'},
    'dec':   {'x': 18, 'w': 2, 'h': 1.5, 'text': '解码器\nDecoder', 'fc': '#e2f0d9', 'ec': '#76933c'},
    'out':   {'x': 21, 'w': 2, 'h': 1.5, 'text': '重构岩心\n$\hat{x}$', 'fc': '#f2f2f2', 'ec': '#666666'}
}

coords = {}
for k, v in boxes.items():
    left, right, top, bottom = draw_box(v['x'], y_main, v['w'], v['h'], v['text'], v['fc'], v['ec'])
    coords[k] = {'left': left, 'right': right, 'top': top, 'bottom': bottom, 'x': v['x']}

# ================= 2. 绘制中间的矢量量化区域 =================
# 外层虚线框
draw_box(11, 0, 5.5, 3, "", facecolor='none', edgecolor='#666666', linestyle='--', zorder=0)
ax.text(11, 1.2, "(1) 最近邻搜索映射", ha='center', va='center', fontsize=12, fontweight='bold', color='#4f81bd')

# 内层码本实线框
codebook_text = "离散码本 $E$\n$\{e_1, e_2, ..., e_K\}$\n\n$\\arg\\min_j ||z_e(x) - e_j||_2$"
draw_box(11, -0.2, 4.5, 1.8, codebook_text, facecolor='#fff9e6', edgecolor='#f7ca18')

# ================= 3. 绘制上方 STE 梯度反传框 =================
ste_text = "(2) 直通估计器 (Straight-Through Estimator, STE) 梯度反向传播通路\n\n$\\partial L / \\partial z_e(x) \\approx \\partial L / \\partial z_q(x)$"
draw_box(11, 3, 10, 1.5, ste_text, facecolor='#fce4d6', edgecolor='#f4b084', text_color='#c00000', fontweight='bold')

# ================= 4. 绘制下方联合损失函数框 =================
loss_text = "(3) 联合损失函数约束\n\n$L_{VQ} = L_{reconstruct}(x, \hat{x}) + ||sg[z_e(x)] - z_q(x)||_2^2 + \\beta ||z_e(x) - sg[z_q(x)]||_2^2$"
draw_box(11, -2.8, 14, 1.2, loss_text, facecolor='#eef8d8', edgecolor='#c4d79b', fontweight='bold')

# ================= 5. 绘制所有箭头 =================
def draw_arrow(x1, y1, x2, y2, color='black', linestyle='-', head_width=0.2):
    ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=head_width, head_length=0.3, fc=color, ec=color, 
             linewidth=1.5, length_includes_head=True, linestyle=linestyle)

# 主干水平箭头
draw_arrow(coords['input']['right'], y_main, coords['enc']['left'], y_main)
draw_arrow(coords['enc']['right'], y_main, coords['ze']['left'], y_main)
draw_arrow(coords['ze']['right'], y_main, 11 - 2.75, y_main) # 到虚线框
draw_arrow(11 + 2.75, y_main, coords['zq']['left'], y_main)
draw_arrow(coords['zq']['right'], y_main, coords['dec']['left'], y_main)
draw_arrow(coords['dec']['right'], y_main, coords['out']['left'], y_main)

# STE 红色虚线梯度箭头 (从 z_q 往上，从上面往 z_e)
# z_q -> STE框
ax.annotate("", xy=(15, 2.25), xytext=(15, 0.75),
            arrowprops=dict(edgecolor='#c00000', facecolor='#c00000', width=1.5, headwidth=8, ls='--', shrink=0))
# STE框 -> z_e
ax.annotate("", xy=(7, 0.75), xytext=(7, 2.25),
            arrowprops=dict(edgecolor='#c00000', facecolor='#c00000', width=1.5, headwidth=8, ls='--', shrink=0))

# ================= 6. 保存高质量图片 =================
plt.tight_layout()

# 保存为 600 DPI 的超清 PNG
plt.savefig(str(DIAGRAM_DIR / 'VQ_VAE_Architecture_HD.png'), dpi=600, bbox_inches='tight')
# 保存为 PDF 矢量图
plt.savefig(str(DIAGRAM_DIR / 'VQ_VAE_Architecture_Vector.pdf'), bbox_inches='tight')
