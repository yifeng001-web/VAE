
from thesis_code.config import DIAGRAM_DIR
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体，防止中文显示方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(10, 14), dpi=200)
ax.set_xlim(-6, 6)
ax.set_ylim(-21, 3)
ax.axis('off')

# ================= 辅助函数库 =================
def draw_box(x, y, w, h, text, fc, ec, text_color='black', fontsize=12, fontweight='normal', ls='-', zorder=1):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        facecolor=fc, edgecolor=ec, linewidth=1.5, linestyle=ls, zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', color=text_color,
            fontsize=fontsize, fontweight=fontweight, zorder=zorder+1)
    return (x, y+h/2), (x, y-h/2), (x-w/2, y), (x+w/2, y)

def draw_diamond(x, y, w, h, text, fc, ec, text_color='black', fontsize=12, fontweight='normal', zorder=1):
    pts = [[x, y+h/2], [x-w/2, y], [x, y-h/2], [x+w/2, y]]
    poly = patches.Polygon(pts, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=zorder)
    ax.add_patch(poly)
    ax.text(x, y, text, ha='center', va='center', color=text_color,
            fontsize=fontsize, fontweight=fontweight, zorder=zorder+1)
    return (x, y+h/2), (x, y-h/2), (x-w/2, y), (x+w/2, y)

def draw_arrow(pt1, pt2, color='black', lw=1.5):
    ax.annotate("", xy=pt2, xytext=pt1,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0))

# ================= 1. 绘制背景虚线分组框 =================
# 物理样本获取层
draw_box(0, 1, 9.5, 3.5, "", fc='#f4f8fe', ec='#4a86e8', ls='--', zorder=0)
ax.text(0, 2.2, "物理样本获取层 (Physical Sample Acquisition)", ha='center', va='center', fontsize=13, fontweight='bold', color='#29539b', zorder=1)

# 自动化数据流水线层
draw_box(0, -11, 7, 9, "", fc='#f7fbf5', ec='#6aa84f', ls='--', zorder=0)
ax.text(0, -7, "自动化数据流水线 (Automated Pipeline)", ha='center', va='center', fontsize=13, fontweight='bold', color='#274e13', zorder=1)

# ================= 2. 绘制所有节点模块 =================
# --- 第一层：样本 ---
_, b_a, _, _ = draw_box(-3, 0.5, 2.6, 1.2, "碳酸盐岩心样本 A", fc='#fce5cd', ec='#a64d79', text_color='#741b47', fontweight='bold')
_, b_b, _, _ = draw_box(0, 0.5, 2.6, 1.2, "碳酸盐岩心样本 B", fc='#fce5cd', ec='#a64d79', text_color='#741b47', fontweight='bold')
_, b_c, _, _ = draw_box(3, 0.5, 2.6, 1.2, "碳酸盐岩心样本 C", fc='#fce5cd', ec='#a64d79', text_color='#741b47', fontweight='bold')

# --- 第二层：Micro-CT ---
t_ct, b_ct, _, _ = draw_box(0, -2.5, 4.5, 1.4, "Micro-CT 高分辨率扫描\n$\\mathit{45 / 50 \\mu m}$", fc='#cfe2f3', ec='#3d85c6', fontweight='bold')

# --- 第三层：FBP重建 ---
t_fbp, b_fbp, _, _ = draw_box(0, -5, 4.5, 1.2, "FBP 算法三维重建", fc='#cfe2f3', ec='#3d85c6', fontweight='bold')

# --- 第四层：流水线处理 ---
t_store, b_store, _, _ = draw_box(0, -9, 4.5, 1.4, "多级嵌套目录存储\n共 4142 张二维 TIFF", fc='#d9ead3', ec='#38761d', fontweight='bold')
t_os, b_os, _, _ = draw_box(0, -11.5, 4.5, 1.2, "OS 底层递归遍历检索", fc='#d9ead3', ec='#38761d', fontweight='bold')

# --- 第五层：判断菱形 ---
t_dia, b_dia, l_dia, r_dia = draw_diamond(0, -14.5, 5.5, 2.2, "基数核验屏障\nValidation Barrier", fc='#fff2cc', ec='#f1c232', fontweight='bold')

# --- 第六层：条件阈值 ---
t_pass, b_pass, _, _ = draw_box(-2.8, -17.5, 1.8, 0.8, "≥ 4000", fc='#d9ead3', ec='#38761d', text_color='#274e13', fontweight='bold')
t_fail, b_fail, _, _ = draw_box(2.8, -17.5, 2.0, 0.8, "数量异常", fc='#f4cccc', ec='#cc0000', text_color='#990000', fontweight='bold')

# --- 第七层：终点 ---
t_gpu, _, _, _ = draw_box(-2.8, -19.5, 4.0, 1.4, "GPU 计算显存队列\n进入 VQ-VAE 特征学习", fc='#38761d', ec='#274e13', text_color='white', fontweight='bold')
t_stop, _, _, _ = draw_box(2.8, -19.5, 2.8, 1.2, "终止并报错", fc='#eeeeee', ec='#999999', fontweight='bold')

# ================= 3. 绘制逻辑连线 =================
# 顶部三路汇聚
mid_y_samples = -1.0
ax.plot([-3, -3], [b_a[1], mid_y_samples], color='black', lw=1.5)
ax.plot([3, 3], [b_c[1], mid_y_samples], color='black', lw=1.5)
ax.plot([-3, 3], [mid_y_samples, mid_y_samples], color='black', lw=1.5)
ax.plot([0, 0], [b_b[1], mid_y_samples], color='black', lw=1.5)
draw_arrow((0, mid_y_samples), t_ct)

# 中轴直线
draw_arrow(b_ct, t_fbp)
draw_arrow(b_fbp, t_store)
draw_arrow(b_store, t_os)
draw_arrow(b_os, t_dia)

# 左侧分支 (成功通路 - 绿色)
ax.plot([l_dia[0], -2.8], [l_dia[1], l_dia[1]], color='#38761d', lw=1.5)
draw_arrow((-2.8, l_dia[1]), t_pass, color='#38761d', lw=2)
draw_arrow(b_pass, t_gpu, color='#38761d', lw=2)

# 右侧分支 (失败通路 - 红色)
ax.plot([r_dia[0], 2.8], [r_dia[1], r_dia[1]], color='#cc0000', lw=1.5)
draw_arrow((2.8, r_dia[1]), t_fail, color='#cc0000', lw=2)
draw_arrow(b_fail, t_stop, color='#cc0000', lw=2)

# ================= 4. 输出与保存 =================
plt.tight_layout()
plt.savefig(str(DIAGRAM_DIR / 'Figure_3_1_Architecture_HD.png'), dpi=600, bbox_inches='tight')
plt.savefig(str(DIAGRAM_DIR / 'Figure_3_1_Architecture_Vector.pdf'), bbox_inches='tight')
print("高清流程图生成完毕！")
