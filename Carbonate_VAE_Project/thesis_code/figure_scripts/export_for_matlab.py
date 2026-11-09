"""
export_for_matlab.py — 导出原始像素和拉伸后像素数据为 .mat 文件，供 MATLAB 画直方图
"""
import numpy as np
import tifffile
import os
import scipy.io as sio

from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
image_path = os.path.join(DATA_DIR, "1633-1663-1203-50um0000.tif")

if not os.path.exists(image_path):
    image_path = os.path.join(PROJECT_ROOT, "data", "raw", "CT", "8-gs101-2",
                              "volume tif-1813-1854-1469-45um",
                              "1813-1854-1469-45um0203.tif")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAT_PATH = os.path.join(OUTPUT_DIR, "figure_2_2_hist_data.mat")

# 读取图像
img_raw = tifffile.imread(image_path)
img_float = img_raw.astype(np.float32)
print(f"[INFO] 图片读取成功: {image_path}")
print(f"       dtype={img_raw.dtype}, shape={img_raw.shape}, "
      f"min={img_raw.min()}, max={img_raw.max()}, mean={img_raw.mean():.1f}")

# 自适应对比度拉伸
lower_percent = 2.0
upper_percent = 98.0
p_low = np.percentile(img_float, lower_percent)
p_high = np.percentile(img_float, upper_percent)
img_clipped = np.clip(img_float, p_low, p_high)
epsilon = 1e-8
img_stretched = (img_clipped - p_low) / (p_high - p_low + epsilon)

print(f"[INFO] P2={p_low:.1f}, P98={p_high:.1f}")
print(f"[INFO] 拉伸后范围: [{img_stretched.min():.6f}, {img_stretched.max():.6f}]")

# 保存为 .mat (MATLAB v7.3 格式以支持大数组)
sio.savemat(MAT_PATH, {
    'img_float': img_float,
    'img_stretched': img_stretched,
    'p_low': p_low,
    'p_high': p_high,
    'img_min': float(img_float.min()),
    'img_max': float(img_float.max()),
    'lower_percent': lower_percent,
    'upper_percent': upper_percent,
})
print(f"[DONE] 数据已导出至: {MAT_PATH}")
print(f"       变量: img_float ({img_float.shape}, {img_float.dtype})")
print(f"       变量: img_stretched ({img_stretched.shape}, {img_stretched.dtype})")
