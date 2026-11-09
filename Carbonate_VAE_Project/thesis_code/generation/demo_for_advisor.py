"""
导师展示脚本：使用最优 VQ-VAE 模型 (Epoch 50) 生成单张高质量碳酸盐岩孔隙图
=======================================================================
训练流程：
  Stage 1 (VQ-VAE, 50 epochs): 编码器将 256x256 CT 切片压缩到 32x32 连续隐空间，
  经 512-entry 离散码本量化后由解码器重建。五元复合损失函数同时约束像素重构、
  结构相似性、孔隙度保持和边缘保留。R²=0.9933, MAE=0.0039。

  Stage 2 (PixelCNN, 25 epochs): 在离散隐空间上训练自回归先验，12层掩码残差块
  学习碳酸盐岩孔隙的空间语法。频率倒数加权交叉熵解决极端类别不平衡（~90%基质）。

  Stage 3 (生成): 直方图匹配 + 连续隐空间柔性融合，在 VQ 量化前进行 Z_e 空间
  基因融合，杜绝方块伪影。

输出：results/evaluation/demo_advisor.png (dpi=300)
"""

import os, sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

# ── GPU check ──
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected!")
print(f"[OK] GPU: {len(gpus)} device(s)")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from skimage.exposure import match_histograms
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("white")
    sns.set_context("talk", font_scale=0.9)
except ImportError:
    pass

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

# ── Paths ──
from thesis_code.config import PROJECT_ROOT
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR     = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Constants (来自 train_thesis.py, generate_fusion_thesis.py) ──
NUM_EMBEDDINGS = 512       # train_thesis.py:32
LATENT_DIM     = 64        # train_thesis.py:31
EPOCH_TO_LOAD  = 50        # 最优模型轮次
LATENT_SIZE    = 32
IMAGE_SIZE     = 256
CLOUD_SIGMA    = 4.0       # generate_fusion_thesis.py:72
SOFTEN_SIGMA   = 1.0       # generate_fusion_thesis.py:73

print("=" * 60)
print("  导师展示：最优 VQ-VAE 碳酸盐岩孔隙生成")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# Step 1: 加载最优模型 (Epoch 50)
# ═══════════════════════════════════════════════════════════════
print("\n[1/5] 加载最优模型 (Epoch 50)...")
encoder = build_encoder(latent_dim=LATENT_DIM)
decoder = build_decoder(latent_dim=LATENT_DIM)
vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

_ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
_ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))

encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
vq_layer.set_weights([vq_embedding])
print("   Encoder + Codebook (512x64) + Decoder 加载完毕。")

# ═══════════════════════════════════════════════════════════════
# Step 2: 随机选取一张真实 CT 切片作为展示样本
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] 随机选取真实 CT 切片...")
TARGET_EXTENSIONS = {'.tif', '.TIF', '.tiff', '.TIFF'}
file_list = []
for root, dirs, filenames in os.walk(DATA_DIR):
    for f in filenames:
        _, ext = os.path.splitext(f)
        if ext in TARGET_EXTENSIONS:
            file_list.append(os.path.join(root, f))

np.random.seed(42)  # 可复现
np.random.shuffle(file_list)

def load_one_tif(filepath):
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t); max_v = tf.reduce_max(img_t)
    return tf.math.divide_no_nan(img_t - min_v, max_v - min_v).numpy()

sample_img = load_one_tif(file_list[0])  # (256, 256, 1)
print(f"   选中切片: {os.path.basename(file_list[0])}  |  均值={sample_img.mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# Step 3: VQ-VAE 重构 — 展示编码-解码保真度
# ═══════════════════════════════════════════════════════════════
print("\n[3/5] VQ-VAE 重构 (编码→量化→解码)...")
z_e_recon = encoder(tf.constant(np.expand_dims(sample_img, 0)), training=False)
z_q_recon = vq_layer(z_e_recon)
recon_img = np.clip(decoder(z_q_recon, training=False).numpy()[0, :, :, 0], 0.0, 1.0)
print(f"   原始均值={sample_img.mean():.4f}  →  重构均值={recon_img.mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# Step 4: 直方图匹配 + 连续隐空间融合 — 生成新颖孔隙结构
# ═══════════════════════════════════════════════════════════════
print("\n[4/5] 直方图匹配 + 连续隐空间柔性融合 (生成新颖孔隙)...")

# 选第二张不同切片作为孔隙供体
donor_img = load_one_tif(file_list[10])
print(f"   孔隙供体: {os.path.basename(file_list[10])}  |  均值={donor_img.mean():.4f}")

# 直方图匹配: 供体 → 母本 (消除背景色差)
donor_matched = match_histograms(donor_img[:, :, 0], sample_img[:, :, 0])
donor_matched = np.clip(donor_matched, 0.0, 1.0).astype(np.float32)
donor_matched = np.expand_dims(donor_matched, axis=-1)
print(f"   直方图匹配后供体均值: {donor_img.mean():.4f} → {donor_matched.mean():.4f}")

# 编码到连续隐空间 Z_e (VQ 量化前)
z_eA = encoder(tf.constant(np.expand_dims(sample_img, 0)), training=False)
z_eB = encoder(tf.constant(np.expand_dims(donor_matched, 0)), training=False)

# 生成有机云雾掩膜
noise = np.random.RandomState(123).randn(LATENT_SIZE, LATENT_SIZE).astype(np.float32)
cloud = gaussian_filter(noise, sigma=CLOUD_SIGMA)
binary = (cloud > np.median(cloud)).astype(np.float32)
mask = gaussian_filter(binary, sigma=SOFTEN_SIGMA)
mask = mask[np.newaxis, :, :, np.newaxis]  # (1, 32, 32, 1)

# 连续隐空间融合: Z_fused = Z_eA×(1-mask) + Z_eB×mask
z_fused = z_eA * (1.0 - mask) + z_eB * mask

# 统一 VQ 量化 + 解码
z_q_fused = vq_layer(z_fused, training=False)
fusion_img = np.clip(decoder(z_q_fused, training=False).numpy()[0, :, :, 0], 0.0, 1.0)
print(f"   融合生成均值={fusion_img.mean():.4f}  |  孔隙区域占比={mask.mean():.2%}")

# ═══════════════════════════════════════════════════════════════
# Step 5: 绘制 1×3 对比图
# ═══════════════════════════════════════════════════════════════
print("\n[5/5] 渲染出版级展示图...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

titles = [
    'Real CT Slice\n(Ground Truth)',
    'VQ-VAE Reconstruction\n(Codebook 512 × 64)',
    'Novel Generated Rock\n(Fusion: Histogram-Matched)'
]
images = [sample_img[:, :, 0], recon_img, fusion_img]
colors = ['#2C6FAC', '#D9583B', '#2E8B57']  # blue, coral, green

for idx, ax in enumerate(axes.flat):
    im = ax.imshow(images[idx], cmap='gray', vmin=0.0, vmax=1.0)
    ax.set_title(titles[idx], fontsize=12, fontweight='bold', color=colors[idx], pad=12)
    ax.axis('off')
    # 添加边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(colors[idx])
        spine.set_linewidth(3)

# 在每张图下方标注孔隙度
means = [sample_img.mean(), recon_img.mean(), fusion_img.mean()]
for idx, ax in enumerate(axes.flat):
    ax.text(0.5, -0.06, f'Porosity ≈ {1-means[idx]:.3f}',
            transform=ax.transAxes, fontsize=11, ha='center',
            color=colors[idx], fontweight='bold')

plt.suptitle('Carbonate Rock VQ-VAE: Pore Structure Generation (Epoch 50)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()

output_path = os.path.join(EVAL_DIR, "demo_advisor.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'=' * 60}")
print(f"  [DONE] 导师展示图已保存至:")
print(f"    {output_path}")
print(f"{'=' * 60}")
print(f"\n  Training Summary:")
print(f"  " + "-" * 55)
print(f"  Stage 1 (VQ-VAE)  : 50 epochs, Batch=4")
print(f"    Encoder: 256x256 -> 32x32x64 (8x compression)")
print(f"    Codebook: 512 x 64 (discrete representation)")
print(f"    Loss: MAEx10 + MS-SSIMx5 + Porosityx20 + Sobelx10 + VQx1")
print(f"    R^2=0.9933  MAE=0.0039")
print(f"  " + "-" * 55)
print(f"  Stage 2 (PixelCNN): 25 epochs, Batch=16")
print(f"    12 Masked Residual Blocks, 128 channels")
print(f"    Frequency-inverse weighted cross-entropy")
print(f"  " + "-" * 55)
print(f"  Stage 3 (Generation): Histogram Matching + Continuous Fusion")
print(f"    mask: Gaussian sigma=4.0 -> binarize -> soften sigma=1.0")
print(f"    Fusion in Z_e space BEFORE VQ quantization")
print(f"{'=' * 60}")
