"""
[第五章 实验结果分析] 出版级全套评估图表生成脚本
====================================================
生成 5 张核心图表：
  图 5-1: loss_curves.png          — 从 training_history.npz 读取的训练曲线
  图 5-2: reconstruction_comparison.png — 高保真重构对比 (2×8)
  图 5-3: porosity_analysis.png    — 孔隙度联合定量评估
  图 5-4: s2_curves.png            — 两点概率函数 S2 曲线
  图 5-5: random_batch_rocks.png   — 批量随机密度生成 (单独运行)

所有图表 dpi=300, 学术配色, seaborn 风格。
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image

# ===== FIREWALL: GPU check =====
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected!")
print(f"[FIREWALL] GPU check PASSED: {len(gpus)} GPU(s) available")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Try seaborn for academic styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("[WARN] seaborn not installed, using default matplotlib style")

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer
from thesis_code.utils.data_loader import create_dataset

# ===== Paths =====
from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# ===== Constants =====
NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
LATENT_SIZE = 32
IMAGE_SIZE = 256
BATCH_SIZE = 8

# Academic color palette
COLOR_REAL = '#2C6FAC'       # steel blue
COLOR_RECON = '#D9583B'      # coral/rust
COLOR_LOSS = '#2C3E50'       # dark slate
COLOR_MAE = '#2980B9'        # blue
COLOR_POROSITY = '#8E44AD'   # purple
COLOR_SOBEL = '#E67E22'      # orange
COLOR_VQ = '#95A5A6'         # gray

# ============================================================
# 通用：加载模型
# ============================================================
def load_models():
    """加载 Encoder, VQ codebook, Decoder (Epoch 50)."""
    print("[INFO] Loading models (Epoch 50)...")
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    # Init graphs
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])
    print("[INFO] All models loaded.\n")
    return encoder, decoder, vq_layer


# ============================================================
# 任务 1：训练收敛曲线 (图 5-1)
# ============================================================
def generate_loss_curves():
    """从保存的真实训练记录绘图；子损失使用未加权的原始值。"""
    history_path = os.path.join(PROJECT_ROOT, "results", "training_history.npz")
    metrics = [
        ("loss", "Total Training Loss", COLOR_LOSS),
        ("mae", "MAE Reconstruction Loss", COLOR_MAE),
        ("porosity", "Mean Grayscale Difference", COLOR_POROSITY),
        ("edge", "Sobel Edge Loss", COLOR_SOBEL),
    ]
    with np.load(history_path, allow_pickle=False) as history:
        curves = [np.asarray(history[key]) for key, _, _ in metrics]
    num_epochs = curves[0].size
    if num_epochs == 0 or any(
        curve.ndim != 1 or curve.size != num_epochs or not np.isfinite(curve).all()
        for curve in curves
    ):
        raise ValueError("Training history must contain finite, equal-length 1-D curves.")
    epochs = np.arange(1, num_epochs + 1)
    print(f"[DATA] Loaded {num_epochs} training epochs from: {history_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (_, title, color), curve in zip(axes.flat, metrics, curves):
        ax.plot(epochs, curve, color=color, linewidth=2.0, label=title)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'VQ-VAE Recorded Training Curves ({num_epochs} Epochs)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(EVAL_DIR, "loss_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Loss curves saved to: {output_path}\n")


# ============================================================
# 任务 2：高保真视觉重构对比图 (图 5-2)
# ============================================================
def generate_reconstruction_comparison(encoder, decoder, vq_layer):
    """2×8 grid: Row 1 = Ground Truth, Row 2 = Reconstruction."""
    print("=" * 60)
    print("  Task 2: Reconstruction Comparison (reconstruction_comparison.png)")
    print("=" * 60)

    # Collect 8 test images
    print("[INFO] Collecting 8 test images...")
    dataset, _ = create_dataset(DATA_DIR, batch_size=8)
    test_images, _ = next(iter(dataset))

    # VQ-VAE reconstruction
    z_e = encoder(test_images, training=False)
    quantized_z = vq_layer(z_e)
    reconstructed = decoder(quantized_z, training=False)
    reconstructed = tf.clip_by_value(reconstructed, 0.0, 1.0)

    # Plot 2×8 grid
    fig, axes = plt.subplots(2, 8, figsize=(26, 7))

    for i in range(8):
        # Row 0: Ground Truth
        axes[0, i].imshow(test_images[i, :, :, 0].numpy(), cmap='gray', vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f'GT #{i+1}', fontsize=10, fontweight='bold', color=COLOR_REAL)
        axes[0, i].axis('off')

        # Row 1: Reconstruction
        axes[1, i].imshow(reconstructed[i, :, :, 0].numpy(), cmap='gray', vmin=0.0, vmax=1.0)
        axes[1, i].set_title(f'Recon #{i+1}', fontsize=10, fontweight='bold', color=COLOR_RECON)
        axes[1, i].axis('off')

    # Row labels
    fig.text(0.02, 0.75, 'Ground\nTruth', fontsize=13, fontweight='bold',
             color=COLOR_REAL, ha='center', va='center', rotation=90)
    fig.text(0.02, 0.25, 'Reconstruction', fontsize=13, fontweight='bold',
             color=COLOR_RECON, ha='center', va='center', rotation=90)

    plt.suptitle('VQ-VAE High-Fidelity Reconstruction Comparison (Epoch 50)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])

    output_path = os.path.join(EVAL_DIR, "reconstruction_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Reconstruction comparison saved to: {output_path}\n")


# ============================================================
# 任务 3：孔隙度联合定量评估图 (图 5-3)
# ============================================================
def generate_porosity_analysis(encoder, decoder, vq_layer):
    """Scatter plot + overlapping histogram for porosity evaluation."""
    print("=" * 60)
    print("  Task 3: Porosity Analysis (porosity_analysis.png)")
    print("=" * 60)

    NUM_SAMPLES = 100

    # Collect real images
    print(f"[INFO] Collecting {NUM_SAMPLES} real CT slices...")
    dataset, _ = create_dataset(DATA_DIR, batch_size=4)

    real_images_list = []
    count = 0
    for batch_images, _ in dataset:
        real_images_list.append(batch_images.numpy())
        count += batch_images.shape[0]
        if count >= NUM_SAMPLES:
            break

    real_images = np.concatenate(real_images_list, axis=0)[:NUM_SAMPLES]
    print(f"[INFO] Collected {real_images.shape[0]} images.")

    # VQ-VAE reconstruction
    print("[INFO] Running VQ-VAE reconstruction...")
    real_tensor = tf.convert_to_tensor(real_images, dtype=tf.float32)
    z_e = encoder(real_tensor, training=False)
    quantized_z = vq_layer(z_e)
    reconstructed_tensor = decoder(quantized_z, training=False)
    reconstructed = np.clip(reconstructed_tensor.numpy(), 0.0, 1.0)

    # Compute porosity (pixel mean)
    real_porosity = np.mean(real_images, axis=(1, 2, 3))
    recon_porosity = np.mean(reconstructed, axis=(1, 2, 3))

    # Compute R^2 and MAE
    mae = np.mean(np.abs(real_porosity - recon_porosity))
    ss_res = np.sum((real_porosity - recon_porosity) ** 2)
    ss_tot = np.sum((real_porosity - np.mean(real_porosity)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"[RESULT] MAE = {mae:.6f}")
    print(f"[RESULT] R^2  = {r2:.6f}")
    print(f"[RESULT] Mean Real Porosity = {real_porosity.mean():.6f}")
    print(f"[RESULT] Mean Recon Porosity = {recon_porosity.mean():.6f}")

    # ===== Plot 1×2 =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Left: Scatter plot ----
    ax1.scatter(real_porosity, recon_porosity, alpha=0.55, s=35,
                c=COLOR_REAL, edgecolors='white', linewidth=0.4, zorder=3)

    # y=x reference line
    rng = [min(real_porosity.min(), recon_porosity.min()) - 0.01,
           max(real_porosity.max(), recon_porosity.max()) + 0.01]
    ax1.plot(rng, rng, '--', color='crimson', linewidth=1.8, label='y = x (Ideal)', zorder=2)

    # Linear fit
    slope, intercept = np.polyfit(real_porosity, recon_porosity, 1)
    x_fit = np.linspace(rng[0], rng[1], 100)
    y_fit = slope * x_fit + intercept
    ax1.plot(x_fit, y_fit, '-', color=COLOR_RECON, linewidth=1.8,
             label=f'Fit: y = {slope:.3f}x + {intercept:.4f}', zorder=2)

    # Annotate R^2 and MAE prominently
    ax1.text(0.05, 0.92, f'$R^2$ = {r2:.4f}\nMAE = {mae:.4f}',
             transform=ax1.transAxes, fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='gray', alpha=0.9),
             verticalalignment='top')

    ax1.set_xlabel('Real Porosity (pixel mean)', fontsize=12)
    ax1.set_ylabel('Reconstructed Porosity (pixel mean)', fontsize=12)
    ax1.set_title(f'Porosity Cross-Plot (N={NUM_SAMPLES})',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(rng)
    ax1.set_ylim(rng)
    ax1.set_aspect('equal')

    # ---- Right: Overlapping histogram ----
    bins = 30
    ax2.hist(real_porosity, bins=bins, alpha=0.55, color=COLOR_REAL,
             edgecolor='white', linewidth=0.5, label='Real Porosity', density=True)
    ax2.hist(recon_porosity, bins=bins, alpha=0.55, color=COLOR_RECON,
             edgecolor='white', linewidth=0.5, label='Reconstructed Porosity', density=True)

    # KDE curves if seaborn available
    if HAS_SEABORN:
        from scipy.stats import gaussian_kde
        for data, color, label in [(real_porosity, COLOR_REAL, 'Real'),
                                    (recon_porosity, COLOR_RECON, 'Recon')]:
            kde = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            ax2.plot(xs, kde(xs), color=color, linewidth=2.0, alpha=0.8)

    ax2.set_xlabel('Porosity (pixel mean)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Porosity Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.25, axis='y')

    plt.suptitle('VQ-VAE Physical Porosity Preservation Analysis (Epoch 50)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(EVAL_DIR, "porosity_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Porosity analysis saved to: {output_path}\n")

    return r2, mae


# ============================================================
# 任务 4：两点概率函数 S2 曲线 (图 5-4)
# ============================================================
def compute_s2_fft(image_binary, max_r=None):
    """
    Compute two-point probability function S2(r) via FFT.

    S2(r) = P[I(x)=1 AND I(x+r)=1]
    For a stationary, isotropic 2D binary field, S2 can be computed as:
        S2 = F^{-1}(|F(I)|^2) / N,  then radially average.

    Args:
        image_binary: (H, W) binary array, 1=pore, 0=matrix
        max_r: maximum distance to compute (default: min(H,W)//2)
    Returns:
        r_values: (max_r,) distances in pixels
        s2_values: (max_r,) S2 probability values
    """
    H, W = image_binary.shape
    if max_r is None:
        max_r = min(H, W) // 2

    # FFT-based autocorrelation
    f = np.fft.fft2(image_binary)
    power_spectrum = np.abs(f) ** 2
    autocorr = np.fft.fftshift(np.fft.ifft2(power_spectrum).real)

    # Normalize: S2(0) should equal porosity
    autocorr = autocorr / (H * W)

    # Radially average
    center_h, center_w = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r_map = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)

    s2_values = np.zeros(max_r)
    r_values = np.arange(max_r)

    for r in range(max_r):
        mask = (r_map >= r - 0.5) & (r_map < r + 0.5)
        if mask.sum() > 0:
            s2_values[r] = autocorr[mask].mean()
        else:
            s2_values[r] = s2_values[r - 1] if r > 0 else s2_values[0]

    return r_values, s2_values


def generate_s2_curves(encoder, decoder, vq_layer):
    """Compute and plot S2 curves for real vs. generated rock slices."""
    print("=" * 60)
    print("  Task 4: Two-Point Probability S2 Curves (s2_curves.png)")
    print("=" * 60)

    N_SAMPLES = 10
    MAX_R = 100  # pixels

    # Collect real images
    print(f"[INFO] Collecting {N_SAMPLES} real images for S2 analysis...")
    dataset, _ = create_dataset(DATA_DIR, batch_size=4)

    real_images_list = []
    count = 0
    for batch_images, _ in dataset:
        real_images_list.append(batch_images.numpy())
        count += batch_images.shape[0]
        if count >= N_SAMPLES:
            break

    real_images = np.concatenate(real_images_list, axis=0)[:N_SAMPLES]
    print(f"[INFO] Collected {real_images.shape[0]} real images.")

    # Reconstruct
    print("[INFO] Running VQ-VAE reconstruction...")
    real_tensor = tf.convert_to_tensor(real_images, dtype=tf.float32)
    z_e = encoder(real_tensor, training=False)
    quantized_z = vq_layer(z_e)
    reconstructed_tensor = decoder(quantized_z, training=False)
    reconstructed = np.clip(reconstructed_tensor.numpy(), 0.0, 1.0)

    # Compute S2 for each image
    print(f"[INFO] Computing S2(r) for {N_SAMPLES} image pairs (r=0..{MAX_R-1})...")

    real_s2_all = np.zeros((N_SAMPLES, MAX_R))
    recon_s2_all = np.zeros((N_SAMPLES, MAX_R))

    for i in range(N_SAMPLES):
        if (i + 1) % 5 == 0:
            print(f"  Processing {i+1}/{N_SAMPLES}...")

        # Binary threshold at mean (standard approach for carbonate rocks)
        img_real = real_images[i, :, :, 0]
        img_recon = reconstructed[i, :, :, 0]

        thresh_real = np.mean(img_real)
        thresh_recon = np.mean(img_recon)

        bin_real = (img_real < thresh_real).astype(np.float32)      # pore=1
        bin_recon = (img_recon < thresh_recon).astype(np.float32)   # pore=1

        _, real_s2_all[i] = compute_s2_fft(bin_real, MAX_R)
        _, recon_s2_all[i] = compute_s2_fft(bin_recon, MAX_R)

    # Average across samples
    real_s2_mean = real_s2_all.mean(axis=0)
    real_s2_std = real_s2_all.std(axis=0)
    recon_s2_mean = recon_s2_all.mean(axis=0)
    recon_s2_std = recon_s2_all.std(axis=0)

    r_values = np.arange(MAX_R)

    # Compute correlation coefficient between mean curves
    corr = np.corrcoef(real_s2_mean, recon_s2_mean)[0, 1]

    print(f"[RESULT] S2 correlation (real vs recon): {corr:.6f}")
    print(f"[RESULT] Real S2(0) = {real_s2_mean[0]:.6f}  |  S2({MAX_R-1}) = {real_s2_mean[-1]:.6f}")
    print(f"[RESULT] Recon S2(0) = {recon_s2_mean[0]:.6f}  |  S2({MAX_R-1}) = {recon_s2_mean[-1]:.6f}")

    # ===== Plot =====
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Real: solid line with shaded std
    ax.plot(r_values, real_s2_mean, '-', color=COLOR_REAL, linewidth=2.5,
            label='Real Rock (Ground Truth)')
    ax.fill_between(r_values,
                    real_s2_mean - real_s2_std,
                    real_s2_mean + real_s2_std,
                    alpha=0.2, color=COLOR_REAL)

    # Reconstruction: dashed line with shaded std
    ax.plot(r_values, recon_s2_mean, '--', color=COLOR_RECON, linewidth=2.5,
            label='VQ-VAE Reconstruction')
    ax.fill_between(r_values,
                    recon_s2_mean - recon_s2_std,
                    recon_s2_mean + recon_s2_std,
                    alpha=0.2, color=COLOR_RECON)

    # S2 asymptotic value (porosity squared for uncorrelated)
    # Add porosity^2 reference line
    pore_ref = real_s2_mean[-1]  # asymptotic value
    ax.axhline(y=pore_ref, color='gray', linestyle=':', linewidth=1.2,
               alpha=0.7, label=f'Asymptotic (porosity^2 ≈ {pore_ref:.4f})')

    # Correlation annotation
    ax.text(0.95, 0.92, f'Pearson r = {corr:.4f}\nN = {N_SAMPLES} pairs',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.9),
            verticalalignment='top', horizontalalignment='right')

    ax.set_xlabel('Distance r (pixels)', fontsize=13)
    ax.set_ylabel('Two-Point Probability S$_2$(r)', fontsize=13)
    ax.set_title('Microstructural Topology: Two-Point Probability Function',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, MAX_R)

    plt.tight_layout()

    output_path = os.path.join(EVAL_DIR, "s2_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] S2 curves saved to: {output_path}\n")


# ============================================================
# 主入口
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("  第五章 实验结果分析 — 出版级评估图表生成")
    print("=" * 70 + "\n")

    # ── Load models once ──
    encoder, decoder, vq_layer = load_models()

    # ── Task 1: Loss curves from the saved training history ──
    generate_loss_curves()

    # ── Task 2: Reconstruction comparison ──
    generate_reconstruction_comparison(encoder, decoder, vq_layer)

    # ── Task 3: Porosity analysis ──
    r2, mae = generate_porosity_analysis(encoder, decoder, vq_layer)

    # ── Task 4: S2 curves ──
    generate_s2_curves(encoder, decoder, vq_layer)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  图表生成完成！")
    print("=" * 70)
    print(f"  图 5-1: {os.path.join(EVAL_DIR, 'loss_curves.png')}")
    print(f"  图 5-2: {os.path.join(EVAL_DIR, 'reconstruction_comparison.png')}")
    print(f"  图 5-3: {os.path.join(EVAL_DIR, 'porosity_analysis.png')}")
    print(f"  图 5-4: {os.path.join(EVAL_DIR, 's2_curves.png')}")
    print(f"  图 5-5: {os.path.join(EVAL_DIR, 'random_batch_rocks.png')} (需单独运行 batch_generate_random_rocks.py)")
    print(f"\n  [关键指标] 孔隙度 R^2 = {r2:.6f}, MAE = {mae:.6f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
