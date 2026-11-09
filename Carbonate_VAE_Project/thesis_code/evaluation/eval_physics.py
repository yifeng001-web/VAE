"""
[VQ-VAE] Physical Porosity Evaluation Script
Evaluates porosity preservation between real and AI-reconstructed CT slices.
Generates: scatter regression plot + overlapping histogram.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thesis_code.utils.data_loader import create_dataset
from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
NUM_SAMPLES = 100


def compute_porosity(images):
    """
    计算每张图像的孔隙度（像素均值）
    在 Min-Max 归一化后，均值可近似代表孔隙度分布。
    输入: (N, H, W, 1) tensor，输出: (N,) numpy array
    """
    return tf.reduce_mean(images, axis=[1, 2, 3]).numpy()


def main():
    print(f"[INFO] Building model and loading epoch {EPOCH_TO_LOAD} weights...")
    
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    # 初始化计算图
    dummy_input = tf.zeros([1, 256, 256, 1])
    _ = decoder(vq_layer(encoder(dummy_input)))

    # 加载权重
    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])
    print("[INFO] All weights loaded successfully.")

    # 从数据管道中收集 100 张真实切片
    print(f"[INFO] Collecting {NUM_SAMPLES} real CT slices from data pipeline...")
    dataset, _ = create_dataset(DATA_DIR, batch_size=4)

    real_images_list = []
    count = 0
    for batch_images, _ in dataset:
        real_images_list.append(batch_images.numpy())
        count += batch_images.shape[0]
        if count >= NUM_SAMPLES:
            break

    real_images = np.concatenate(real_images_list, axis=0)[:NUM_SAMPLES]
    print(f"[INFO] Collected {real_images.shape[0]} real images.")

    # 模型推理：生成 AI 重构切片
    print("[INFO] Running VQ-VAE reconstruction...")
    real_tensor = tf.convert_to_tensor(real_images, dtype=tf.float32)
    z_e = encoder(real_tensor, training=False)
    quantized_z = vq_layer(z_e)
    reconstructed_tensor = decoder(quantized_z, training=False)
    reconstructed = reconstructed_tensor.numpy()
    
    # Clip 到 [0, 1] 保证物理有效性
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    # 计算孔隙度
    real_porosity = np.mean(real_images, axis=(1, 2, 3))
    recon_porosity = np.mean(reconstructed, axis=(1, 2, 3))

    # 计算统计指标
    mae = np.mean(np.abs(real_porosity - recon_porosity))
    
    # R² 计算
    ss_res = np.sum((real_porosity - recon_porosity) ** 2)
    ss_tot = np.sum((real_porosity - np.mean(real_porosity)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"[RESULT] Mean Absolute Porosity Error: {mae:.6f}")
    print(f"[RESULT] R^2 Score: {r2:.4f}")

    # ===== 绘制 1x2 组合图 =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ---- 左图：散点回归图 ----
    ax1.scatter(real_porosity, recon_porosity, alpha=0.6, s=30, 
                c='steelblue', edgecolors='white', linewidth=0.3)
    
    # y = x 基准线
    min_val = min(real_porosity.min(), recon_porosity.min())
    max_val = max(real_porosity.max(), recon_porosity.max())
    padding = (max_val - min_val) * 0.05
    line_range = [min_val - padding, max_val + padding]
    ax1.plot(line_range, line_range, 'r--', linewidth=1.5, label='y = x (ideal)')
    
    # 拟合线（使用 numpy polyfit 避免 sklearn 依赖）
    slope, intercept = np.polyfit(real_porosity, recon_porosity, 1)
    x_fit = np.linspace(line_range[0], line_range[1], 100)
    y_fit = slope * x_fit + intercept
    ax1.plot(x_fit, y_fit, 'orange', linewidth=1.5,
             label=f'Fit: y={slope:.3f}x+{intercept:.4f}')
    
    ax1.set_xlabel('Real Porosity (pixel mean)', fontsize=12)
    ax1.set_ylabel('Reconstructed Porosity (pixel mean)', fontsize=12)
    ax1.set_title(f'Porosity Cross-Plot (N={NUM_SAMPLES})\nMAE={mae:.4f}  R^2={r2:.4f}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(line_range)
    ax1.set_ylim(line_range)
    ax1.set_aspect('equal')

    # ---- 右图：交叠直方图 ----
    ax2.hist(real_porosity, bins=25, alpha=0.6, color='steelblue', 
             edgecolor='navy', label='Real Porosity')
    ax2.hist(recon_porosity, bins=25, alpha=0.6, color='coral', 
             edgecolor='darkred', label='Reconstructed Porosity')
    ax2.set_xlabel('Porosity (pixel mean)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Porosity Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('VQ-VAE Physical Porosity Preservation Analysis (Epoch 50)', 
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path = os.path.join(EVAL_DIR, "porosity_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Porosity analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
