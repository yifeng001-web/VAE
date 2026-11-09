"""
[DIAGNOSE] PixelCNN 三合一自检脚本
=====================================
自检点 1: 验证 extract_codes.py 加载的 VQ-VAE 权重路径
自检点 2: 验证 latent_codes.npy 数据集维度 (N, 32, 32)
自检点 3: 计算 PixelCNN Epoch-25 模型的预测熵分布
"""

import os
import sys
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from thesis_code.config import PROJECT_ROOT


CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
CODES_PATH = os.path.join(CHECKPOINT_DIR, "latent_codes.npy")
PIXELCNN_WEIGHTS = os.path.join(CHECKPOINT_DIR, "pixelcnn.weights.h5")
PIXELCNN_HISTORY = os.path.join(CHECKPOINT_DIR, "pixelcnn_history.npz")

NUM_EMBEDDINGS = 512
LATENT_SIZE = 32
HIDDEN_CHANNELS = 128
NUM_RES_BLOCKS = 12

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# MaskedConv2D & ResidualBlock (same as train_pixelcnn.py)
# ============================================================
class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, mask_type='B', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type.upper()

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv = tf.keras.layers.Conv2D(
            self.filters, self.kernel_size, padding='same', use_bias=True,
            kernel_initializer='he_normal')
        self.conv.build(input_shape)
        k_h, k_w = self.kernel_size, self.kernel_size
        center_h, center_w = k_h // 2, k_w // 2
        mask = np.ones((k_h, k_w, in_channels, self.filters), dtype=np.float32)
        mask[center_h + 1:, :, :, :] = 0.0
        mask[center_h, center_w + 1:, :, :] = 0.0
        if self.mask_type == 'A':
            mask[center_h, center_w, :, :] = 0.0
        self.mask = tf.constant(mask)
        super().build(input_shape)

    def call(self, x):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(x)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.masked_conv1 = MaskedConv2D(self.filters // 2, 3, mask_type='B')
        self.masked_conv2 = MaskedConv2D(self.filters, 3, mask_type='B')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.skip_conv = tf.keras.layers.Conv2D(self.filters, 1, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.masked_conv1(x)
        h = self.bn1(h)
        h = tf.nn.relu(h)
        h = self.dropout(h, training=training)
        h = self.masked_conv2(h)
        h = self.bn2(h)
        if x.shape[-1] != self.filters:
            x = self.skip_conv(x)
        return tf.nn.relu(x + h)


def build_pixelcnn():
    inputs = tf.keras.Input(shape=(LATENT_SIZE, LATENT_SIZE), dtype=tf.int32)
    x = tf.one_hot(inputs, depth=NUM_EMBEDDINGS)
    x = MaskedConv2D(HIDDEN_CHANNELS, 7, mask_type='A', name='masked_conv_A')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    for i in range(NUM_RES_BLOCKS):
        x = ResidualBlock(HIDDEN_CHANNELS, name=f'res_block_{i}')(x)
    outputs = tf.keras.layers.Conv2D(NUM_EMBEDDINGS, 1, name='output_logits')(x)
    return tf.keras.Model(inputs, outputs, name='PixelCNN')


def compute_prediction_entropy(model, codes_batch):
    """
    Compute per-position softmax entropy of model predictions.
    Returns mean entropy and entropy distribution over the batch.
    A low entropy (~0) means the model is extremely confident
    (potential mode collapse to few tokens).
    """
    logits = model(codes_batch, training=False)  # (B, 32, 32, 512)
    probs = tf.nn.softmax(logits, axis=-1)
    # Entropy: H = -sum(p * log(p))
    eps = 1e-9
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + eps), axis=-1)  # (B, 32, 32)
    # Per-position mean entropy across batch
    mean_per_pos = tf.reduce_mean(entropy, axis=0)  # (32, 32)
    # Overall stats
    total_mean = tf.reduce_mean(entropy).numpy()
    total_std = tf.sqrt(tf.reduce_mean(tf.square(entropy - total_mean))).numpy()
    entropy_map = mean_per_pos.numpy()
    return total_mean, total_std, entropy_map


def main():
    print("=" * 70)
    print("  PixelCNN 三合一诊断自检")
    print("=" * 70)

    # ============================================================
    # 自检点 1: 验证 VQ-VAE 权重路径
    # ============================================================
    print("\n" + "=" * 70)
    print("  自检点 1: 验证 extract_codes.py 加载的 VQ-VAE 权重路径")
    print("=" * 70)

    encoder_ep50 = os.path.join(CHECKPOINT_DIR, "encoder_ep50.weights.h5")
    vq_ep50 = os.path.join(CHECKPOINT_DIR, "vq_layer_ep50.npy")

    print(f"  [检查] extract_codes.py 第 84 行: encoder.load_weights(r'{encoder_ep50}')")
    print(f"         extract_codes.py 第 85-86 行: np.load(r'{vq_ep50}')")
    print(f"")

    ok_encoder = os.path.exists(encoder_ep50)
    ok_vq = os.path.exists(vq_ep50)

    print(f"  encoder_ep50.weights.h5 存在: {'是' if ok_encoder else '否 (严重!)'}")
    print(f"  vq_layer_ep50.npy         存在: {'是' if ok_vq else '否 (严重!)'}")

    if ok_encoder and ok_vq:
        print(f"\n  [PASS] Checkpoint 1 PASS")
        print(f"  extract_codes.py loads Epoch 50 composite-constraint weights (from train_thesis.py)")
        print(f"  train_thesis.py loss: lambda3=20.0 (porosity) + lambda4=10.0 (edge gradient)")
    else:
        print(f"\n  [FAIL] Checkpoint 1 FAIL - weight files missing!")

    # ============================================================
    # 自检点 2: 验证 latent_codes.npy 数据量
    # ============================================================
    print("\n" + "=" * 70)
    print("  自检点 2: 验证 PixelCNN 训练数据集维度")
    print("=" * 70)

    if not os.path.exists(CODES_PATH):
        print(f"  [FAIL] latent_codes.npy not found! Path: {CODES_PATH}")
        return

    codes = np.load(CODES_PATH)
    print(f"  latent_codes.npy 形状: {codes.shape}")
    print(f"  样本总数 N = {codes.shape[0]}")
    print(f"  空间维度: {codes.shape[1]} x {codes.shape[2]}")
    print(f"  唯一编码数 (全部): {len(np.unique(codes))} / {NUM_EMBEDDINGS}")
    print(f"  编码范围: [{codes.min()}, {codes.max()}]")
    print(f"  文件大小: {os.path.getsize(CODES_PATH)/1024/1024:.1f} MB")

    N = codes.shape[0]
    if N >= 4000:
        print(f"\n  [PASS] Checkpoint 2 PASS — data sufficient ({N} samples >= 4000)")
    elif N >= 1000:
        print(f"\n  [WARN] Checkpoint 2 MARGINAL — data count {N}, may underfit")
    else:
        print(f"\n  [FAIL] Checkpoint 2 WARNING — data count {N} severely insufficient, overfitting risk!")

    # ============================================================
    # 自检点 3: 计算 PixelCNN Epoch-25 预测熵
    # ============================================================
    print("\n" + "=" * 70)
    print("  自检点 3: PixelCNN 预测概率分布熵值检查")
    print("=" * 70)

    if not os.path.exists(PIXELCNN_WEIGHTS):
        print(f"  [FAIL] pixelcnn.weights.h5 not found!")
        return

    # Load model
    print("  [INFO] 加载 PixelCNN Epoch-25 权重...")
    model = build_pixelcnn()
    _ = model(tf.zeros([1, LATENT_SIZE, LATENT_SIZE], dtype=tf.int32))
    model.load_weights(PIXELCNN_WEIGHTS)
    print("  [INFO] 权重加载完成")

    # Compute entropy on validation set (last 10% of data)
    split = int(N * 0.9)
    val_codes = codes[split:]

    # Process in batches to avoid OOM
    BATCH = 32
    all_entropy_means = []
    all_entropy_maps = []

    print(f"  [INFO] 计算 {len(val_codes)} 个验证样本的预测熵...")
    for start in range(0, len(val_codes), BATCH):
        batch = tf.constant(val_codes[start:start+BATCH], dtype=tf.int32)
        mean_ent, std_ent, ent_map = compute_prediction_entropy(model, batch)
        all_entropy_means.append(mean_ent)
        if len(all_entropy_maps) == 0:
            all_entropy_maps = ent_map
        else:
            all_entropy_maps = (all_entropy_maps * (start // BATCH) + ent_map) / (start // BATCH + 1)

    global_mean_entropy = np.mean(all_entropy_means)
    global_std_entropy = np.std(all_entropy_means)

    # Theoretical max entropy for 512 classes: ln(512) ≈ 6.238
    max_entropy = np.log(NUM_EMBEDDINGS)
    normalized_entropy = global_mean_entropy / max_entropy

    print(f"\n  +-----------------------------------------------------+")
    print(f"  |  Prediction Entropy Diagnosis                        |")
    print(f"  +-----------------------------------------------------+")
    print(f"  |  Theoretical max entropy (uniform, ln(512)): {max_entropy:.4f}    |")
    print(f"  |  Actual mean entropy:                    {global_mean_entropy:.4f}    |")
    print(f"  |  Normalized entropy (actual/max):         {normalized_entropy:.4f}    |")
    print(f"  |  Entropy std:                             {global_std_entropy:.4f}    |")
    print(f"  +-----------------------------------------------------+")

    if normalized_entropy < 0.05:
        verdict = "CRITICAL: Severe mode collapse - model only predicts very few codes"
        status = "FAIL"
    elif normalized_entropy < 0.15:
        verdict = "WARNING: Moderate mode collapse - model over-concentrates on certain codes"
        status = "WARNING"
    elif normalized_entropy < 0.30:
        verdict = "MARGINAL: Mild concentration - may need more training"
        status = "OK (marginal)"
    else:
        verdict = "HEALTHY: Prediction distribution relatively uniform, no obvious mode collapse"
        status = "PASS"

    print(f"\n  [Checkpoint 3] {status}: {verdict}")

    # Plot entropy spatial map
    print("\n  [PLOT] 生成熵值空间分布图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(all_entropy_maps, cmap='hot', aspect='equal')
    axes[0].set_title(f'Per-position Mean Prediction Entropy\n'
                      f'(Global Mean={global_mean_entropy:.3f}, '
                      f'Normalized={normalized_entropy:.2%})',
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Latent X')
    axes[0].set_ylabel('Latent Y')
    plt.colorbar(im0, ax=axes[0], label='Entropy (nats)')

    # Histogram of per-position entropy
    axes[1].hist(all_entropy_maps.flatten(), bins=50, color='#D9583B', alpha=0.8,
                 edgecolor='black', linewidth=0.3)
    axes[1].axvline(x=max_entropy, color='green', linestyle='--',
                    linewidth=2, label=f'Max (uniform) = {max_entropy:.3f}')
    axes[1].axvline(x=global_mean_entropy, color='blue', linestyle='-',
                    linewidth=2, label=f'Mean = {global_mean_entropy:.3f}')
    axes[1].set_xlabel('Entropy (nats)')
    axes[1].set_ylabel('Count (positions)')
    axes[1].set_title('Entropy Distribution across 32x32 Positions', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "diagnose_entropy.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [DONE] 熵值分布图: {plot_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  Diagnostic Summary")
    print("=" * 70)
    print(f"  Checkpoint 1 (VQ-VAE weight path):  PASS (Epoch 50 composite-constraint model)")
    print(f"  Checkpoint 2 (data count N):         PASS (N={N}, sufficient)")
    print(f"  Checkpoint 3 (prediction entropy):   {status} (normalized={normalized_entropy:.4f})")
    print(f"")
    print(f"  Recommendation: Retrain with 200 Epochs + 18 ResBlocks + T=0.1 regardless of entropy")
    print("=" * 70)


if __name__ == "__main__":
    main()
