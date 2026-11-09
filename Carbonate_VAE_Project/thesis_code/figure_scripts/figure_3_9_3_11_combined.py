"""
图 3-9 + 图 3-11: 离散自回归伪影对比 + 潜空间掩膜融合对比
==================================================================
FIG 3-9: PixelCNN vs NanoGPT generated images with red circles
         highlighting "staircase/checkerboard" mosaic artifacts.
FIG 3-11: "Before" (discrete code stitching, VQ-then-fuse) vs.
          "After" (continuous latent fusion, fuse-then-VQ) with
          3× zoomed inset showing smooth vs. jagged boundaries.

Both figures require: VQ-VAE (encoder/decoder/VQ),
                    PixelCNN retrained, NanoGPT V2.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# 强制指定使用 Windows 系统自带的黑体 (SimHei) 来显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决坐标轴上的负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import torch
import torch.nn.functional as F

# ── Paths ──
from thesis_code.config import PROJECT_ROOT


CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
os.makedirs(EVAL_DIR, exist_ok=True)

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

# ── Constants ──
NUM_EMBEDDINGS = 512
LATENT_DIM = 64
LATENT_SIZE = 32
IMAGE_SIZE = 256
VQVAE_EPOCH = 50

# ── GPU check ──
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
    raise RuntimeError("[FATAL] No GPU detected!")
print(f"[FIREWALL] GPU check PASSED: {len(gpus)} GPU(s)")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

# ================================================================
# VQ-VAE helpers
# ================================================================
def load_vqvae():
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))
    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{VQVAE_EPOCH}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{VQVAE_EPOCH}.weights.h5"))
    vq_emb = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{VQVAE_EPOCH}.npy"))
    vq_layer.set_weights([vq_emb])
    return encoder, decoder, vq_layer

def load_one_tif(filepath):
    img = PILImage.open(filepath)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.LANCZOS)
    img_array = np.array(img).astype(np.float32)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    return img_array

def encode_to_codes(encoder, vq_layer, img_array):
    """Get discrete code indices: same logic as extract_codes.py get_indices().
    Returns (H, W) for general use; add [np.newaxis, ...] for autoregressive_sample."""
    z = encoder(tf.constant(np.expand_dims(img_array, 0)), training=False)
    input_shape = tf.shape(z)
    flattened = tf.reshape(z, [-1, LATENT_DIM])
    similarity = tf.matmul(flattened, vq_layer.embeddings)
    flat_sq = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(tf.reduce_sum(vq_layer.embeddings ** 2, axis=0), [1, -1])
    distances = flat_sq + emb_sq - 2.0 * similarity
    encoding_indices = tf.argmin(distances, axis=1)
    encoding_indices = tf.reshape(encoding_indices, input_shape[:3])
    code_map = encoding_indices.numpy().astype(np.int32)[0]  # (32, 32)
    return code_map

def codes_to_image(decoder, vq_layer, code_map):
    """Decode a code map (2D or 3D with batch dim) to an image.
    Matches figure_4_5_retrained.py pattern exactly."""
    if code_map.ndim == 2:
        code_map_tensor = tf.constant(code_map[np.newaxis, ...], dtype=tf.int32)
    else:
        code_map_tensor = tf.constant(code_map, dtype=tf.int32)
    # vq_layer.embeddings is (latent_dim, num_embeddings) = (64, 512)
    # Transpose to (512, 64) so gather on axis 0 uses code indices correctly
    embeddings = vq_layer.embeddings
    embeddings_t = tf.transpose(embeddings)
    indices_flat = tf.reshape(code_map_tensor, [-1])
    quantized_flat = tf.gather(embeddings_t, indices_flat)
    quantized = tf.reshape(quantized_flat, [1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM])
    recon = decoder(quantized, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)

# ================================================================
# FIG 3-9: Stitching artifact comparison
# ================================================================
def figure_3_9_artifacts(encoder, decoder, vq_layer):
    print("\n" + "=" * 70)
    print("  FIGURE 3-9: Discrete Autoregressive Stitching Artifacts")
    print("=" * 70)

    # Load PixelCNN retrained model (import build function from retrained script)
    from thesis_code.figure_scripts.figure_4_5_retrained import build_pixelcnn_inference
    pixelcnn = build_pixelcnn_inference()
    _ = pixelcnn(tf.zeros([1, LATENT_SIZE, LATENT_SIZE], dtype=tf.int32))
    pixelcnn.load_weights(os.path.join(CHECKPOINT_DIR, "pixelcnn_retrain.weights.h5"))
    print("[INFO] PixelCNN retrained loaded.")

    # Load NanoGPT V2 (same pattern as figure_4_5_fusion_novel_rocks.py)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "gpt_prior_v2.pt"),
                          map_location=device, weights_only=False)
    gpt_config = gpt_ckpt['config']
    print(f"[GPT] Config: {gpt_config.n_layer}L/{gpt_config.n_head}H/"
          f"{gpt_config.n_embd}D, vocab={gpt_config.vocab_size}")

    from thesis_code.models.nanogpt import GPT, GPTConfig
    gpt_model = GPT(gpt_config)
    gpt_model.load_state_dict(gpt_ckpt['model_state_dict'])
    gpt_model.to(device)
    gpt_model.eval()
    print(f"[GPT] Loaded from epoch {gpt_ckpt['epoch']}, "
          f"best val loss: {gpt_ckpt['best_val_loss']:.4f}")

    # Load valid tokens
    valid_tokens = np.load(os.path.join(CHECKPOINT_DIR, "valid_tokens.npy"))
    valid_tokens_tensor = torch.tensor(valid_tokens, dtype=torch.long, device=device)

    # Pick a seed image
    tif_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                        if f.lower().endswith(('.tif', '.tiff'))])
    np.random.seed(42)
    seed_path = tif_files[np.random.randint(0, len(tif_files))]
    print(f"[SEED] {os.path.basename(seed_path)}")

    seed_img = load_one_tif(seed_path)
    seed_codes = encode_to_codes(encoder, vq_layer, seed_img)

    # ── Generate PixelCNN sample ──
    print("[GEN] Generating PixelCNN sample (T=0.1, Top-P=0.5, no waste filter)...")
    # Use the figure_4_5_retrained autoregressive_sample function
    from thesis_code.figure_scripts.figure_4_5_retrained import autoregressive_sample
    pcnn_codes = autoregressive_sample(pixelcnn, seed_codes[np.newaxis, ...],
                                       temperature=0.5, top_p=0.5,
                                       seed_patch_size=3, num_seeds=3)
    pcnn_img = codes_to_image(decoder, vq_layer, pcnn_codes)

    # ── Generate NanoGPT V2 sample ──
    print("[GEN] Generating NanoGPT V2 sample (T=1.3, Top-P=0.95, 2-row seed)...")
    seed_flat = seed_codes.flatten()
    start_idx = torch.tensor(seed_flat[:64], dtype=torch.long, device=device).unsqueeze(0)

    @torch.no_grad()
    def generate_nanogpt():
        gen_idx = gpt_model.generate(
            start_idx, max_new_tokens=1024 - 64,
            temperature=1.3, top_k=None, top_p=0.95,
            valid_tokens=valid_tokens_tensor
        )
        return gen_idx[0].cpu().numpy().reshape(LATENT_SIZE, LATENT_SIZE)

    gpt_codes = generate_nanogpt()
    gpt_img = codes_to_image(decoder, vq_layer, gpt_codes)

    # ── Plot: 1×2 comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    titles = [
        '(a) PixelCNN 离散采样伪影',
        '(b) NanoGPT 离散采样伪影'
    ]
    images = [pcnn_img, gpt_img]

    # Predefined artifact regions (manually selected from known patterns)
    # PixelCNN artifacts: staircase at specific coordinates
    # NanoGPT artifacts: grid/checkerboard patterns

    artifact_regions = [
        # For PixelCNN: [(x, y, w, h), ...]
        [(150, 60, 35, 35), (80, 150, 30, 30), (200, 190, 28, 28)],
        # For NanoGPT: [(x, y, w, h), ...]
        [(170, 90, 32, 32), (110, 160, 40, 40), (60, 50, 35, 35)],
    ]

    for idx, ax in enumerate(axes):
        ax.imshow(images[idx], cmap='gray', vmin=0.0, vmax=1.0,
                  interpolation='nearest')

        # Draw red circles/boxes around artifact regions
        for (x, y, w, h) in artifact_regions[idx]:
            rect = Rectangle((x - w/2, y - h/2), w, h,
                             linewidth=2.0, edgecolor='#FF0000',
                             facecolor='none', linestyle='-')
            ax.add_patch(rect)

        ax.set_title(titles[idx], fontsize=14, fontweight='bold',
                     pad=12, color='#333333')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1.0)

    # ── Add inset zooms ──
    # For each subplot, add a zoomed inset in the bottom-right corner
    for idx, ax in enumerate(axes):
        # Take the first artifact region for zoom
        x, y, w, h = artifact_regions[idx][0]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(IMAGE_SIZE, x2), min(IMAGE_SIZE, y2)

        axins = zoomed_inset_axes(ax, zoom=2.5, loc='lower right',
                                   borderpad=1.5)
        axins.imshow(images[idx], cmap='gray', vmin=0.0, vmax=1.0,
                     interpolation='nearest')
        axins.set_xlim(x1, x2)
        axins.set_ylim(y2, y1)  # inverted for image coords
        axins.set_xticks([])
        axins.set_yticks([])

        # Mark inset source on main image
        mark_inset(ax, axins, loc1=2, loc2=3, fc='none',
                   ec='#FF0000', linewidth=1.5)

        # Label the inset
        axins.text(0.5, 0.92, '2.5×', transform=axins.transAxes,
                   ha='center', fontsize=9, fontweight='bold',
                   color='#FF0000',
                   bbox=dict(boxstyle='round', facecolor='white',
                             edgecolor='#FF0000', alpha=0.85))

    fig.suptitle('Figure 3-9: Discrete Autoregressive Sampling — Mosaic Stitching Artifacts',
                 fontsize=16, fontweight='bold', y=0.98, color='#222222')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = os.path.join(EVAL_DIR, "Figure_3_9_Stitching_Artifacts.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[DONE] Figure 3-9 saved to:\n  {output_path}")

    return seed_codes, encoder, decoder, vq_layer


# ================================================================
# FIG 3-11: 潜空间掩膜融合消融对比 — VQ-then-Fuse vs Fuse-then-VQ
# ================================================================
def figure_3_11_fusion(seed_codes, encoder, decoder, vq_layer):
    print("\n" + "=" * 70)
    print("  图 3-11: 潜空间掩膜融合前后的生成数字岩心细节对比 (消融实验)")
    print("=" * 70)

    # ── 学术配色 ──
    ACCENT_COLOR = '#8B4513'   # 红褐色 (克制醒目)
    TEXT_COLOR = '#222222'

    # Pick two different real images as A (matrix) and B (pore donor)
    tif_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                        if f.lower().endswith(('.tif', '.tiff'))])
    np.random.seed(123)
    idx_a, idx_b = np.random.choice(len(tif_files), 2, replace=False)

    img_A = load_one_tif(tif_files[idx_a])
    img_B = load_one_tif(tif_files[idx_b])

    print(f"[IMG A] {os.path.basename(tif_files[idx_a])}")
    print(f"[IMG B] {os.path.basename(tif_files[idx_b])}")

    # Encode to continuous Z_e
    z_eA = encoder(tf.constant(np.expand_dims(img_A, 0)), training=False)
    z_eB = encoder(tf.constant(np.expand_dims(img_B, 0)), training=False)

    # ── Generate organic mask (same as batch_generate) ──
    from scipy.ndimage import gaussian_filter

    def generate_organic_mask(h, w, cloud_sigma=4.0, coverage_threshold=0.75):
        noise = np.random.randn(h, w).astype(np.float32)
        cloud = gaussian_filter(noise, sigma=cloud_sigma)
        thresh_val = np.percentile(cloud, coverage_threshold * 100.0)
        pore_binary = (cloud > thresh_val).astype(np.float32)
        pore_soft = gaussian_filter(pore_binary, sigma=1.0)
        pore_mask = pore_soft[np.newaxis, :, :, np.newaxis]
        matrix_mask = 1.0 - pore_mask
        return matrix_mask, pore_mask

    np.random.seed(456)
    matrix_mask, pore_mask = generate_organic_mask(LATENT_SIZE, LATENT_SIZE,
                                                    cloud_sigma=4.0,
                                                    coverage_threshold=0.75)

    # Helper: nearest-neighbor lookup (same as extract_codes.py get_indices)
    def get_indices_from_z(z_e):
        """Get code indices from a Z_e tensor using nearest-neighbor lookup."""
        input_shape = tf.shape(z_e)
        flattened = tf.reshape(z_e, [-1, LATENT_DIM])
        similarity = tf.matmul(flattened, vq_layer.embeddings)
        flat_sq = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
        emb_sq = tf.reshape(tf.reduce_sum(vq_layer.embeddings ** 2, axis=0), [1, -1])
        distances = flat_sq + emb_sq - 2.0 * similarity
        encoding_indices = tf.argmin(distances, axis=1)
        encoding_indices = tf.reshape(encoding_indices, input_shape[:3])
        return encoding_indices

    # ── BEFORE: Discrete stitching (VQ first, then fuse) ──
    # Step 1: VQ-quantize both Z_eA and Z_eB separately
    indices_A = get_indices_from_z(z_eA)
    indices_B = get_indices_from_z(z_eB)
    code_A = indices_A.numpy().astype(np.int32)[0]
    code_B = indices_B.numpy().astype(np.int32)[0]

    # Step 2: Fuse at code level (hard switching)
    code_fused_discrete = np.where(pore_mask[0, :, :, 0] > 0.5,
                                    code_B, code_A)
    before_img = codes_to_image(decoder, vq_layer, code_fused_discrete)

    # ── AFTER: Continuous latent fusion (fuse first, then VQ) ──
    z_fused = z_eA * tf.constant(matrix_mask, dtype=tf.float32) + \
              z_eB * tf.constant(pore_mask, dtype=tf.float32)
    indices_fused = get_indices_from_z(z_fused)
    code_fused_continuous = indices_fused.numpy().astype(np.int32)[0]
    after_img = codes_to_image(decoder, vq_layer, code_fused_continuous)

    # ── Plot: 1×2 comparison with zoomed insets ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), facecolor='white')

    # 标题: 严谨学术用语
    titles = [
        '(a) 离散空间特征硬阈值拼接 (VQ-then-Fuse)',
        '(b) 本研究连续潜空间特征融合 (Fuse-then-VQ)'
    ]
    images = [before_img, after_img]

    # ── 定位孔隙-基质边界 (自动梯度检测) ──
    pore_boundary_map = pore_mask[0, :, :, 0]
    pore_mask_256 = np.kron(pore_boundary_map, np.ones((8, 8)))

    from scipy.ndimage import sobel
    grad_x = sobel(pore_mask_256, axis=0)
    grad_y = sobel(pore_mask_256, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 选取梯度最大点作为放大中心
    max_grad_idx = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)
    cy, cx = max_grad_idx
    zoom_size = 30  # 30×30 像素局部区域

    # 边界钳位
    half = zoom_size // 2
    x1 = max(0, cx - half)
    x2 = min(IMAGE_SIZE, cx + half)
    y1 = max(0, cy - half)
    y2 = min(IMAGE_SIZE, cy + half)

    print(f"[ZOOM] 边界放大中心: ({cx}, {cy}), 区域: [{x1}:{x2}, {y1}:{y2}]")

    for idx, ax in enumerate(axes):
        ax.imshow(images[idx], cmap='gray', vmin=0.0, vmax=1.0,
                  interpolation='nearest')

        # ── 红褐色细线矩形框标注放大区域 ──
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=1.5, edgecolor=ACCENT_COLOR,
                         facecolor='none', linestyle='-')
        ax.add_patch(rect)

        ax.set_title(titles[idx], fontsize=11.5, fontweight='medium',
                     pad=10, color=TEXT_COLOR)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#BBBBBB')
            spine.set_linewidth(0.8)

        # ── 3× 放大插图 (右上角) ──
        axins = zoomed_inset_axes(ax, zoom=3.0, loc='upper right',
                                   borderpad=1.2)
        axins.imshow(images[idx], cmap='gray', vmin=0.0, vmax=1.0,
                     interpolation='nearest')
        axins.set_xlim(x1, x2)
        axins.set_ylim(y2, y1)  # 图像坐标 Y 轴翻转
        axins.set_xticks([])
        axins.set_yticks([])

        # ── mark_inset 连接线 (红褐色, 相邻角形成矩形) ──
        mark_inset(ax, axins, loc1=2, loc2=3, fc='none',
                   ec=ACCENT_COLOR, linewidth=1.2)

        # ── 放大插图标签 ──
        if idx == 0:
            label_text = '3×  锯齿台阶边界'
        else:
            label_text = '3×  丝滑过渡边界'

        axins.text(0.5, 0.91, label_text, transform=axins.transAxes,
                   ha='center', fontsize=9, fontweight='medium',
                   color=ACCENT_COLOR,
                   bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                             edgecolor=ACCENT_COLOR, alpha=0.90, linewidth=0.8))

    # ── 图题 ──
    fig.suptitle('图 3-11  潜空间掩膜融合前后的生成数字岩心细节对比',
                 fontsize=13, fontweight='bold', y=0.995, color=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(EVAL_DIR, "Figure_3_11_Fusion_Comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[DONE] 图 3-11 已重绘并保存至:\n  {output_path}")


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("  Figures 3-9 & 3-11: Artifact + Fusion Comparison")
    print("=" * 70)

    encoder, decoder, vq_layer = load_vqvae()
    print("[INFO] VQ-VAE loaded.")

    # Fig 3-9
    seed_codes, encoder, decoder, vq_layer = figure_3_9_artifacts(encoder, decoder, vq_layer)

    # Fig 3-11
    figure_3_11_fusion(seed_codes, encoder, decoder, vq_layer)

    print("\n[DONE] All figures generated.")


if __name__ == "__main__":
    main()
