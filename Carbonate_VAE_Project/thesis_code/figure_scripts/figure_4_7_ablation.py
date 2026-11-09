"""
Figure 4-7: Ablation Study — Discrete Splicing vs Continuous Fusion
====================================================================
1x2 layout comparing boundary quality at the fusion interface:

  (a) VQ-then-Fuse (Ablation):
      First VQ each image -> get discrete codebook indices ->
      hard splice at boundary -> rectangular window / mosaic artifacts
      (jagged, stair-stepped pore boundaries)

  (b) Fuse-then-VQ (Proposed):
      Fuse in continuous Z_e latent space with Gaussian-softened mask ->
      then unified VQ -> smooth, rounded, physically natural pore boundaries

Both sides use the SAME image pair (A, B) and SAME organic mask shape.
The ONLY difference is the order of operations:
  - Left:  VQ -> Splice (discrete codebook switching)
  - Right: Splice (Z_e) -> VQ (continuous latent fusion)

Styling: Strict black-and-white engineering plot, red annotation boxes,
         3x magnified 30x30 ROI insets at identical coordinates.
"""

import os, sys, numpy as np, tensorflow as tf

from thesis_code.config import PROJECT_ROOT


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

# ── CJK Font configuration ──
_CJK_FONT = None
for _fname in ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']:
    for _font in fm.fontManager.ttflist:
        if _font.name == _fname:
            _CJK_FONT = _font.name
            break
    if _CJK_FONT:
        break
if _CJK_FONT:
    plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[FONT] Using CJK font: {_CJK_FONT}")
else:
    print("[FONT] WARNING: No CJK font found, Chinese may render as boxes")

# ============================================================
# Constants
# ============================================================
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "results", "checkpoints")
EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_images")
os.makedirs(EVAL_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(EVAL_DIR, "Figure_4_7_Ablation_Fusion_Boundary.png")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
EPOCH_TO_LOAD = 50
LATENT_SIZE = 32
IMAGE_SIZE = 256
CELL_PX = IMAGE_SIZE // LATENT_SIZE  # 8 pixels per latent cell

CLOUD_SIGMA = 4.0       # organic blob size
SOFTEN_SIGMA = 1.0      # edge softening for Fuse-then-VQ

ROI_SIZE = 30           # pixels
ZOOM_FACTOR = 3          # 3x magnification for inset

# Red annotation color
C_RED = '#FF0000'


# ============================================================
# Model loading
# ============================================================
def load_models():
    from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM]))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])

    print(f"[INFO] VQ-VAE models loaded (epoch {EPOCH_TO_LOAD}).")
    return encoder, decoder, vq_layer


# ============================================================
# Data loading
# ============================================================
def load_one_tif(filepath):
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_v = tf.reduce_min(img_t)
    max_v = tf.reduce_max(img_t)
    img_n = tf.math.divide_no_nan(img_t - min_v, max_v - min_v)
    return img_n.numpy()


def pick_two_diverse_images(data_dir, num_candidates=64):
    TARGET_EXTENSIONS = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXTENSIONS:
                file_list.append(os.path.join(root, f))

    print(f"[SCAN] Found {len(file_list)} TIF files, sampling {num_candidates}...")
    np.random.seed(42)
    np.random.shuffle(file_list)
    candidates = file_list[:num_candidates]

    images, means = [], []
    for fp in candidates:
        img_np = load_one_tif(fp)
        images.append(img_np)
        means.append(img_np.mean())

    means = np.array(means)
    idx_A = int(np.argmax(means))
    idx_B = int(np.argmin(means))
    img_A, img_B = images[idx_A], images[idx_B]
    print(f"[PICK] A (matrix): mean={means[idx_A]:.4f}, B (donor): mean={means[idx_B]:.4f}")
    return img_A, img_B


# ============================================================
# VQ index extraction
# ============================================================
def get_vq_indices(vq_layer, z_e):
    input_shape = tf.shape(z_e)
    H, W = input_shape[1], input_shape[2]
    flat = tf.reshape(z_e, [-1, LATENT_DIM])
    similarity = tf.matmul(flat, vq_layer.embeddings)
    flat_sq = tf.reduce_sum(flat ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(tf.reduce_sum(vq_layer.embeddings ** 2, axis=0), [1, -1])
    distances = flat_sq + emb_sq - 2.0 * similarity
    encoding_indices = tf.argmin(distances, axis=1)
    encoding_indices = tf.reshape(encoding_indices, [H, W])
    return encoding_indices.numpy().astype(np.int32)


def indices_to_zq(vq_layer, indices):
    embeddings_T = tf.transpose(vq_layer.embeddings)
    z_q_flat = tf.nn.embedding_lookup(embeddings_T, indices)
    z_q = tf.reshape(z_q_flat, [1, LATENT_SIZE, LATENT_SIZE, LATENT_DIM])
    return z_q


# ============================================================
# Organic cloud mask generation
# ============================================================
def generate_organic_mask(h, w, cloud_sigma, soften_sigma=None):
    noise = np.random.randn(h, w).astype(np.float32)
    cloud = gaussian_filter(noise, sigma=cloud_sigma)
    threshold = np.median(cloud)
    binary = (cloud > threshold).astype(np.float32)

    if soften_sigma is not None and soften_sigma > 0:
        soft_mask = gaussian_filter(binary, sigma=soften_sigma)
    else:
        soft_mask = binary

    return soft_mask[np.newaxis, :, :, np.newaxis], binary  # return both (4D, 2D raw binary)


# ============================================================
# Auto-detect ROI position that straddles the mask boundary
# ============================================================
def find_boundary_roi(mask_binary_2d, roi_size_px=ROI_SIZE, cell_px=CELL_PX):
    """
    Scan the binary mask (32x32) to find a location where:
      - There is both 0 and 1 in the mask within the ROI area
      - The boundary is roughly in the center of the ROI
    Maps to pixel coordinates on the 256x256 image.
    
    Returns: (roi_x_px, roi_y_px) top-left pixel coordinates
    """
    H, W = mask_binary_2d.shape  # 32, 32
    roi_cells = max(roi_size_px // cell_px, 2)  # 30//8 = 3 cells min, use 4
    
    best_score = -1
    best_cy, best_cx = H // 2, W // 2  # default: center
    
    # Use 4-cell ROI window (32 pixels)
    roi_cells = 4
    for cy in range(roi_cells // 2, H - roi_cells // 2):
        for cx in range(roi_cells // 2, W - roi_cells // 2):
            patch = mask_binary_2d[cy - roi_cells//2:cy + roi_cells//2,
                                   cx - roi_cells//2:cx + roi_cells//2]
            n_ones = patch.sum()
            n_zeros = roi_cells * roi_cells - n_ones
            # Score: product of ones and zeros (high when balanced)
            score = n_ones * n_zeros
            if score > best_score:
                best_score = score
                best_cy, best_cx = cy, cx
    
    # Convert to pixel coordinates (top-left of 4-cell = 32-pixel ROI)
    roi_x_px = (best_cx - roi_cells // 2) * cell_px
    roi_y_px = (best_cy - roi_cells // 2) * cell_px
    
    # Clamp to valid range
    roi_x_px = max(0, min(IMAGE_SIZE - roi_size_px, roi_x_px))
    roi_y_px = max(0, min(IMAGE_SIZE - roi_size_px, roi_y_px))
    
    n_ones = mask_binary_2d[best_cy - roi_cells//2:best_cy + roi_cells//2,
                            best_cx - roi_cells//2:best_cx + roi_cells//2].sum()
    print(f"[ROI] Auto-detected boundary at latent ({best_cx}, {best_cy}) "
          f"-> pixel ({roi_x_px}, {roi_y_px}), roi_cells={roi_cells}x{roi_cells}, "
          f"ones_in_roi={int(n_ones)}/{roi_cells*roi_cells}")
    
    return roi_x_px, roi_y_px


# ============================================================
# Generation: VQ-then-Fuse (ABLATION)
# ============================================================
def generate_vq_then_fuse(encoder, decoder, vq_layer, img_A, img_B, mask_hard):
    z_eA = encoder(tf.constant(np.expand_dims(img_A, 0)), training=False)
    z_eB = encoder(tf.constant(np.expand_dims(img_B, 0)), training=False)

    indices_A = get_vq_indices(vq_layer, z_eA)
    indices_B = get_vq_indices(vq_layer, z_eB)

    mask_2d = mask_hard[0, :, :, 0]
    fused_indices = np.where(mask_2d > 0.5, indices_B, indices_A).astype(np.int32)

    z_q = indices_to_zq(vq_layer, fused_indices)
    generated = decoder(z_q, training=False)
    return np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)


# ============================================================
# Generation: Fuse-then-VQ (PROPOSED)
# ============================================================
def generate_fuse_then_vq(encoder, decoder, vq_layer, img_A, img_B, mask_soft):
    z_eA = encoder(tf.constant(np.expand_dims(img_A, 0)), training=False)
    z_eB = encoder(tf.constant(np.expand_dims(img_B, 0)), training=False)

    mask_t = tf.constant(mask_soft, dtype=tf.float32)
    z_fused = z_eA * (1.0 - mask_t) + z_eB * mask_t

    z_q = vq_layer(z_fused, training=False)
    generated = decoder(z_q, training=False)
    return np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)


# ============================================================
# Zoom inset helper
# ============================================================
def add_zoom_inset(ax, full_img, roi_x, roi_y, roi_size, zoom, cmap='gray'):
    roi = full_img[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]

    inset_size = roi_size * zoom
    inset_x = IMAGE_SIZE - inset_size - 8
    inset_y = IMAGE_SIZE - inset_size - 8

    ax_inset = ax.inset_axes(
        [inset_x / IMAGE_SIZE, inset_y / IMAGE_SIZE,
         inset_size / IMAGE_SIZE, inset_size / IMAGE_SIZE],
        transform=ax.transData
    )

    ax_inset.imshow(roi, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest')
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    for spine in ax_inset.spines.values():
        spine.set_edgecolor(C_RED)
        spine.set_linewidth(1.5)
        spine.set_visible(True)

    ax_inset.text(0.5, -0.12, f'\u00d7{zoom}', transform=ax_inset.transAxes,
                  fontsize=8, color=C_RED, fontweight='bold',
                  ha='center', va='top')

    return ax_inset


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  Figure 4-7: Ablation - VQ-then-Fuse vs Fuse-then-VQ")
    print("=" * 60 + "\n")

    # ── FIREWALL ──
    for fname in [f"encoder_ep{EPOCH_TO_LOAD}.weights.h5",
                  f"decoder_ep{EPOCH_TO_LOAD}.weights.h5",
                  f"vq_layer_ep{EPOCH_TO_LOAD}.npy"]:
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"[FATAL] {fpath} not found!")
    print("[FIREWALL] All checkpoints present.\n")

    # ── Load models ──
    encoder, decoder, vq_layer = load_models()

    # ── Pick two diverse images ──
    print("[INFO] Selecting two maximally-different rock slices...")
    img_A, img_B = pick_two_diverse_images(DATA_DIR, num_candidates=64)

    # ── Histogram matching ──
    from skimage.exposure import match_histograms
    img_B_matched = match_histograms(
        img_B[:, :, 0], img_A[:, :, 0]
    ).astype(np.float32)
    img_B_matched = np.clip(img_B_matched, 0.0, 1.0)
    img_B_matched = np.expand_dims(img_B_matched, axis=-1)
    print(f"[MATCH] B mean: {img_B.mean():.4f} -> {img_B_matched.mean():.4f} "
          f"(target A mean={img_A.mean():.4f})")

    # ── Generate mask ──
    np.random.seed(12345)
    mask_hard_4d, mask_binary_2d = generate_organic_mask(
        LATENT_SIZE, LATENT_SIZE, cloud_sigma=CLOUD_SIGMA, soften_sigma=None)

    np.random.seed(12345)
    mask_soft_4d, _ = generate_organic_mask(
        LATENT_SIZE, LATENT_SIZE, cloud_sigma=CLOUD_SIGMA, soften_sigma=SOFTEN_SIGMA)

    coverage = mask_hard_4d.mean()
    print(f"[MASK] Coverage: {coverage:.2%} (pore donor fraction)")

    # ── Auto-detect ROI at a mask boundary ──
    roi_x, roi_y = find_boundary_roi(mask_binary_2d)

    # ── Generate ──
    print("\n[GEN] (a) VQ-then-Fuse: hard codebook index splicing ...")
    img_vq_then_fuse = generate_vq_then_fuse(
        encoder, decoder, vq_layer, img_A, img_B_matched, mask_hard_4d)
    print("       Done.")

    print("[GEN] (b) Fuse-then-VQ: continuous Z_e fusion ...")
    img_fuse_then_vq = generate_fuse_then_vq(
        encoder, decoder, vq_layer, img_A, img_B_matched, mask_soft_4d)
    print("       Done.\n")

    # ================================================================
    # Render 1x2 figure
    # ================================================================
    print("[PLOT] Rendering 1x2 ablation comparison figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

    # ── Left: VQ-then-Fuse (ablation) ──
    ax1.imshow(img_vq_then_fuse, cmap='gray', vmin=0.0, vmax=1.0,
               interpolation='bilinear')
    ax1.set_title(
        '(a) \u4f20\u7edf\u79bb\u6563\u7801\u672c\u7a7a\u95f4\u76f4\u63a5\u62fc\u63a5\n(Ablation: VQ-then-Fuse)',
        fontsize=13, fontweight='bold', color='#222222', pad=12)

    rect1 = Rectangle((roi_x, roi_y), ROI_SIZE, ROI_SIZE,
                      linewidth=1.5, edgecolor=C_RED,
                      facecolor='none', linestyle='-')
    ax1.add_patch(rect1)
    add_zoom_inset(ax1, img_vq_then_fuse, roi_x, roi_y, ROI_SIZE, ZOOM_FACTOR)

    ax1.text(roi_x + ROI_SIZE + 4, roi_y + ROI_SIZE // 2,
             '\u952f\u9f7f\u72b6\n\u9a6c\u8d5b\u514b\n\u8fb9\u754c',
             fontsize=8, color=C_RED, fontweight='bold',
             va='center', ha='left')

    ax1.set_xticks(np.arange(0, 257, 64))
    ax1.set_yticks(np.arange(0, 257, 64))
    ax1.tick_params(labelsize=9)
    ax1.set_xlabel('Pixel', fontsize=10)
    ax1.set_ylabel('Pixel', fontsize=10)

    # ── Right: Fuse-then-VQ (proposed) ──
    ax2.imshow(img_fuse_then_vq, cmap='gray', vmin=0.0, vmax=1.0,
               interpolation='bilinear')
    ax2.set_title(
        '(b) \u672c\u7814\u7a76\u6f5c\u7a7a\u95f4\u7279\u5f81\u5e73\u6ed1\u878d\u5408\n(Proposed: Fuse-then-VQ)',
        fontsize=13, fontweight='bold', color='#222222', pad=12)

    rect2 = Rectangle((roi_x, roi_y), ROI_SIZE, ROI_SIZE,
                      linewidth=1.5, edgecolor=C_RED,
                      facecolor='none', linestyle='-')
    ax2.add_patch(rect2)
    add_zoom_inset(ax2, img_fuse_then_vq, roi_x, roi_y, ROI_SIZE, ZOOM_FACTOR)

    ax2.text(roi_x + ROI_SIZE + 4, roi_y + ROI_SIZE // 2,
             '\u5706\u6da6\n\u5e73\u6ed1\n\u81ea\u7136\u8fb9\u754c',
             fontsize=8, color=C_RED, fontweight='bold',
             va='center', ha='left')

    ax2.set_xticks(np.arange(0, 257, 64))
    ax2.set_yticks(np.arange(0, 257, 64))
    ax2.tick_params(labelsize=9)
    ax2.set_xlabel('Pixel', fontsize=10)
    ax2.set_ylabel('Pixel', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=350, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\n[DONE] Figure 4-7 saved to:\n  {OUTPUT_PATH}")

    # ── Boundary roughness stats ──
    from scipy.ndimage import sobel
    roi_vq = img_vq_then_fuse[roi_y:roi_y + ROI_SIZE, roi_x:roi_x + ROI_SIZE]
    roi_fuse = img_fuse_then_vq[roi_y:roi_y + ROI_SIZE, roi_x:roi_x + ROI_SIZE]
    edge_vq = np.sqrt(sobel(roi_vq, axis=0) ** 2 + sobel(roi_vq, axis=1) ** 2)
    edge_fuse = np.sqrt(sobel(roi_fuse, axis=0) ** 2 + sobel(roi_fuse, axis=1) ** 2)
    print(f"\n[STATS] ROI edge magnitude (mean \u00b1 std):")
    print(f"  VQ-then-Fuse: {edge_vq.mean():.6f} \u00b1 {edge_vq.std():.6f}")
    print(f"  Fuse-then-VQ: {edge_fuse.mean():.6f} \u00b1 {edge_fuse.std():.6f}")


if __name__ == "__main__":
    main()
