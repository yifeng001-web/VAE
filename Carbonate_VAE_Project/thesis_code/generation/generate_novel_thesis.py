"""
[VQ-VAE] Novel Digital Rock Generation — Single-Image Latent Mutation
Abandons dual-image mixing. Uses only ONE parent image (Image A).
Perturbs the continuous latent feature map with organic Gaussian noise,
then re-quantizes through VQ layer — the perturbed vectors get snapped
to the nearest legitimate codebook entries, producing novel pore structures
while keeping the background matrix absolutely consistent.

Algorithm:
  1. Encode Image A → continuous Z_eA (1, 32, 32, 64)  [BEFORE VQ]
  2. For each mutation strength, generate 3D Gaussian noise
     → spatially blur (sigma=[1.5, 1.5, 0]) → add to Z_eA
  3. Re-quantize: Z_q_novel = VQ(Z_mutated)
     (perturbed vectors snap to nearest codebook entries)
  4. Decode → entirely new carbonate rock CT slice
"""

import os
import numpy as np
import tensorflow as tf
import scipy.ndimage as ndimage
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

# Mutation strengths — gradually increasing perturbation intensity
MUTATION_STRENGTHS = [0.5, 1.0, 1.5, 2.0, 2.5]


def load_model():
    """Build and load epoch-50 final weights."""
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    # Init computation graph
    dummy = tf.zeros([1, 256, 256, 1])
    _ = decoder(vq_layer(encoder(dummy)))

    encoder.load_weights(os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    decoder.load_weights(os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"))
    vq_embedding = np.load(os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"))
    vq_layer.set_weights([vq_embedding])

    print(f"[INFO] Epoch {EPOCH_TO_LOAD} weights loaded successfully.")
    return encoder, decoder, vq_layer


def main():
    # ── 1. Load model ──
    encoder, decoder, vq_layer = load_model()

    # ── 2. Pick single parent image (Image A only) ──
    print("[INFO] Picking parent image from data pipeline...")
    dataset, _ = create_dataset(DATA_DIR, batch_size=16)
    batch, _ = next(iter(dataset))
    batch_np = batch.numpy()

    # Pick the most porous (darkest) slice as parent — more room for mutation
    means = np.mean(batch_np, axis=(1, 2, 3))
    idx_a = int(np.argmin(means))
    img_a_np = batch_np[idx_a:idx_a + 1]  # (1, 256, 256, 1)
    print(f"[INFO] Parent Image A selected (mean={means[idx_a]:.4f})")

    img_a = tf.constant(img_a_np)

    # ── 3. Extract continuous latent feature Z_eA (BEFORE VQ!) ──
    # We must bypass the VQ call inside encoder; encoder outputs Z_e directly.
    # build_encoder returns: inputs -> Conv layers -> Conv2D(latent_dim) = Z_e
    z_ea = encoder(img_a, training=False)  # (1, 32, 32, 64)
    b, h, w, c = z_ea.shape
    print(f"[INFO] Continuous latent Z_e shape: (1, {h}, {w}, {c})")

    # ── 4. Organic gene mutation + re-quantization ──
    print(f"[INFO] Generating {len(MUTATION_STRENGTHS)} novel slices via Latent Mutation...")
    novel_images = []

    for idx, strength in enumerate(MUTATION_STRENGTHS):
        # a. Generate random Gaussian noise in latent space
        noise = np.random.randn(h, w, c).astype(np.float32)

        # b. Gaussian blur — spatially smooth (sigma=[1.5, 1.5, 0])
        #    Channel axis (last) is NOT blurred — keeps feature semantics intact
        organic_noise = ndimage.gaussian_filter(noise, sigma=[1.5, 1.5, 0])

        # c. Convert to tensor, add batch dim
        organic_noise_tf = tf.convert_to_tensor(organic_noise, dtype=tf.float32)
        organic_noise_tf = tf.expand_dims(organic_noise_tf, axis=0)  # (1, h, w, c)

        # d. Gene mutation: add structured perturbation to continuous features
        z_mutated = z_ea + strength * organic_noise_tf

        # e. Magic moment: re-quantize through VQ layer
        #    Perturbed vectors get snapped to nearest valid codebook entry.
        #    Background (close to original entries) stays stable;
        #    Pore-edge features get pushed to NEW codebook entries = NOVEL pores!
        z_q_novel = vq_layer(z_mutated)

        # f. Decode to image space
        generated = decoder(z_q_novel, training=False)
        novel_img = np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)
        novel_images.append(novel_img)

        print(f"  Mutation {idx+1}/{len(MUTATION_STRENGTHS)} | "
              f"strength={strength:.1f}")

    # ── 5. Render 1×6 grid ──
    print("[INFO] Rendering 1x6 visualization grid...")
    n_cols = len(MUTATION_STRENGTHS) + 1  # 6 columns
    all_images = [img_a_np[0, :, :, 0]] + novel_images
    titles = ['Real Image A\n(Parent)'] + \
             [f'Mutation\nstrength={s:.1f}' for s in MUTATION_STRENGTHS]

    fig, axes = plt.subplots(1, n_cols, figsize=(20, 5))

    for j, ax in enumerate(axes.flat):
        ax.imshow(all_images[j], cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(titles[j], fontsize=11, fontweight='bold',
                     color='darkorange' if j > 0 else 'limegreen')
        ax.set_xticks([])
        ax.set_yticks([])

        # Border: green for real parent, orange for AI-generated mutations
        edge_color = 'limegreen' if j == 0 else 'darkorange'
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(3 if j == 0 else 2)

    plt.suptitle(
        'VQ-VAE Novel Digital Rock Generation — Single-Image Latent Mutation (Epoch 50)',
        fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(EVAL_DIR, "novel_generations.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DONE] Novel rock generations saved to:\n  {output_path}")


if __name__ == "__main__":
    main()
