"""Quick test: load standard VAE weights and do one reconstruction."""

from thesis_code.config import PROJECT_ROOT
import os, sys, time

import numpy as np
import tensorflow as tf
from PIL import Image
from thesis_code.models.standard_vae import build_standard_encoder, build_standard_decoder

gpus = tf.config.list_physical_devices('GPU')
print(f"[GPU] {len(gpus)} device(s)")

# Build models
encoder = build_standard_encoder(latent_dim=128)
decoder = build_standard_decoder(latent_dim=128)

# Build by calling once
_ = encoder(tf.zeros([1, 256, 256, 1]))
_ = decoder(tf.zeros([1, 128]))

# Print weight variable names for debugging
print("\n[ENCODER weight paths]")
for w in encoder.weights:
    print(f"  {w.name}  shape={w.shape}")
print("\n[DECODER weight paths]")
for w in decoder.weights:
    print(f"  {w.name}  shape={w.shape}")

# Load weights
ckpt_dir = os.path.join(PROJECT_ROOT, 'results', 'checkpoints')
tf.print("\n[INFO] Loading encoder weights...")
encoder.load_weights(os.path.join(ckpt_dir, 'vae_carbonate.weights.h5'), by_name=True)
tf.print("\n[INFO] Loading decoder weights...")
decoder.load_weights(os.path.join(ckpt_dir, 'vae_carbonate.weights.h5'), by_name=True)
print("\n[OK] Standard VAE weights loaded!")

# Test reconstruction on one image
data_dir = os.path.join(PROJECT_ROOT, 'data', 'real_images')
tif_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                    if f.lower().endswith(('.tif', '.tiff'))])
fp = tif_files[0]
print(f"Testing on: {os.path.basename(fp)}")

img = np.array(Image.open(fp), dtype=np.float32)
if len(img.shape) == 2:
    img = np.expand_dims(img, axis=-1)
img_t = tf.image.resize(img, [256, 256])
min_v = tf.reduce_min(img_t)
max_v = tf.reduce_max(img_t)
img_n = tf.math.divide_no_nan(img_t - min_v, max_v - min_v)

inp = tf.convert_to_tensor(img_n[np.newaxis, ...], dtype=tf.float32)
z_mean, z_log_var = encoder(inp, training=False)
recon = decoder(z_mean, training=False)
recon_np = recon.numpy()[0, :, :, 0]

# Also test with sampling
z_sampled = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(tf.shape(z_mean))
recon_sampled = decoder(z_sampled, training=False)
recon_sampled_np = recon_sampled.numpy()[0, :, :, 0]

print(f"  Input range: [{img_n.numpy().min():.4f}, {img_n.numpy().max():.4f}]")
print(f"  Recon range: [{recon_np.min():.4f}, {recon_np.max():.4f}]")
print(f"  Recon sampled range: [{recon_sampled_np.min():.4f}, {recon_sampled_np.max():.4f}]")
print(f"  MSE (z_mean): {np.mean((img_n.numpy()[:,:,0] - recon_np)**2):.6f}")
print(f"  KL divergence approx: {tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)):.4f}")
print("[DONE] Standard VAE reconstruction works!")
