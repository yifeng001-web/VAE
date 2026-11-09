"""Diagnose standard VAE collapse in figure_4_2."""

from thesis_code.config import PROJECT_ROOT
import os, sys

import numpy as np
import tensorflow as tf
from PIL import Image
from thesis_code.models.standard_vae import build_standard_encoder, build_standard_decoder
from thesis_code.models.manual_load_vae import manual_load_weights

gpus = tf.config.list_physical_devices('GPU')
print(f'[GPU] {len(gpus)} device(s)')

CKPT_PATH = os.path.join(PROJECT_ROOT,
                         'results', 'checkpoints', 'vae_carbonate.weights.h5')
DATA_DIR = os.path.join(PROJECT_ROOT,
                        'data', 'real_images')

# Build models
encoder = build_standard_encoder(latent_dim=128)
decoder = build_standard_decoder(latent_dim=128)
_ = encoder(tf.zeros([1, 256, 256, 1]))
_ = decoder(tf.zeros([1, 128]))

print('\n=== Loading ENCODER weights ===')
ea, em = manual_load_weights(encoder, CKPT_PATH, 'encoder')
print(f'\n=== Loading DECODER weights ===')
da, dm = manual_load_weights(decoder, CKPT_PATH, 'decoder')

# Test on 5 images
tif_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                    if f.lower().endswith(('.tif', '.tiff'))])
np.random.seed(42)
test_files = np.random.choice(tif_files, min(5, len(tif_files)), replace=False)

print('\n=== Testing on 5 images ===')
first_recon = None
first_z = None

for i, fp in enumerate(test_files):
    fname = os.path.basename(fp)
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
    recon_s = decoder(z_sampled, training=False)
    recon_s_np = recon_s.numpy()[0, :, :, 0]
    
    mse_det = np.mean((img_n.numpy()[:,:,0] - recon_np) ** 2)
    mse_sam = np.mean((img_n.numpy()[:,:,0] - recon_s_np) ** 2)
    
    print(f'  [{i+1}] {fname}')
    print(f'      Input:  min={img_n.numpy().min():.4f} max={img_n.numpy().max():.4f} mean={img_n.numpy().mean():.4f}')
    print(f'      Recon:  min={recon_np.min():.4f} max={recon_np.max():.4f} mean={recon_np.mean():.4f}')
    print(f'      ReconS: min={recon_s_np.min():.4f} max={recon_s_np.max():.4f} mean={recon_s_np.mean():.4f}')
    print(f'      MSE(det)={mse_det:.6f}  MSE(sam)={mse_sam:.6f}')
    print(f'      z_mean: min={z_mean.numpy().min():.4f} max={z_mean.numpy().max():.4f} std={z_mean.numpy().std():.4f}')
    print(f'      z_log_var: min={z_log_var.numpy().min():.4f} max={z_log_var.numpy().max():.4f} mean={z_log_var.numpy().mean():.4f}')
    
    if i == 0:
        first_recon = recon_np.copy()
        first_z = z_mean.numpy().copy()
    else:
        recon_diff = np.mean((first_recon - recon_np) ** 2)
        z_diff = np.mean((first_z - z_mean.numpy()) ** 2)
        print(f'      vs sample1: recon_diff={recon_diff:.6f} z_diff={z_diff:.6f}')

print('\n[DONE]')
