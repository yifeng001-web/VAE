"""Verify weight loading: check if weights actually change after loading."""

from thesis_code.config import PROJECT_ROOT
import os, sys

import numpy as np
import tensorflow as tf
from thesis_code.models.standard_vae import build_standard_encoder, build_standard_decoder

encoder = build_standard_encoder(latent_dim=128)
decoder = build_standard_decoder(latent_dim=128)
_ = encoder(tf.zeros([1, 256, 256, 1]))
_ = decoder(tf.zeros([1, 128]))

# Check weight BEFORE loading
w_before = encoder.get_layer('z_mean').get_weights()[0].copy()
print(f"[BEFORE] z_mean kernel: mean={w_before.mean():.6f}, std={w_before.std():.6f}")

# Load
ckpt = os.path.join(PROJECT_ROOT, 'results', 'checkpoints', 'vae_carbonate.weights.h5')
encoder.load_weights(ckpt, by_name=True)
decoder.load_weights(ckpt, by_name=True)

# Check weight AFTER loading
w_after = encoder.get_layer('z_mean').get_weights()[0].copy()
print(f"[AFTER]  z_mean kernel: mean={w_after.mean():.6f}, std={w_after.std():.6f}")
print(f"[CHANGE] max abs diff: {np.max(np.abs(w_after - w_before)):.6f}")

# Check a few more
for lname in ['enc_conv1', 'enc_dense', 'z_log_var']:
    w = encoder.get_layer(lname).get_weights()[0]
    print(f"  {lname}: mean={w.mean():.6f}, std={w.std():.6f}")

for lname in ['dec_dense', 'dec_output']:
    w = decoder.get_layer(lname).get_weights()[0]
    print(f"  {lname}: mean={w.mean():.6f}, std={w.std():.6f}")

# Check BN layers
for lname in ['enc_bn1', 'dec_bn_out']:
    layer = encoder if 'enc' in lname else decoder
    bn = layer.get_layer(lname)
    gamma = bn.get_weights()[0]
    beta = bn.get_weights()[1]
    mean = bn.get_weights()[2]
    var = bn.get_weights()[3]
    print(f"  {lname}: gamma_mean={gamma.mean():.4f}, beta_mean={beta.mean():.4f}, moving_mean={mean.mean():.4f}, moving_var={var.mean():.4f}")
