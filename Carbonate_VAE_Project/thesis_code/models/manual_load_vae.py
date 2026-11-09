"""
Manually load standard VAE weights from vae_carbonate.weights.h5.
The h5 visititems keys include ':0' suffix (e.g., enc_conv1/kernel:0)
while Keras model variable names (after .name stripping) don't.

This script strips ':0' from both sides and matches exactly.
"""

from thesis_code.config import PROJECT_ROOT
import os, sys

import numpy as np
import tensorflow as tf
import h5py
from thesis_code.models.standard_vae import build_standard_encoder, build_standard_decoder

CKPT_PATH = os.path.join(PROJECT_ROOT,
                         'results', 'checkpoints', 'vae_carbonate.weights.h5')


def manual_load_weights(model, h5_path, model_prefix):
    """
    Load weights from h5 by matching variable names exactly.
    
    h5 visititems keys: enc_conv1/kernel:0
    Model var names:    enc_conv1/kernel:0  (before stripping)
    
    We strip the trailing ':0' from BOTH sides for comparison,
    then assign by exact match.
    """
    # Collect h5 weights (keys from visititems, relative to model_prefix)
    h5_raw = {}
    with h5py.File(h5_path, 'r') as f:
        def collect(name, obj):
            if hasattr(obj, 'shape'):
                h5_raw[name] = obj[:]
        f[model_prefix].visititems(collect)
    
    print(f"  Found {len(h5_raw)} tensors in h5 under '{model_prefix}'")
    
    assigned = 0
    missing = 0
    
    for var in model.weights:
        var_name = var.name  # e.g., 'encoder/enc_conv1/kernel:0'
        if var_name.endswith(':0'):
            var_name = var_name[:-2]  # 'encoder/enc_conv1/kernel'
        
        # Also strip model prefix for matching
        h5_prefix = f'{model_prefix}/'
        if var_name.startswith(h5_prefix):
            var_key = var_name[len(h5_prefix):]  # 'enc_conv1/kernel'
        else:
            var_key = var_name
        
        # Try exact match against h5 keys (also with :0 stripped)
        matched = False
        for h5_key, h5_val in h5_raw.items():
            h5_key_stripped = h5_key[:-2] if h5_key.endswith(':0') else h5_key
            if h5_key_stripped == var_key:
                if h5_val.shape == tuple(var.shape):
                    var.assign(h5_val)
                    assigned += 1
                    matched = True
                    break
                else:
                    print(f"  SHAPE MISMATCH: {var_key} model={var.shape} h5={h5_val.shape}")
        
        if not matched:
            missing += 1
            print(f"  MISSING: {var_key} shape={var.shape}")
    
    print(f"  Assigned: {assigned}, Missing: {missing}")
    return assigned, missing


def main():
    encoder = build_standard_encoder(latent_dim=128)
    decoder = build_standard_decoder(latent_dim=128)
    _ = encoder(tf.zeros([1, 256, 256, 1]))
    _ = decoder(tf.zeros([1, 128]))
    
    print("=== Loading ENCODER weights ===")
    ea, em = manual_load_weights(encoder, CKPT_PATH, 'encoder')
    
    print("\n=== Loading DECODER weights ===")
    da, dm = manual_load_weights(decoder, CKPT_PATH, 'decoder')
    
    # Verify
    print("\n=== Verification (encoder key layers) ===")
    for lname in ['enc_conv1', 'enc_dense', 'z_mean', 'z_log_var']:
        w = encoder.get_layer(lname).get_weights()
        if w:
            k = w[0]
            print(f"  {lname}: mean={k.mean():.6f}, std={k.std():.6f}")
    
    for lname in ['dec_conv1', 'dec_dense', 'dec_output']:
        w = decoder.get_layer(lname).get_weights()
        if w:
            k = w[0]
            print(f"  {lname}: mean={k.mean():.6f}, std={k.std():.6f}")
    
    bn = encoder.get_layer('enc_bn1')
    bn_w = bn.get_weights()
    print(f"  enc_bn1 gamma mean={bn_w[0].mean():.4f} (should != 1.0)")
    print(f"  enc_bn1 moving_mean mean={bn_w[2].mean():.4f} (should != 0.0)")
    
    # Test reconstruction
    print("\n=== Test Reconstruction ===")
    from PIL import Image
    data_dir = os.path.join(PROJECT_ROOT,
                            'data', 'real_images')
    tifs = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.tif','.tiff'))])
    fp = os.path.join(data_dir, tifs[0])
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
    
    print(f"  Input range:  [{img_n.numpy().min():.4f}, {img_n.numpy().max():.4f}]")
    print(f"  Recon range:  [{recon_np.min():.4f}, {recon_np.max():.4f}]")
    print(f"  Recon mean:   {recon_np.mean():.6f}")
    print(f"  MSE:          {np.mean((img_n.numpy()[:,:,0] - recon_np)**2):.6f}")
    print(f"  z_mean range: [{z_mean.numpy().min():.4f}, {z_mean.numpy().max():.4f}]")
    
    if ea == 62 and em == 0 and da == 65 and dm == 0:
        print("\n*** ALL WEIGHTS LOADED SUCCESSFULLY! ***")
    else:
        print(f"\n*** WARNING: Encoder {ea}/{ea+em} assigned, Decoder {da}/{da+dm} assigned ***")


if __name__ == '__main__':
    main()
