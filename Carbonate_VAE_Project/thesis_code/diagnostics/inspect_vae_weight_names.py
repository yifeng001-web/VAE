"""
Quick diagnostic: print both h5 keys and model variable names side-by-side
to identify the exact mismatch pattern.
"""

from thesis_code.config import PROJECT_ROOT
import os, sys

import h5py
import tensorflow as tf
from thesis_code.models.standard_vae import build_standard_encoder, build_standard_decoder

CKPT_PATH = os.path.join(PROJECT_ROOT,
                         'results', 'checkpoints', 'vae_carbonate.weights.h5')
print(f"CKPT: {CKPT_PATH}")
print(f"Exists: {os.path.exists(CKPT_PATH)}")

# Build models and call once to create weights
encoder = build_standard_encoder(latent_dim=128)
decoder = build_standard_decoder(latent_dim=128)
_ = encoder(tf.zeros([1, 256, 256, 1]))
_ = decoder(tf.zeros([1, 128]))

# Collect h5 keys (from visititems, these are relative to the visited group)
h5_keys = {}
with h5py.File(CKPT_PATH, 'r') as f:
    def collect(name, obj):
        if hasattr(obj, 'shape'):
            h5_keys[name] = obj.shape
    f['encoder'].visititems(collect)

# Collect model variable names
model_vars = {}
for var in encoder.weights:
    name = var.name
    if name.endswith(':0'):
        name = name[:-2]
    model_vars[name] = var.shape

print("\n" + "=" * 80)
print("H5 KEYS (from encoder visititems - these are the bare keys):")
for k, v in sorted(h5_keys.items()):
    print(f"  h5:  {k:60s} shape={v}")

print("\n" + "=" * 80)
print("MODEL VARIABLE NAMES (encoder.weights[i].name without :0):")
for k, v in sorted(model_vars.items()):
    print(f"  var: {k:60s} shape={v}")

# Compare
h5_set = set(h5_keys.keys())
model_set = set(model_vars.keys())

print("\n" + "=" * 80)
print(f"H5 keys count: {len(h5_set)}")
print(f"Model var count: {len(model_set)}")
print(f"In h5 but NOT in model: {sorted(h5_set - model_set)}")
print(f"In model but NOT in h5: {sorted(model_set - h5_set)}")
print(f"In both (intersection): {len(h5_set & model_set)}")

if h5_set == model_set:
    print("\n*** KEYS MATCH EXACTLY! ***")
else:
    print("\n*** MISMATCH DETECTED ***")
    # Show first few differences
    only_h5 = sorted(h5_set - model_set)
    only_model = sorted(model_set - h5_set)
    for i in range(min(5, max(len(only_h5), len(only_model)))):
        h = only_h5[i] if i < len(only_h5) else "(none)"
        m = only_model[i] if i < len(only_model) else "(none)"
        print(f"  h5: {h:50s}  |  model: {m}")
