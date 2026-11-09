"""
Standard VAE (Variational Autoencoder) with CBAM attention.
Built to match the architecture in vae_carbonate.weights.h5.

Key differences from VQ-VAE:
  - Continuous latent space (z_mean, z_log_var) with reparameterization trick
  - KL divergence regularization (encourages smooth latent manifold)
  - CBAM attention in both encoder and decoder
  - No discrete codebook — leads to Gaussian blur on fine details

Architecture:
  Encoder: Input(256,256,1) → 4×[Conv2D(stride=2)+BN+LeakyReLU+CBAM] → Flatten(65536)
           → Dense(512,relu) → z_mean(128), z_log_var(128)
  Decoder: Input(128) → Dense(65536,relu) → Reshape(16,16,256)
           → 4×[Conv2D+BN+LeakyReLU+CBAM+UpSampling2D] → Conv_out(16)+BN+LeakyReLU
           → Output(1,sigmoid)

Weight file uses FLAT naming (no nested Keras sub-layer scoping):
  enc_cbam1_ca_dense1_avg, enc_cbam1_sa_conv, etc.
To match this, CBAM operations are INLINED in the Functional API
using named Keras Dense/Conv2D layers directly.
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# ============================================================
# Inline CBAM helper (uses Functional API named layers)
# ============================================================
def _cbam_block(x, channels, name_prefix):
    """
    Apply Channel + Spatial attention inline.
    Uses Keras Dense and Conv2D layers with flat names
    matching vae_carbonate.weights.h5 naming convention.
    """
    reduction = max(1, channels // 8)

    # --- Channel Attention ---
    avg_pool = tf.reduce_mean(x, axis=[1, 2])                          # [B, C]
    max_pool = tf.reduce_max(x, axis=[1, 2])                           # [B, C]

    # Avg path
    ca_avg = layers.Dense(reduction, activation='relu',
                          name=f'{name_prefix}_ca_dense1_avg')(avg_pool)
    ca_avg = layers.Dense(channels, activation='sigmoid',
                          name=f'{name_prefix}_ca_dense2_avg')(ca_avg)
    # Max path
    ca_max = layers.Dense(reduction, activation='relu',
                          name=f'{name_prefix}_ca_dense1_max')(max_pool)
    ca_max = layers.Dense(channels, activation='sigmoid',
                          name=f'{name_prefix}_ca_dense2_max')(ca_max)

    ca_scale = tf.expand_dims(tf.expand_dims(ca_avg + ca_max, 1), 1)  # [B, 1, 1, C]
    x = x * ca_scale

    # --- Spatial Attention ---
    sa_avg = tf.reduce_mean(x, axis=-1, keepdims=True)                 # [B, H, W, 1]
    sa_max = tf.reduce_max(x, axis=-1, keepdims=True)                  # [B, H, W, 1]
    sa_cat = tf.concat([sa_avg, sa_max], axis=-1)                      # [B, H, W, 2]
    sa_scale = layers.Conv2D(1, 7, padding='same', activation='sigmoid', use_bias=False,
                             name=f'{name_prefix}_sa_conv')(sa_cat)    # [B, H, W, 1]
    x = x * sa_scale
    return x


# ============================================================
# Standard VAE Encoder
# ============================================================
def build_standard_encoder(latent_dim=128):
    """
    Encoder matching vae_carbonate.weights.h5 encoder structure.
    CBAM operations are inlined as named Keras layers.
    """
    inputs = layers.Input(shape=(256, 256, 1), name='encoder_input')

    # Block 1: 256→128, channels: 1→32
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name='enc_conv1')(inputs)
    x = layers.BatchNormalization(name='enc_bn1')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = _cbam_block(x, channels=32, name_prefix='enc_cbam1')

    # Block 2: 128→64, channels: 32→64
    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False, name='enc_conv2')(x)
    x = layers.BatchNormalization(name='enc_bn2')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = _cbam_block(x, channels=64, name_prefix='enc_cbam2')

    # Block 3: 64→32, channels: 64→128
    x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False, name='enc_conv3')(x)
    x = layers.BatchNormalization(name='enc_bn3')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = _cbam_block(x, channels=128, name_prefix='enc_cbam3')

    # Block 4: 32→16, channels: 128→256
    x = layers.Conv2D(256, 3, strides=2, padding='same', use_bias=False, name='enc_conv4')(x)
    x = layers.BatchNormalization(name='enc_bn4')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = _cbam_block(x, channels=256, name_prefix='enc_cbam4')

    # Flatten → Dense → z_mean / z_log_var
    x = layers.Flatten()(x)  # 16*16*256 = 65536
    x = layers.Dense(512, activation='relu', name='enc_dense')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    return tf.keras.Model(inputs, [z_mean, z_log_var], name='encoder')


# ============================================================
# Standard VAE Decoder
# ============================================================
def build_standard_decoder(latent_dim=128):
    """
    Decoder matching vae_carbonate.weights.h5 decoder structure.
    CBAM operations are inlined as named Keras layers.
    """
    inputs = layers.Input(shape=(latent_dim,), name='decoder_input')

    # Dense → Reshape
    x = layers.Dense(16 * 16 * 256, activation='relu', name='dec_dense')(inputs)
    x = layers.Reshape((16, 16, 256))(x)

    # Block 1: channels 256→256, 16→32
    x = layers.Conv2D(256, 3, padding='same', use_bias=False, name='dec_conv1')(x)
    x = layers.BatchNormalization(name='dec_bn1')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = _cbam_block(x, channels=256, name_prefix='dec_cbam1')
    x = layers.UpSampling2D(size=(2, 2))(x)

    # Block 2: channels 256→128, 32→64
    x = layers.Conv2D(128, 3, padding='same', use_bias=False, name='dec_conv2')(x)
    x = layers.BatchNormalization(name='dec_bn2')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = _cbam_block(x, channels=128, name_prefix='dec_cbam2')
    x = layers.UpSampling2D(size=(2, 2))(x)

    # Block 3: channels 128→64, 64→128
    x = layers.Conv2D(64, 3, padding='same', use_bias=False, name='dec_conv3')(x)
    x = layers.BatchNormalization(name='dec_bn3')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = _cbam_block(x, channels=64, name_prefix='dec_cbam3')
    x = layers.UpSampling2D(size=(2, 2))(x)

    # Block 4: channels 64→32, 128→256
    x = layers.Conv2D(32, 3, padding='same', use_bias=False, name='dec_conv4')(x)
    x = layers.BatchNormalization(name='dec_bn4')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = _cbam_block(x, channels=32, name_prefix='dec_cbam4')
    x = layers.UpSampling2D(size=(2, 2))(x)

    # Output head
    x = layers.Conv2D(16, 3, padding='same', use_bias=False, name='dec_conv_out')(x)
    x = layers.BatchNormalization(name='dec_bn_out')(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    outputs = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='dec_output')(x)

    return tf.keras.Model(inputs, outputs, name='decoder')


# ============================================================
# Reconstruct using Standard VAE
# ============================================================
def reconstruct_vae(encoder, decoder, img_array):
    """
    Reconstruct a single image [H, W, 1] through the standard VAE.
    Uses z_mean (no sampling noise) for deterministic reconstruction.
    """
    inp = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_mean, _ = encoder(inp, training=False)
    recon = decoder(z_mean, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)
