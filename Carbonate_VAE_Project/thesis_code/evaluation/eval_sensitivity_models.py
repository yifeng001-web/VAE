"""
eval_sensitivity_models.py — 评估不同 lambda_porosity 模型的测试集 MAE 和孔隙度偏差
==================================================================================
读取已训练的模型权重，在测试集 (100 张切片) 上计算：
  - MAE (像素重构误差)
  - 孔隙度绝对偏差 |porosity_GT - porosity_recon|
"""

import os
import sys
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer
from PIL import Image

from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
BASELINE_CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "baseline_checkpoints")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
IMAGE_SIZE = 256
EPOCH_TO_LOAD = 50
NUM_TEST = 200  # 测试集大小


def gather_all_tif_paths(data_dir):
    TARGET_EXT = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXT:
                file_list.append(os.path.join(root, f))
    return file_list


def load_and_normalize_tif(filepath):
    img = np.array(Image.open(filepath), dtype=np.float32)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_t = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    min_val = tf.reduce_min(img_t)
    max_val = tf.reduce_max(img_t)
    return tf.math.divide_no_nan(img_t - min_val, max_val - min_val).numpy()


def load_model(encoder_path, decoder_path, vq_path):
    encoder = build_encoder(latent_dim=LATENT_DIM)
    decoder = build_decoder(latent_dim=LATENT_DIM)
    vq_layer = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)
    _ = encoder(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]))
    _ = decoder(tf.zeros([1, 32, 32, LATENT_DIM]))
    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    vq_embedding = np.load(vq_path)
    vq_layer.set_weights([vq_embedding])
    return encoder, decoder, vq_layer


def reconstruct(encoder, decoder, vq_layer, img_array):
    input_tensor = tf.convert_to_tensor(img_array[np.newaxis, ...], dtype=tf.float32)
    z_e = encoder(input_tensor, training=False)
    quantized_z = vq_layer(z_e)
    recon = decoder(quantized_z, training=False)
    return np.clip(recon.numpy()[0, :, :, 0], 0.0, 1.0)


def evaluate_model(encoder, decoder, vq_layer, test_paths, label):
    """在测试集上评估 MAE 和孔隙度绝对偏差."""
    mae_list = []
    porosity_dev_list = []

    for i, fp in enumerate(test_paths):
        img = load_and_normalize_tif(fp)
        orig_2d = img[..., 0]
        recon = reconstruct(encoder, decoder, vq_layer, img)

        mae = np.mean(np.abs(orig_2d - recon))
        porosity_gt = 1.0 - float(orig_2d.mean())
        porosity_recon = 1.0 - float(recon.mean())
        porosity_dev = abs(porosity_gt - porosity_recon)

        mae_list.append(mae)
        porosity_dev_list.append(porosity_dev)

        if (i + 1) % 50 == 0:
            print(f"  {label} eval: {i+1}/{len(test_paths)}")

    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    porosity_dev_mean = np.mean(porosity_dev_list)
    porosity_dev_std = np.std(porosity_dev_list)

    print(f"\n  [{label} 结果] (N={len(test_paths)})")
    print(f"    MAE                = {mae_mean:.6f} +/- {mae_std:.6f}")
    print(f"    孔隙度绝对偏差       = {porosity_dev_mean:.6f} +/- {porosity_dev_std:.6f}")

    return {
        'mae_mean': mae_mean,
        'mae_std': mae_std,
        'porosity_dev_mean': porosity_dev_mean,
        'porosity_dev_std': porosity_dev_std,
    }


def main():
    # ── 准备测试集 (固定种子 999) ──
    all_paths = gather_all_tif_paths(DATA_DIR)
    np.random.seed(999)
    np.random.shuffle(all_paths)
    test_paths = all_paths[:NUM_TEST]
    print(f"[INFO] 测试集: {len(test_paths)} 张切片 (固定种子 999)")

    results = {}

    # ── 评估 λ=0 (Baseline) ──
    print("\n" + "=" * 60)
    print("  评估 lambda_porosity = 0.0 (Baseline)")
    print("=" * 60)
    b_enc, b_dec, b_vq = load_model(
        os.path.join(BASELINE_CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(BASELINE_CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(BASELINE_CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"),
    )
    results[0.0] = evaluate_model(b_enc, b_dec, b_vq, test_paths, "lambda=0.0")

    # ── 评估 λ=20 (Proposed) ──
    print("\n" + "=" * 60)
    print("  评估 lambda_porosity = 20.0 (Proposed)")
    print("=" * 60)
    p_enc, p_dec, p_vq = load_model(
        os.path.join(CHECKPOINT_DIR, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5"),
        os.path.join(CHECKPOINT_DIR, f"vq_layer_ep{EPOCH_TO_LOAD}.npy"),
    )
    results[20.0] = evaluate_model(p_enc, p_dec, p_vq, test_paths, "lambda=20.0")

    # ── 保存结果 ──
    output_path = os.path.join(RESULTS_DIR, "evaluation", "sensitivity_eval.npz")
    # Convert to savable format
    save_dict = {}
    for lam, r in results.items():
        key = f"lam_{lam}"
        save_dict[f"{key}_mae_mean"] = r['mae_mean']
        save_dict[f"{key}_mae_std"] = r['mae_std']
        save_dict[f"{key}_por_dev_mean"] = r['porosity_dev_mean']
        save_dict[f"{key}_por_dev_std"] = r['porosity_dev_std']
    np.savez(output_path, **save_dict)
    print(f"\n[DONE] 评估结果已保存至: {output_path}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("  当前汇总")
    print("=" * 60)
    for lam in sorted(results.keys()):
        r = results[lam]
        print(f"  lambda={lam:5.1f}  MAE={r['mae_mean']:.6f}+/-{r['mae_std']:.6f}  "
              f"PorDev={r['porosity_dev_mean']:.6f}+/-{r['porosity_dev_std']:.6f}")


if __name__ == "__main__":
    main()
