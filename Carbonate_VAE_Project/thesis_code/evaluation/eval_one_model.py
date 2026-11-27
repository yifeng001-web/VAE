"""
eval_one_model.py — 单模型抽样重构评估 (MAE + 平均灰度差)
用法: python eval_one_model.py --ckpt_dir PATH --label "lambda=10"
从训练数据目录抽取至多 200 张切片，未排除训练样本，不是独立测试。
保留 por_dev 历史字段名，其含义为平均灰度差，不是物理孔隙度误差。
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float32')

from thesis_code.models.vae_model import build_encoder, build_decoder, VectorQuantizer

from thesis_code.config import PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "CT")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

NUM_EMBEDDINGS = 512
LATENT_DIM = 64
IMAGE_SIZE = 256
EPOCH_TO_LOAD = 50
NUM_TEST = 200  # 历史变量名：抽样重构评估数量


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Checkpoint directory containing encoder/decoder/vq weights')
    parser.add_argument('--label', type=str, default='model',
                        help='Label for this model')
    args = parser.parse_args()

    # 抽样重构评估；未排除训练样本 (固定种子)
    all_paths = gather_all_tif_paths(DATA_DIR)
    np.random.seed(999)
    np.random.shuffle(all_paths)
    test_paths = all_paths[:NUM_TEST]

    # 加载模型
    enc_path = os.path.join(args.ckpt_dir, f"encoder_ep{EPOCH_TO_LOAD}.weights.h5")
    dec_path = os.path.join(args.ckpt_dir, f"decoder_ep{EPOCH_TO_LOAD}.weights.h5")
    vq_path = os.path.join(args.ckpt_dir, f"vq_layer_ep{EPOCH_TO_LOAD}.npy")
    encoder, decoder, vq_layer = load_model(enc_path, dec_path, vq_path)

    print(f"\n[{args.label}] 对 {len(test_paths)} 张切片进行重构评估...")
    print("[NOTE] 样本来自训练数据目录，未排除训练样本；结果不代表独立测试性能。")

    mae_list = []
    por_dev_list = []

    for i, fp in enumerate(test_paths):
        img = load_and_normalize_tif(fp)
        orig_2d = img[..., 0]
        recon = reconstruct(encoder, decoder, vq_layer, img)
        mae_list.append(np.mean(np.abs(orig_2d - recon)))
        por_gt = 1.0 - float(orig_2d.mean())
        por_recon = 1.0 - float(recon.mean())
        por_dev_list.append(abs(por_gt - por_recon))
        if (i + 1) % 50 == 0:
            print(f"  {args.label}: {i+1}/{len(test_paths)}")

    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    por_dev_mean = np.mean(por_dev_list)
    por_dev_std = np.std(por_dev_list)

    print(f"\n  [{args.label} 结果]")
    print(f"    MAE                = {mae_mean:.6f} +/- {mae_std:.6f}")
    print(f"    平均灰度差          = {por_dev_mean:.6f} +/- {por_dev_std:.6f}")

    # 保存
    out_path = os.path.join(args.ckpt_dir, "eval_result.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"label={args.label}\n")
        f.write(f"mae_mean={mae_mean:.8f}\n")
        f.write(f"mae_std={mae_std:.8f}\n")
        f.write(f"por_dev_mean={por_dev_mean:.8f}\n")
        f.write(f"por_dev_std={por_dev_std:.8f}\n")
    print(f"  结果已保存至: {out_path}")

    # Also save as npz for convenience
    np.savez(os.path.join(args.ckpt_dir, "eval_result.npz"),
             mae_mean=mae_mean, mae_std=mae_std,
             por_dev_mean=por_dev_mean, por_dev_std=por_dev_std)


if __name__ == "__main__":
    main()
