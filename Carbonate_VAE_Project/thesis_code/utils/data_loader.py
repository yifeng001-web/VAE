import os
import numpy as np
import tensorflow as tf
from PIL import Image


def decode_tif(file_path):
    """
    使用 PIL 直接从磁盘路径解码 TIFF 为 float32 numpy 数组 (H, W, 1)
    此函数通过 tf.py_function 桥接到 TensorFlow 计算图中。
    绕开 TF 原生 decode_image（不支持 16-bit TIFF），使用 PIL 作为可靠解码器。
    """
    path = file_path.numpy().decode('utf-8')
    img = Image.open(path)
    img_array = np.array(img, dtype=np.float32)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    return img_array


def process_path(file_path):
    """
    核心的图片读取与动态对比度拉伸（Min-Max Scaling）函数
    专门用于拯救低对比度的碳酸盐岩 CT 切片
    """
    # 1. 使用 PIL 通过 tf.py_function 解码 TIFF（TF 原生 decode_image 不支持 TIFF）
    img = tf.py_function(func=decode_tif, inp=[file_path], Tout=tf.float32)
    # 锁定形状，让 TF 计算图知道维度信息（H, W, 1），防止后续 resize 报错
    img.set_shape([None, None, 1])

    # 2. 统一尺寸为 256x256
    img = tf.image.resize(img, [256, 256])

    # 3. 核心防线：动态 Min-Max 对比度拉伸
    # 获取当前这张切片的极小值（孔隙/噪点）和极大值（岩石骨架）
    min_val = tf.reduce_min(img)
    max_val = tf.reduce_max(img)

    # 公式：(当前像素 - 最小值) / (最大值 - 最小值)
    # 作用：强制把整张图最黑的地方拉到 0.0，最亮的地方拉到 1.0，彻底暴露孔隙细节！
    # 使用 divide_no_nan 防止极其罕见的"整张图全是一种纯色"导致的除以 0 崩溃
    img_normalized = tf.math.divide_no_nan(
        img - min_val,
        max_val - min_val
    )

    # 4. 锁定张量形状，防止后续 Batch 拼接时报错
    img_normalized.set_shape([256, 256, 1])

    # VAE 模型的输入 (x) 和目标 (y) 都是图片本身
    return img_normalized, img_normalized


def create_dataset(data_dir, batch_size=4):
    """
    构建高效的异步数据管道
    注意：已显式禁用 cache()，确保每轮 Epoch 重新从磁盘读取，
    杜绝 tf.py_function + cache 序列化不稳定导致的数据截断假象。

    文件搜索策略：使用 os.walk 遍历 data_dir 下所有子目录，
    收集 .tif / .TIF / .tiff / .TIFF 文件的绝对路径。
    彻底弃用 glob（存在匹配遗漏风险）。
    """
    # 使用 os.walk 遍历所有子目录，收集所有 TIFF 格式文件
    TARGET_EXTENSIONS = {'.tif', '.TIF', '.tiff', '.TIFF'}
    file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in TARGET_EXTENSIONS:
                file_list.append(os.path.join(root, f))

    num_files = len(file_list)

    # ===== 打印横幅：真实文件统计 =====
    print("\n" + "=" * 60)
    print(f"  [DATA] 真实找到的 TIF 图片总数: {num_files}")
    print("=" * 60 + "\n")

    if num_files == 0:
        raise ValueError(f"[FATAL] 未在 {data_dir} 找到任何 TIF 文件！请检查路径。")

    # 防爆墙：如果找到的图片少于 4000 张，直接报错停止
    if num_files < 4000:
        raise ValueError(
            f"[FATAL] 读取异常！应该有 4000+ 张图片，"
            f"但只找到了 {num_files} 张！请立即检查数据目录和匹配规则！"
        )

    print(f" [DATA] Steps per epoch (batch={batch_size}): {num_files // batch_size}")

    # 1. 创建文件路径数据集
    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    # 2. 全局打乱 (buffer_size 越大打乱越彻底)
    dataset = dataset.shuffle(buffer_size=2000)

    # 3. 多线程并行读取和动态拉伸处理 (PIL 解码 + Min-Max 拉伸)
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. [已禁用] 内存缓存 — 防止 tf.py_function 桥接序列化不稳定
    # dataset = dataset.cache()

    # 5. 批处理与异步预取
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_files