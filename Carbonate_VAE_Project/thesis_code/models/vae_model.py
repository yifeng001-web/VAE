import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VectorQuantizer(layers.Layer):
    """
    终极防爆版离散码本 (Codebook)
    已修复 TensorFlow 动态维度广播 (AddV2) 崩溃的 Bug
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        # 初始化离散码本
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        input_shape = tf.shape(x)
        # 1. 展平输入，锁定最后一个维度
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # 2. 核心防爆修复区 (AddV2 广播修复)
        # similarity 形状: (N, M)
        similarity = tf.matmul(flattened, self.embeddings)
        
        # flat_sq 形状强制锁定为列向量: (N, 1)
        flat_sq = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
        
        # emb_sq 形状强制重塑为行向量: (1, M)
        # 这里是解决你报错的核心！绝对不给 TF 猜维度的机会
        emb_sq = tf.reshape(tf.reduce_sum(self.embeddings ** 2, axis=0), [1, -1])

        # (N, 1) + (1, M) = (N, M) 完美广播矩阵加法
        distances = flat_sq + emb_sq - 2.0 * similarity

        # 3. 最邻近搜索匹配
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        
        # 4. 离散化与形状还原
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # 5. 论文核心 VQ 损失
        commitment_loss = self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # 6. 直通估计器 (STE)
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

def build_encoder(latent_dim=64):
    """带残差与层归一化的高级编码器"""
    inputs = keras.Input(shape=(256, 256, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    
    res = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.add([x, res])
    x = layers.LayerNormalization()(x)
    
    outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(inputs, outputs, name="encoder")

def build_decoder(latent_dim=64):
    """带残差重构结构的高级解码器"""
    inputs = keras.Input(shape=(32, 32, latent_dim))
    
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    
    res = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.add([x, res])
    
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    
    outputs = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="decoder")