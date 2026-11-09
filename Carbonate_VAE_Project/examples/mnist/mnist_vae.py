import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义采样层 (Reparameterization Trick)
# 这是 VAE 的核心：从均值和方差中抽取潜在变量 z
# ==========================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # 核心公式：z = mu + sigma * epsilon
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ==========================================
# 2. 构建编码器 (Encoder)
# 把 28x28 的图片压缩成两个向量：z_mean 和 z_log_var
# ==========================================s
latent_dim = 2  # 潜在空间维度，设为2是为了方便在二维平面画图

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# ==========================================
# 3. 构建解码器 (Decoder)
# 把潜在变量 z 还原回 28x28 的图片
# ==========================================
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# ==========================================
# 4. 定义 VAE 模型并重写 train_step
# 这是为了自定义损失函数 (重构损失 + KL散度)
# ==========================================
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        
        # 👇 就是这两行！必须把传入的 encoder 和 decoder 绑定到 self 上
        self.encoder = encoder 
        self.decoder = decoder 
        
        # 下面是损失函数的追踪器...
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # 前向传播
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # 计算重构损失 (二元交叉熵)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            # 计算 KL 散度损失 (公式参考书中 8.4.3)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        # 计算梯度并更新权重
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # 更新监控指标
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# ==========================================
# 5. 准备数据并开始训练
# ==========================================
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
# 训练 30 轮，为了快速演示可以改成 10
vae.fit(mnist_digits, epochs=30, batch_size=128)

# ==========================================
# 6. 结果可视化：生成潜在空间的 2D 网格图
# 对应书中图 8-14 的效果
# def plot_latent_space(vae, n=15, digit_size=28):
#     scale = 1.0
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # 在潜在空间构建坐标网格
#     grid_x = np.linspace(-scale, scale, n)
#     grid_y = np.linspace(-scale, scale, n)[::-1]

#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = vae.decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[i * digit_size: (i + 1) * digit_size,
#                    j * digit_size: (j + 1) * digit_size] = digit

#     plt.figure(figsize=(10, 10))
#     plt.imshow(figure, cmap="Greys_r")
#     plt.show()

# plot_latent_space(vae)
# ==========================================
def plot_latent_space(vae, n=15, digit_size=28):
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    # 1. 优化：不再使用嵌套循环一个个预测，而是先把所有的坐标收集起来
    z_samples = []
    for yi in grid_y:
        for xi in grid_x:
            z_samples.append([xi, yi])
    z_samples = np.array(z_samples)  # 变成一个 shape 为 (225, 2) 的大矩阵

    # 2. 一次性让解码器生成所有 225 张图片！(告别 Retracing 警告，速度极快)
    # verbose=0 是为了不让控制台打印出 225 次的进度条
    x_decoded = vae.decoder.predict(z_samples, verbose=0)

    # 3. 把生成好的 225 张图片，像贴瓷砖一样贴到大图上
    idx = 0
    for i in range(n):
        for j in range(n):
            digit = x_decoded[idx].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
            idx += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

# 运行画图函数
plot_latent_space(vae)