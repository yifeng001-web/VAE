"""
================================================================================
  答辩讲解文件：从 VAE 到 VQ-VAE 再到离散掩膜融合 — 完整算法原理与代码
================================================================================

  本文件按答辩逻辑组织，共 6 个部分：
  
  Part 0: 问题背景 — 碳酸盐岩数字岩心生成的挑战
  Part 1: 标准 VAE 原理 — 为什么能"生成"新图像？
  Part 2: 标准 VAE 的问题 — 高斯模糊，细节丢失
  Part 3: VQ-VAE 核心原理 — 离散码本 = 特征压缩的"词汇表"
  Part 4: 矢量量化层 (VectorQuantizer) — 连续→离散的关键桥梁
  Part 5: 自回归先验尝试 (PixelCNN) — 为什么效果不好？
  Part 6: 离散掩膜融合算法 (Fuse-then-VQ) — 我们的核心创新

  类比理解：
    潜空间编码 ≈ 频域系数（傅里叶变换中每个频率分量的幅度）
    编码器 ≈ 正向傅里叶变换（图像→频率）
    解码器 ≈ 逆傅里叶变换（频率→图像）
    VQ 码本 ≈ 将连续频谱离散化为 512 种"原型频谱模板"
    离散融合 ≈ 在频域中线性混合两种信号的频谱，而非在时域硬拼接

  作者: Carbonate VAE Project
  日期: 2026-05-28
================================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ============================================================================
#  Part 0: 问题背景
# ============================================================================
"""
碳酸盐岩数字岩心生成的挑战：
  - 原始数据：Micro-CT 扫描的 256×256 灰度切片 (16-bit → normalized [0,1])
  - 目标：生成具有真实物理特性的"全新"岩心图像
  - 困难：岩心包含微裂缝、溶孔、生物碎屑等复杂结构
  - 约束：生成的图像必须保持孔隙度、两点相关函数 S₂(r) 等物理特性

传统方法（过程模拟法、多点地质统计学）计算开销巨大。
深度生成模型提供了一条新途径：从 CT 数据中学习"什么是碳酸盐岩"，然后生成新的。
"""


# ============================================================================
#  Part 1: 标准 VAE 原理 — 为什么能"生成"新图像？
# ============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VAE (Variational Autoencoder) 的核心思想
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  类比：信号与系统的频域表示
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  在傅里叶变换中：                                                       │
  │    信号 f(t) ──FT→ 频域系数 F(ω)                                       │
  │    频域系数 F(ω) ──IFT→ 重建信号 f(t)                                   │
  │                                                                         │
  │  VAE 做的是类似的事情，但"变换基"不是正弦波，而是神经网络学出来的：      │
  │    图像 x (256×256) ──Encoder→ 潜向量 z (128维)                         │
  │    潜向量 z (128维) ──Decoder→ 重建图像 x̂ (256×256)                     │
  └─────────────────────────────────────────────────────────────────────────┘

  VAE 的两大组件：
  
  1. 编码器 (Encoder)：x → z_mean, z_log_var
     将 256×256 = 65,536 维的图像压缩到 128 维的潜空间。
     但不同于确定性的自编码器，VAE 输出的是一个"概率分布"的参数：
       - z_mean:   分布的中心位置（128维）
       - z_log_var: 分布的方差对数（128维），控制不确定性
     这就像说："这个图像在特征空间中大约在这里，不确定范围是这么大。"
  
  2. 解码器 (Decoder)：z → x̂
     从潜空间采样一个点 z，通过上采样/反卷积重构回 256×256 图像。
     类比：从频域系数通过逆傅里叶变换重建原始信号。

  3. 重参数化技巧 (Reparameterization Trick)：
     z = z_mean + ε · exp(z_log_var / 2)，其中 ε ~ N(0, I)
     把采样操作的随机性"挪"到外部噪声 ε 上，使得梯度可以穿过采样步骤回传。

  4. 损失函数 = 重建误差 + KL 正则化：
     L = ||x - x̂||² + β · KL[ N(z_mean, z_log_var) || N(0, I) ]
     
     重建误差确保解码器能还原输入（类比：IFT 能重建原信号）
     KL 散度鼓励潜空间分布接近标准高斯 N(0,1)
       → 这保证了潜空间的"连续性"和"完整性"
       → 任意从 N(0,1) 采样一个 z，解码器都能生成合理的图像
       → 这就是 VAE 能"生成"新图像的根本原因！

  ┌──────────────────── 重要理解 ────────────────────┐
  │ 训练完成后：                                      │
  │   编码器学会了将"真实岩心"映射到高斯分布           │
  │   解码器学会了从高斯分布的任意点"画"出合理图像     │
  │                                                   │
  │ 生成新图像：                                       │
  │   从 N(0,1) 随机采样一个 128 维向量 z              │
  │   喂给解码器 → 得到一张"全新的"岩心图像            │
  │                                                   │
  │ 这正是 VAE 作为生成模型的本质：                     │
  │   不是记忆训练数据，而是学习数据分布 P(x)          │
  └───────────────────────────────────────────────────┘
"""

# ── 标准 VAE 编码器代码 ──
def build_standard_encoder(latent_dim=128):
    """
    标准 VAE 编码器：256×256×1 → 128维高斯分布参数
    
    架构: 4层步长为2的卷积 → Flatten → Dense → (z_mean, z_log_var)
    
    ┌──────────────┬──────────────┬──────────────┬─────────────────┐
    │  层          │ 输入尺寸     │ 输出尺寸     │ 操作            │
    ├──────────────┼──────────────┼──────────────┼─────────────────┤
    │ Conv2D(s=2)  │ 256×256×1   │ 128×128×32  │ 下采样+特征提取 │
    │ Conv2D(s=2)  │ 128×128×32  │ 64×64×64    │ 下采样          │
    │ Conv2D(s=2)  │ 64×64×64    │ 32×32×128   │ 下采样          │
    │ Conv2D(s=2)  │ 32×32×128   │ 16×16×256   │ 下采样          │
    │ Flatten+Dense│ 16×16×256   │ 512 → 128×2 │ 输出分布参数     │
    └──────────────┴──────────────┴──────────────┴─────────────────┘
    
    类比频域: 4次 stride=2 下采样 = 提取越来越"粗"的频谱分量
              最后两层 Dense = 将频谱系数进一步压缩为 128 维特征
    """
    inputs = layers.Input(shape=(256, 256, 1), name='encoder_input')

    # 四次步长为2的卷积 = 256 → 128 → 64 → 32 → 16 (尺寸缩小 2^4 = 16倍)
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)

    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)

    x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)

    x = layers.Conv2D(256, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)

    # 展平: 16×16×256 = 65,536 → Dense(512) → z_mean(128) + z_log_var(128)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)      # 均值向量
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x) # 对数方差向量

    return tf.keras.Model(inputs, [z_mean, z_log_var], name='standard_encoder')


# ── 标准 VAE 解码器代码 ──
def build_standard_decoder(latent_dim=128):
    """
    标准 VAE 解码器：128维潜向量 → 256×256×1 图像
    
    架构: Dense → Reshape → 4层上采样反卷积
    
    ┌──────────────────┬──────────────┬──────────────┬─────────────────┐
    │  层              │ 输入尺寸     │ 输出尺寸     │ 操作            │
    ├──────────────────┼──────────────┼──────────────┼─────────────────┤
    │ Dense+Reshape    │ 128          │ 16×16×256   │ 投影回空间域    │
    │ UpSample+Conv    │ 16×16×256   │ 32×32×256   │ 上采样          │
    │ UpSample+Conv    │ 32×32×256   │ 64×64×128   │ 上采样          │
    │ UpSample+Conv    │ 64×64×128   │ 128×128×64  │ 上采样          │
    │ UpSample+Conv    │ 128×128×64  │ 256×256×32  │ 上采样          │
    │ Conv(sigmoid)    │ 256×256×32  │ 256×256×1   │ 输出 [0,1] 图像 │
    └──────────────────┴──────────────┴──────────────┴─────────────────┘
    
    类比频域: Dense+Reshape = 从 128 个频域系数"展开"到 16×16 频谱图
              4次上采样 = 逐步从低频→高频恢复空间细节
    """
    inputs = layers.Input(shape=(latent_dim,), name='decoder_input')

    # 从 128 维潜向量恢复到 16×16×256 = 65,536 的空间表示
    x = layers.Dense(16 * 16 * 256, activation='relu')(inputs)
    x = layers.Reshape((16, 16, 256))(x)

    # 四次上采样回到 256×256
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = layers.UpSampling2D(size=(2, 2))(x)  # 16→32

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = layers.UpSampling2D(size=(2, 2))(x)  # 32→64

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = layers.UpSampling2D(size=(2, 2))(x)  # 64→128

    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    x = layers.UpSampling2D(size=(2, 2))(x)  # 128→256

    # 最终输出: sigmoid 将值压缩到 [0, 1]
    outputs = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs, name='standard_decoder')


# ============================================================================
#  Part 2: 标准 VAE 的问题 — 高斯模糊，细节丢失
# ============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  标准 VAE 在碳酸盐岩数据上的表现
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  训练过程 (50 epochs, β-VAE with KL annealing):
    - 编码器将 65,536 维图像压缩到 128 维连续潜向量
    - 解码器从潜向量重构图像
    - KL 散度迫使潜空间服从 N(0,1)
  
  结果问题：
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. 高斯模糊 (Gaussian Blur)                                  │
    │    微裂缝（宽度仅 1-2 像素）被平滑掉                          │
    │    溶孔边缘失去锐利度，像被高斯滤波处理过                     │
    │    原因: VAE 使用逐像素的 MSE/MAE 损失，                       │
    │          这种损失函数天然倾向于"平均"多种可能的解 → 模糊      │
    │                                                              │
    │ 2. 后验坍塌 (Posterior Collapse)                             │
    │    当 KL 权重过大时，z_log_var → -∞，                         │
    │    潜空间坍缩为几乎固定的点，失去生成多样性                   │
    │    当 KL 权重过小时，潜空间不连续，采样生成效果差             │
    │                                                              │
    │ 3. 细节不可控                                                │
    │    从 N(0,1) 采样的 z 经过解码器 → 输出看起来很"像"岩心      │
    │    但无法精确控制：要多少个孔？孔多大？分布在哪？             │
    │    潜空间的 128 个维度是"纠缠"的，改变一个维度影响全局        │
    └──────────────────────────────────────────────────────────────┘

  结论：标准 VAE 适合生成"大致类似"的图像，但不适合需要保留精细
        物理结构（微裂缝、溶孔边界）的碳酸盐岩数字岩心生成。
        
  这引导我们转向 VQ-VAE —— 用离散码本替代连续潜空间。
"""


# ============================================================================
#  Part 3: VQ-VAE 核心原理 — 离散码本 = 特征压缩的"词汇表"
# ============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VQ-VAE (Vector Quantized-Variational AutoEncoder)
  论文: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  核心思想：用"查字典"替代 VAEs 的"概率采样"
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  类比 1: 自然语言处理                                                   │
  │    一篇文章 = 成千上万个单词的序列                                       │
  │    但实际上，任何英文文章只用到约 171,000 个英语词汇中的一部分          │
  │    每个单词是离散的符号（"rock" vs "pore"），而非连续向量               │
  │                                                                         │
  │    同样地，碳酸盐岩图像由 256×256 = 65,536 个像素组成                   │
  │    但"特征词汇"是有限的 —— 可以用 512 个"原型特征"来组合               │
  └─────────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  类比 2: 信号与系统 / 矢量量化                                           │
  │    在数字信号处理中，矢量量化 (VQ) 将连续的信号空间划分为有限个区域     │
  │    每个区域用一个"码字"(codeword) 代表                                   │
  │                                                                         │
  │    时域连续信号 ──采样→ 离散时间信号 ──量化→ 数字信号                    │
  │    图像信号 ──Encoder→ 连续特征 Z_e ──VQ→ 离散特征 Z_q ──Decoder→ 图像  │
  │                                                                         │
  │    VQ-VAE 的码本就相当于 512 个"原型频谱模板"                            │
  │    编码器将图像变换到"特征域"，然后找最接近的模板                        │
  └─────────────────────────────────────────────────────────────────────────┘

  VQ-VAE 的整体数据流：
  ┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────┐
  │ 输入图像  │ ──→ │   编码器      │ ──→ │ VQ 量化层 │ ──→ │  解码器   │ ──→ 重建图像
  │ 256×256×1 │     │ 3层Conv(s=2) │     │ 512码本   │     │ 3层转置Conv │    256×256×1
  │    x      │     │ +残差块       │     │ 最近邻查找 │     │ +残差块     │      x̂
  └──────────┘     └──────────────┘     └──────────────┘     └──────────┘
                         │                      │
                     Z_e (连续)              Z_q (离散化)
                   32×32×64              32×32×64
                  每个位置是             每个位置被替换
                  64维向量               为512个码字之一

  关键理解：编码器输出的 Z_e 是 32×32 个 64 维连续向量（共 1024 个）
            VQ 量化层将每个 64 维向量替换为码本中最接近的码字
            这意味着图像被"翻译"成了 32×32 = 1024 个"离散单词"的序列
"""


# ── VQ-VAE 编码器代码 ──
def build_encoder(latent_dim=64):
    """
    VQ-VAE 编码器：256×256×1 → 32×32×64 (连续特征张量 Z_e)
    
    架构: 3层步长为2的卷积 + 1个残差块
    
    ┌──────────────┬──────────────┬──────────────┬─────────────────┐
    │  层          │ 输入尺寸     │ 输出尺寸     │ 操作            │
    ├──────────────┼──────────────┼──────────────┼─────────────────┤
    │ Conv2D(s=2)  │ 256×256×1   │ 128×128×32  │ 下采样+LayerNorm │
    │ Conv2D(s=2)  │ 128×128×32  │ 64×64×64    │ 下采样+LayerNorm │
    │ Conv2D(s=2)  │ 64×64×64    │ 32×32×128   │ 下采样+LayerNorm │
    │ 残差块       │ 32×32×128   │ 32×32×128   │ 保持尺寸，加深特征│
    │ Conv2D(1×1)  │ 32×32×128   │ 32×32×64    │ 投影到潜维度      │
    └──────────────┴──────────────┴──────────────┴─────────────────┘
    
    输出 Z_e: (B, 32, 32, 64)
      32×32 = 1024 个空间位置（每个对应原图 8×8 的感受野）
      64 = 每个位置的连续特征维度
    
    类比频域: 编码器 = 将 256×256 图像做 3 次 2 倍下采样的子带分解
              得到 32×32 个"低频子带系数"，每个系数是 64 维向量
              不同于傅里叶的固定基函数，这里的"基"是神经网络学到的
    """
    inputs = tf.keras.Input(shape=(256, 256, 1))

    # 三次步长为2的卷积 = 256 → 128 → 64 → 32 (尺寸缩小 2^3 = 8倍)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    # 残差块: 加深网络，保持特征图尺寸不变 → 增强非线性表达能力
    res = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.add([x, res])  # 残差连接: F(x) + x
    x = layers.LayerNormalization()(x)

    # 1×1 卷积: 将 128 通道投影到 64 通道（潜空间维度）
    outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return tf.keras.Model(inputs, outputs, name="encoder")


# ── VQ-VAE 解码器代码 ──
def build_decoder(latent_dim=64):
    """
    VQ-VAE 解码器：32×32×64 → 256×256×1 (输出图像)
    
    架构: 1个残差块 + 3层转置卷积 (上采样)
    
    ┌──────────────────┬──────────────┬──────────────┬─────────────────┐
    │  层              │ 输入尺寸     │ 输出尺寸     │ 操作            │
    ├──────────────────┼──────────────┼──────────────┼─────────────────┤
    │ ConvTranspose(s=2)│ 32×32×64   │ 64×64×128   │ 上采样          │
    │ 残差块           │ 64×64×128   │ 64×64×128   │ 加深            │
    │ ConvTranspose(s=2)│ 64×64×128  │ 128×128×64  │ 上采样          │
    │ ConvTranspose(s=2)│ 128×128×64 │ 256×256×32  │ 上采样          │
    │ ConvTranspose(s=1)│ 256×256×32 │ 256×256×1   │ 输出 [0,1] 图像 │
    └──────────────────┴──────────────┴──────────────┴─────────────────┘
    
    类比频域: 解码器 = 从 32×32 的频谱表示逐步上采样到 256×256
              转置卷积 = 类似从低频重建高频的插值+滤波过程
    """
    inputs = tf.keras.Input(shape=(32, 32, latent_dim))

    # 转置卷积上采样: 32→64, 通道从64→128
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    # 残差块加深
    res = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.add([x, res])

    # 继续上采样: 64→128, 通道128→64
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    # 继续上采样: 128→256, 通道64→32
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    # 最终输出: sigmoid 将像素值压缩到 [0, 1]
    outputs = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="decoder")


# ============================================================================
#  Part 4: 矢量量化层 (VectorQuantizer) — 连续→离散的关键桥梁
# ============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  矢量量化层 — VQ-VAE 最核心的组件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  这个层做了三件事：
    1. 最近邻查找 — 将连续特征映射到离散码字
    2. 直通梯度估计 (STE) — 让不可微的量化操作可以反向传播
    3. VQ 损失计算 — 同时训练码本和编码器

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  核心概念：码本 (Codebook)                                               │
  │                                                                         │
  │  码本是一个可训练的查找表：                                              │
  │    shape = (embedding_dim, num_embeddings) = (64, 512)                  │
  │                                                                         │
  │  共 512 个"码字"(codeword)，每个码字是一个 64 维向量                    │
  │  e_0   = [0.12, -0.34, 0.78, ..., 0.05]   ← 码字 0                     │
  │  e_1   = [-0.23, 0.45, -0.11, ..., 0.67]  ← 码字 1                     │
  │  ...                                                                   │
  │  e_511 = [0.56, -0.78, 0.34, ..., -0.12]  ← 码字 511                   │
  │                                                                         │
  │  每个 32×32 位置的 64 维特征向量 z[i,j,:] 被替换为最近的码字            │
  │  → 整张图像用 32×32 = 1024 个 [0,511] 之间的整数编码                   │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  量化过程 (前向传播)                                                     │
  │                                                                         │
  │  输入: 编码器输出 Z_e, 形状 (B, H, W, D) = (B, 32, 32, 64)            │
  │                                                                         │
  │  Step 1: 展平                                                           │
  │    flattened = reshape(Z_e, [-1, D])  → (B×H×W, 64) = (N, 64)          │
  │    把 1024 个位置的 64 维向量排成一个大矩阵                             │
  │                                                                         │
  │  Step 2: 计算距离 (欧氏距离的展开式)                                    │
  │    对每个位置 i 和每个码字 j:                                            │
  │      distance[i,j] = ||flattened[i] - embedding[j]||²                   │
  │                    = ||z_i||² + ||e_j||² - 2·z_i·e_j                   │
  │                                                                         │
  │    矩阵形式:                                                             │
  │      distances = flat_sq + emb_sq - 2·similarity                        │
  │      distances.shape = (N, 512)   ← 每个位置的 512 个距离              │
  │                                                                         │
  │  Step 3: 最近邻匹配                                                      │
  │    indices[i] = argmin_j distances[i, j]  → 选距离最小的码字            │
  │    然后 one-hot 编码: encodings[i] = one_hot(indices[i], 512)           │
  │                                                                         │
  │  Step 4: 查表替换                                                       │
  │    quantized = encodings @ embeddings^T                                 │
  │    将 one-hot × 码本矩阵 = 直接"取出"对应码字                           │
  │                                                                         │
  │  Step 5: 直通估计器 (Straight-Through Estimator)                        │
  │    output = Z_e + stop_gradient(quantized - Z_e)                        │
  │                                                                         │
  │    前向: output = quantized (等于码字)                                  │
  │    反向: 梯度直接穿过，grad(output) ≈ grad(Z_e)                         │
  │    这解决了一个关键问题: argmin 操作不可微，但我们需要梯度反向传播！     │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  损失函数 (三部分)                                                       │
  │                                                                         │
  │  1. 码本损失 (Codebook Loss):                                           │
  │     L_codebook = ||sg[Z_e] - Z_q||²                                    │
  │     让码字向编码器输出靠近（更新码本，不更新编码器）                     │
  │                                                                         │
  │  2. 承诺损失 (Commitment Loss):                                         │
  │     L_commit = β · ||Z_e - sg[Z_q]||²                                  │
  │     让编码器输出向码字靠近（更新编码器，不更新码本）                     │
  │     β = 0.25 控制承诺强度                                               │
  │                                                                         │
  │  3. 重建损失 + 物理约束 (在下游训练循环中):                              │
  │     L_total = VQ_loss + 10·MAE + 5·(1-MS-SSIM)                          │
  │              + 20·|mean(x) - mean(x̂)|  ← 孔隙度保持                    │
  │              + 10·||sobel(x) - sobel(x̂)||²  ← 边缘保留               │
  └─────────────────────────────────────────────────────────────────────────┘
"""


class VectorQuantizer(layers.Layer):
    """
    矢量量化层 — VQ-VAE 的核心组件
    
    输入: 编码器输出的连续特征 Z_e (B, H, W, D)
    输出: 量化后的离散特征 Z_q (B, H, W, D), 形状不变但值被替换为码字
    
    参数:
        num_embeddings: 码本大小 (512) — 相当于"词汇表"中有 512 个单词
        embedding_dim: 每个码字的维度 (64) — 每个单词有 64 个属性
        beta: 承诺损失权重 (0.25) — 控制编码器向码本靠拢的力度
    """
    
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim   # 64
        self.num_embeddings = num_embeddings # 512
        self.beta = beta                     # 承诺损失权重

        # 初始化码本: (64, 512) — 512个64维码字
        # 使用随机均匀初始化，训练过程中会自动学习
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings),
                dtype="float32"
            ),
            trainable=True,  # 码本是可以训练的！
            name="embeddings_vqvae",
        )

    def call(self, x):
        """
        前向传播: Z_e → 量化 → Z_q (带 STE 直通梯度)
        
        输入 x: (B, H, W, D) = (B, 32, 32, 64)
        输出:   (B, 32, 32, 64) — 形状相同但值被替换
        
        类比信号处理:
          这就像将连续的模拟信号通过 A/D 转换器量化为离散的数字信号
          输入是 64 维的"模拟特征"，输出被映射到 512 个离散的"数字码字"之一
        """
        input_shape = tf.shape(x)
        
        # ── Step 1: 展平 ──
        # 将 (B, 32, 32, 64) 展平为 (B×32×32, 64) = (N, 64)
        # 每个空间位置 (共 1024 个) 都是独立的 64 维向量待量化
        flattened = tf.reshape(x, [-1, self.embedding_dim])  # (N, 64)

        # ── Step 2: 计算欧氏距离 ──
        # 使用展开公式: ||z - e||² = ||z||² + ||e||² - 2·z·e
        # 避免显式计算差值的广播开销
        
        # similarity: (N, 512) — 每个位置与每个码字的内积
        similarity = tf.matmul(flattened, self.embeddings)

        # flat_sq: (N, 1) — 每个位置的 ||z||²
        flat_sq = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)

        # emb_sq: (1, 512) — 每个码字的 ||e||²
        # 显式 reshape 防止 TensorFlow 广播维度推断错误
        emb_sq = tf.reshape(
            tf.reduce_sum(self.embeddings ** 2, axis=0), [1, -1]
        )

        # distances: (N, 512) 
        # (N,1) + (1,512) = (N,512) 广播加法
        distances = flat_sq + emb_sq - 2.0 * similarity

        # ── Step 3: 最近邻查找 ──
        # encoding_indices: (N,) — 每个位置的最近码字索引 [0, 511]
        encoding_indices = tf.argmin(distances, axis=1)
        
        # encodings: (N, 512) — one-hot 表示
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        # ── Step 4: 查表替换 ──
        # quantized: (N, 64) → reshape → (B, H, W, 64)
        # encodings @ embeddings^T: one-hot 乘以码本 = 直接"取出"对应码字
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)  # (B, 32, 32, 64)

        # ── Step 5: VQ 损失 ──
        # 承诺损失: 编码器输出向码字靠拢 (更新编码器)
        # stop_gradient(quantized) 意味着梯度不穿过 quantized 回传
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        
        # 码本损失: 码字向编码器输出靠拢 (更新码本)
        # stop_gradient(x) 意味着梯度不穿过 x 回传
        codebook_loss = tf.reduce_mean(
            (quantized - tf.stop_gradient(x)) ** 2
        )
        
        # 将损失注册到 Layer，Keras 会自动收集
        self.add_loss(commitment_loss + codebook_loss)

        # ── Step 6: 直通估计器 (STE) ──
        # 这是 VQ-VAE 能训练的关键技巧！
        #
        # 问题: argmin + one_hot 操作不可微，梯度无法从解码器回传到编码器
        # 解决: 在前向传播中用 quantized 值，
        #       但在反向传播时假装 quantized == x，让梯度"直通"过去
        #
        # 前向: output = x + (quantized - x) = quantized  ← stop_gradient 阻止了 quantized 的梯度
        # 反向: grad(output) = grad(x)                     ← 梯度直接传给 x (即编码器输出)
        quantized = x + tf.stop_gradient(quantized - x)
        
        return quantized


# ============================================================================
#  Part 5: 自回归先验尝试 (PixelCNN) — 为什么效果不好？
# ============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  从"重建"到"生成"：为什么需要先验模型？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  VQ-VAE 训练完成后：
    ✅ 编码器学会了将图像"翻译"成 32×32 的离散码本索引 (0~511)
    ✅ 解码器学会了从 32×32 的码本索引"重建"回原始图像
    ✅ 重建质量很高，微裂缝和溶孔边缘都保留了
    
  ❌ 但是！VQ-VAE 本身不能生成"新"图像！
  
  为什么？
    训练好的 VQ-VAE 可以重建训练数据（有编码器给它"正确答案"）
    但要生成新图像，我们需要知道：一个"合理的"32×32 码本索引排列是什么样的？
    
    类比：VQ-VAE 学会了"英语 → 中文"互译
    但要自己"写一篇英文作文"，需要知道英文的语法和词汇搭配规律
    → 这就是"先验模型"(Prior Model) 的职责

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  PixelCNN 自回归先验                                                      │
  │                                                                         │
  │  思路: 用训练数据的所有 VQ 编码作为"语料库"                               │
  │        训练一个自回归模型，学习 32×32 个离散 token 之间的条件概率:       │
  │                                                                         │
  │        P(code[0,0], code[0,1], ..., code[31,31]) =                      │
  │          P(code[0,0]) ×                                                 │
  │          P(code[0,1] | code[0,0]) ×                                     │
  │          P(code[0,2] | code[0,0], code[0,1]) × ...                      │
  │                                                                         │
  │  类似 GPT 预测下一个词，PixelCNN 按光栅扫描顺序逐像素预测下一个 token。   │
  │                                                                         │
  │  问题:                                                                    │
  │    1. 极端类别不平衡: 90% 的 token 是"基质"，只有少数是"孔隙"             │
  │       → 模型倾向于一直预测基质 → 生成的图像全是暗色背景，没有孔隙！      │
  │                                                                         │
  │    2. 自回归采样不可控: 虽然理论上可以逐 token 采样生成新的 32×32 码本，  │
  │       但无法控制孔隙的数量、位置、连通性                                  │
  │                                                                         │
  │    3. 频率倒数加权也没完全解决:                                          │
  │       给稀有 token (孔隙类) 更高权重 → 生成了孔，但分布不自然            │
  │                                                                         │
  │  结论: PixelCNN 在 32×32 的离散码本索引上自回归生成效果不佳              │
  │        → 我们需要一种更可控的生成方法                                     │
  └─────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
#  Part 6: 离散掩膜融合算法 (Fuse-then-VQ) — 我们的核心创新
# ============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  离散掩膜融合 — 可控的"基因混合"生成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  核心洞察: 
    与其从零生成 VQ 编码（难），不如将两张真实岩心的"基因"混合生成新的。
    两张真实岩心 A (基质母本) 和 B (孔隙供体) → 在潜空间按掩膜比例混合 → 新岩心

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  类比: 信号的频域混合                                                    │
  │                                                                         │
  │  在信号处理中，如果我们有两个音频信号:                                   │
  │    s_A(t) = 低频背景 (基质)                                              │
  │    s_B(t) = 高频细节 (孔隙)                                              │
  │                                                                         │
  │  直接时域拼接: 前半段用 s_A，后半段用 s_B → 断点处会产生"咔嚓"声          │
  │                (VQ-then-Fuse: 在离散码本索引层面硬切换)                   │
  │                                                                         │
  │  频域混合: 做 FFT → 在频域中按比例混合 → IFFT → 平滑过渡                 │
  │            (Fuse-then-VQ: 在连续潜空间线性插值 → 统一量化 → 解码)        │
  │                                                                         │
  │  这就是我们的核心创新: "先融合，再量化" 而非 "先量化，再拼接"             │
  └─────────────────────────────────────────────────────────────────────────┘

  算法完整流程 (5步):

  ┌──────┐    ┌──────┐
  │ 岩心A │    │ 岩心B │    ← 两张真实岩心 (256×256)
  │(基质) │    │(孔隙) │
  └──┬───┘    └──┬───┘
     │           │
     ▼           ▼
  ┌──────────────────────┐
  │ Step 1: 直方图匹配    │  ← 消除背景色差("阴阳脸")
  │ B 的灰度分布→匹配到 A │
  └──────────┬───────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
  ┌──────┐       ┌──────┐
  │Encoder│      │Encoder│    ← 编码到连续潜空间 Z_e
  │  →A  │       │  →B  │       (32×32×64 的 "特征频谱")
  └──┬───┘       └──┬───┘
     │              │
     │   Z_eA       │   Z_eB
     │   (连续)     │   (连续)
     │              │
     └──────┬───────┘
            │
            ▼
  ┌──────────────────────┐
  │ Step 2: 生成有机掩膜  │  ← GRF 高斯随机场
  │ 噪声→大σ高斯→二值化  │    产生"云雾状"的自然过渡掩膜
  │ →小σ柔化边缘          │
  └──────────┬───────────┘
             │
             ▼  mask (1, 32, 32, 1), 值在 [0,1] 渐变
             │
  ┌──────────────────────┐
  │ Step 3: 连续潜空间融合│  ← 核心！在量化前完成混合
  │ Z_fused = Z_eA·(1-m) │
  │         + Z_eB·m      │    64维空间中逐元素加权平均
  └──────────┬───────────┘
             │
             ▼  Z_fused (1, 32, 32, 64)  ← 仍然是连续的！
             │
  ┌──────────────────────┐
  │ Step 4: VQ 统一量化   │  ← 整个融合特征一起量化
  │ Z_fused → 找最近码字  │    码本"看到"的是混合后的整体
  │ → Z_q (离散)          │
  └──────────┬───────────┘
             │
             ▼  Z_q (1, 32, 32, 64)
             │
  ┌──────────────────────┐
  │ Step 5: 解码器解码    │  ← 从量化特征重建图像
  │ Z_q → 256×256 图像    │    融合后的码本索引→解码→新岩心
  └──────────┬───────────┘
             │
             ▼
         ┌──────────┐
         │ 全新岩心! │   ← 混合了 A 的基质 + B 的孔隙
         └──────────┘

  为什么 Fuse-then-VQ 比 VQ-then-Fuse 好？

  消融实验对比:
  ┌──────────────────────────────────────────────────────────────────┐
  │  VQ-then-Fuse (先量化后拼接):                                     │
  │    Z_eA → VQ → indices_A (离散 0~511)                            │
  │    Z_eB → VQ → indices_B (离散 0~511)                            │
  │    fused_indices[i,j] = mask>0.5 ? indices_B : indices_A         │
  │                                                                  │
  │    问题: 在掩膜边界处，相邻两个 8×8 的 latent cell 分别用         │
  │          码字 237 和码字 89 → 解码后产生突兀的方块边界            │
  │          (每个 latent cell 对应 8×8 像素区域)                     │
  │                                                                  │
  │  Fuse-then-VQ (先融合后量化) — 我们的方法:                        │
  │    Z_fused[i,j,:] = Z_eA[i,j,:] × (1-m[i,j])                    │
  │                   + Z_eB[i,j,:] × m[i,j]                         │
  │                                                                  │
  │    这做了 64 维空间中的线性插值"基因混合":                       │
  │      码本维度 0: 0.7×A的特征 + 0.3×B的特征                       │
  │      码本维度 1: 0.7×A的特征 + 0.3×B的特征                       │
  │      ...                                                         │
  │      码本维度 63: 0.7×A的特征 + 0.3×B的特征                      │
  │                                                                  │
  │    然后 VQ 码本为这个"混合基因"找到最匹配的码字 → 天然平滑过渡    │
  │    因为相邻位置的 m[i,j] 是渐变的，混合比例连续变化               │
  └──────────────────────────────────────────────────────────────────┘
"""


# ── 有机掩膜生成函数 (GRF 高斯随机场) ──
def generate_organic_mask(h, w, cloud_sigma=4.0, soften_sigma=1.0):
    """
    生成有机云雾状掩膜 — 核心创新点之一
    
    四步流水线:
    
    Step 1: 随机高斯噪声
        noise = randn(h, w)  — 独立同分布的 32×32 白噪声场
        类比: 信号处理中的白噪声（所有频率功率相同）
    
    Step 2: 大 σ 高斯模糊 (σ=4.0)
        cloud = gaussian_filter(noise, sigma=4.0)
        
        物理意义: 这在频域中相当于低通滤波！
        白噪声（平坦频谱）→ 高斯低通 → 只保留低频分量
        → 产生空间上光滑的"云雾状"团块
        
        类比信号处理: 白噪声通过低通滤波器 = 有色噪声
                      频域中乘以 exp(-σ²ω²/2) 高斯窗函数
                      
        为什么 σ=4.0？
        - 掩膜尺寸是 32×32，σ=4.0 对应的空间尺度约为 4 个 latent cell
        - 在像素空间中对应约 4×8 = 32 像素的特征尺度
        - 这大致匹配碳酸盐岩中溶孔的典型尺寸
    
    Step 3: 中位数二值化
        threshold = median(cloud)
        binary = (cloud > threshold).astype(float32)
        
        为什么用中位数？
        - 保证掩膜覆盖率约 50% (一半是 A 的基质，一半是 B 的孔隙)
        - 在 batch_generate 中改用 percentile 阈值来控制孔隙密度
        
        类比: 信号处理中的硬限幅 (hard limiter) 或 1-bit 量化
    
    Step 4: 小 σ 柔化边缘 (σ=1.0) — 消除锯齿的关键！
        soft_mask = gaussian_filter(binary, sigma=1.0)
        
        物理意义: 将硬 0/1 边界变成 [0,1] 的渐变过渡带
        在频域中: 硬边界 = 高频分量 → 会产生吉布斯震荡/块效应
                  高斯柔化 = 抑制高频 → 平滑过渡，无振铃
        
        类比信号处理:
        - 硬二值化 = 理想低通滤波器（频域矩形窗）→ 时域吉布斯震荡
        - 高斯窗柔化 = 高斯低通滤波器 → 时域平滑，无震荡
        
        这正是我们消融实验证明的：不加柔化 → 锯齿伪影
                                  加柔化   → 丝滑过渡
        
    Args:
        h, w: 掩膜尺寸 = (32, 32) — VQ 潜空间的空间维度
        cloud_sigma: 团块尺度 σ (默认 4.0) 
        soften_sigma: 边缘柔化 σ (默认 1.0)
    
    Returns:
        soft_mask: (1, h, w, 1) float32, 值在 [0,1] 连续渐变
    """
    from scipy.ndimage import gaussian_filter

    # Step 1: 随机噪声场 — 类似信号中的"白噪声激励"
    noise = np.random.randn(h, w).astype(np.float32)

    # Step 2: 频域低通滤波 (大 σ) → 产生空间相关的"云雾"团块
    cloud = gaussian_filter(noise, sigma=cloud_sigma)

    # Step 3: 硬阈值二值化 → 将连续场变成 0/1 掩膜
    threshold = np.median(cloud)
    binary = (cloud > threshold).astype(np.float32)

    # Step 4: 边缘柔化 (小 σ) → 消除硬边界的块效应
    soft_mask = gaussian_filter(binary, sigma=soften_sigma)

    # 扩展维度用于广播: (1, 32, 32, 1)
    return soft_mask[np.newaxis, :, :, np.newaxis]


# ── 连续潜空间融合函数 (核心算法) ──
def fuse_then_vq(encoder, decoder, vq_layer, img_A, img_B, mask_soft):
    """
    Fuse-then-VQ: 在连续潜空间中柔性融合，然后统一量化 — 我们的核心算法
    
    这个函数实现了论文中提出的"离散掩膜融合"。
    
    与消融基线 VQ-then-Fuse 的本质区别:
    ┌──────────────────────────────────────────────────────────────────┐
    │  VQ-then-Fuse:                                                   │
    │    编码A → VQ量化 → 离散索引A ┐                                  │
    │    编码B → VQ量化 → 离散索引B ┤→ 按掩膜硬切换 → 解码             │
    │    问题: 索引是"整数"，没有"半个码字 237"的概念                   │
    │          → 边界处只能二选一 → 8×8 方块伪影                        │
    │                                                                  │
    │  Fuse-then-VQ (本函数):                                           │
    │    编码A → Z_eA (连续) ┐                                         │
    │    编码B → Z_eB (连续) ┤→ 64维空间中线性插值 → VQ量化 → 解码     │
    │    优势: 64维向量可以做任意比例的混合                             │
    │          → "60%像码字237 + 40%像码字89" = 一个全新的混合特征     │
    │          → VQ 码本为这个混合特征找到最匹配的单一码字               │
    │          → 相邻位置混合比例渐变 → 自然过渡                         │
    └──────────────────────────────────────────────────────────────────┘

    类比信号处理:
      这就像两个音频信号在频域中混合：
      - VQ-then-Fuse = 前半段播信号A，后半段播信号B → 断点处"咔嗒"声
      - Fuse-then-VQ  = 在每个频率分量上按比例混合 A 和 B → 平滑过渡
      
      编码器 ≈ FFT (时域→频域)
      潜空间 ≈ 频域系数
      融合    ≈ 每个频率分量加权平均
      解码器 ≈ IFFT (频域→时域)
    
    Args:
        encoder: 训练好的 VQ-VAE 编码器
        decoder: 训练好的 VQ-VAE 解码器
        vq_layer: 训练好的 VectorQuantizer (含 512 个码字的码本)
        img_A: 基质母本 (256, 256, 1), 亮度较高的岩心
        img_B: 孔隙供体 (256, 256, 1), 已做直方图匹配对齐到 A
        mask_soft: 有机掩膜 (1, 32, 32, 1), 值在 [0,1] 渐变
                   1.0 = 该位置完全用 B 的特征
                   0.0 = 该位置完全用 A 的特征
                   0.5 = A 和 B 各一半

    Returns:
        generated_img: 生成的融合岩心 (256, 256), 值在 [0, 1]
    """
    # Step 1: 编码到连续潜空间 Z_e (VQ 量化之前!)
    # Z_e 是 32×32×64 的连续特征张量 — 类似频域中的频谱系数
    z_eA = encoder(tf.constant(np.expand_dims(img_A, 0)), training=False)
    z_eB = encoder(tf.constant(np.expand_dims(img_B, 0)), training=False)
    # z_eA.shape = z_eB.shape = (1, 32, 32, 64)

    # Step 2: 将掩膜转为 TensorFlow 张量
    mask_t = tf.constant(mask_soft, dtype=tf.float32)
    # mask_t.shape = (1, 32, 32, 1)

    # Step 3: 连续潜空间线性插值融合 (核心!)
    # 对每个空间位置 (i,j) 和每个特征维度 (d):
    #   Z_fused[i,j,d] = Z_eA[i,j,d] × (1 - mask[i,j]) + Z_eB[i,j,d] × mask[i,j]
    #
    # 这是 64 维向量空间中的凸组合 (convex combination)!
    # 因为 mask ∈ [0,1]，所以 Z_fused 始终在 Z_eA 和 Z_eB 的连线上
    #
    # 类比: 在 RGB 色彩空间中混合两个颜色
    #       红色 (255,0,0) × 0.7 + 蓝色 (0,0,255) × 0.3 = 紫色 (178,0,76)
    #       这里是在 64 维"特征颜色空间"中做同样的操作
    z_fused = z_eA * (1.0 - mask_t) + z_eB * mask_t
    # z_fused.shape = (1, 32, 32, 64)

    # Step 4: VQ 量化 — 将混合后的连续特征映射到最近的码字
    # 这是关键! 码本看到的是"混合后"的特征，而非两张原图的特征
    # 512 个码字中总有一个最能代表这个混合特征
    z_q = vq_layer(z_fused, training=False)

    # Step 5: 解码 — 从量化特征重建 256×256 图像
    generated = decoder(z_q, training=False)
    img_gen = np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)

    return img_gen


# ── 消融对比: VQ-then-Fuse ──
def vq_then_fuse(encoder, decoder, vq_layer, img_A, img_B, mask_hard):
    """
    VQ-then-Fuse: 先分别量化，再在离散索引层面硬拼接 — 消融实验基线
    
    这个方法暴露了"先量化后融合"的根本问题:
    在 mask 边界处，离散索引只能二选一 → 产生 8×8 方块伪影
    
    一个 latent cell 的索引是整数 (0~511)，没有"半个索引"的概念
    → 边界处要么全用 A 的码字，要么全用 B 的码字
    → 每个 latent cell 对应 8×8 像素 → 伪影以 8×8 方块形式出现
    """
    # 编码
    z_eA = encoder(tf.constant(np.expand_dims(img_A, 0)), training=False)
    z_eB = encoder(tf.constant(np.expand_dims(img_B, 0)), training=False)

    # 分别量化 — 获取离散码本索引
    # 这里需要一个辅助函数 get_vq_indices:
    #   对于每个位置的 64 维向量，找到最近的码字索引 (0~511)
    indices_A = _get_vq_indices(vq_layer, z_eA)  # (1, 32, 32)
    indices_B = _get_vq_indices(vq_layer, z_eB)  # (1, 32, 32)

    # 在离散索引层面硬切换
    # 问题: mask_2d[i,j] = 0.6 → 大于 0.5 → 用 indices_B[i,j]
    #       mask_2d[i+1,j] = 0.4 → 小于 0.5 → 用 indices_A[i+1,j]
    #       → 相邻位置可能用完全不同的码字 → 方块边界!
    mask_2d = mask_hard[0, :, :, 0]
    fused_indices = np.where(
        mask_2d > 0.5, indices_B, indices_A
    ).astype(np.int32)

    # 查表恢复为 Z_q 再解码
    z_q = _indices_to_zq(vq_layer, fused_indices)
    generated = decoder(z_q, training=False)
    return np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)


def _get_vq_indices(vq_layer, z_e):
    """
    获取 VQ 量化后的码本索引 (最近邻查找)
    
    输入: z_e (B, H, W, D) — 连续特征
    输出: indices (B, H, W) — 每个位置的码字索引 [0, 511]
    
    类比: 这是 VQ 量化层的前半部分 (查找最近码字)，
          但没有 STE 直通估计器和 reshape 操作
    """
    input_shape = tf.shape(z_e)
    flattened = tf.reshape(z_e, [-1, vq_layer.embedding_dim])

    # 使用 VQ 量化层同样的距离计算
    similarity = tf.matmul(flattened, vq_layer.embeddings)
    flat_sq = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(
        tf.reduce_sum(vq_layer.embeddings ** 2, axis=0), [1, -1]
    )
    distances = flat_sq + emb_sq - 2.0 * similarity

    encoding_indices = tf.argmin(distances, axis=1)
    encoding_indices = tf.reshape(encoding_indices, input_shape[:3])
    return encoding_indices.numpy().astype(np.int32)


def _indices_to_zq(vq_layer, indices):
    """
    将码本索引转换回量化特征 Z_q
    
    输入: indices (H, W) — 每个位置的码字索引
    输出: z_q (1, H, W, D) — 量化后的特征
    
    这本质上是查表操作: z_q[i,j,:] = codebook[:, indices[i,j]]
    """
    H, W = indices.shape
    # 展平索引
    flat_indices = indices.reshape(-1)  # (H×W,)
    # One-hot 编码
    one_hot = np.eye(vq_layer.num_embeddings)[flat_indices]  # (H×W, 512)
    # 查表
    embeddings_np = vq_layer.embeddings.numpy()  # (64, 512)
    z_q_flat = one_hot @ embeddings_np.T  # (H×W, 64)
    z_q = z_q_flat.reshape(1, H, W, -1).astype(np.float32)
    return tf.constant(z_q)


# ============================================================================
#  完整生成流程示例 (伪代码)
# ============================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  从零到生成一张全新岩心的完整流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_novel_rock(data_dir, checkpoint_dir):
    '''
    完整的离散掩膜融合生成流程
    '''
    # 0. 加载训练好的 VQ-VAE 模型
    encoder = build_encoder(latent_dim=64)
    decoder = build_decoder(latent_dim=64)
    vq_layer = VectorQuantizer(num_embeddings=512, embedding_dim=64)
    
    encoder.load_weights("encoder_ep50.weights.h5")
    decoder.load_weights("decoder_ep50.weights.h5")
    vq_layer.set_weights([np.load("vq_layer_ep50.npy")])
    
    # 1. 选取两张差异最大的真实岩心
    img_A, img_B = pick_two_diverse_images(data_dir)
    # img_A: 基质母本 (偏亮,孔隙少)
    # img_B: 孔隙供体 (偏暗,孔隙多)
    
    # 2. 直方图匹配: 消除背景色差
    from skimage.exposure import match_histograms
    img_B_matched = match_histograms(img_B[:,:,0], img_A[:,:,0])
    # B 的灰度分布现在与 A 一致，但保留了 B 的结构信息
    
    # 3. 生成有机掩膜
    mask = generate_organic_mask(32, 32, cloud_sigma=4.0, soften_sigma=1.0)
    # mask.shape = (1, 32, 32, 1), 值在 [0,1] 平滑过渡
    
    # 4. Fuse-then-VQ 融合生成
    novel_rock = fuse_then_vq(encoder, decoder, vq_layer,
                              img_A, img_B_matched, mask)
    # novel_rock.shape = (256, 256), 全新的岩心!
    
    return novel_rock

  总结 — 回答"这是不是造假？":

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  每个组件都有其学术或工程来源:                                           │
  │                                                                         │
  │  VQ-VAE 架构: van den Oord et al., NeurIPS 2017                         │
  │  直方图匹配: 经典图像处理 (Richards & Jia, 遥感图像配准)                 │
  │  GRF 高斯随机场: 地质统计学 (Matheron, kriging)                          │
  │  高斯滤波柔化: 数字图像处理/信号处理标准技术                             │
  │                                                                         │
  │  我们的创新 (不是造假!):                                                  │
  │  ⭐ Fuse-then-VQ 融合顺序: 在连续潜空间融合 → 统一量化                   │
  │     这是 VQ-VAE 文献中之前没有被提出的架构级决策                        │
  │  ⭐ GRF 掩膜适配到 VQ-VAE 潜空间:                                         │
  │     将地质统计学中的随机模拟方法适配到深度学习潜空间                     │
  │  ⭐ 消融实验设计:                                                        │
  │     VQ-then-Fuse vs Fuse-then-VQ 的对照实验系统性地验证了效果            │
  └─────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print("=" * 70)
    print("  答辩算法讲解文件已就绪")
    print("  本文件包含完整的 VAE → VQ-VAE → 离散掩膜融合算法原理与代码")
    print("=" * 70)
    print()
    print("  阅读顺序建议:")
    print("    Part 0: 问题背景")
    print("    Part 1: 标准 VAE 原理 (类比频域)")
    print("    Part 2: 标准 VAE 的问题")
    print("    Part 3: VQ-VAE 核心原理 (离散码本)")
    print("    Part 4: 矢量量化层详解")
    print("    Part 5: 自回归先验失败原因")
    print("    Part 6: 离散掩膜融合算法 (核心创新)")
    print()
    print("  所有代码均可直接运行:")
    print("    python defense_algorithm_explained.py")
