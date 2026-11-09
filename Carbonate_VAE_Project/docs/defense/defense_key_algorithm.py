"""
================================================================================
  答辩核心代码：离散掩膜融合算法 (Fuse-then-VQ)
================================================================================
  这是我在硕士论文中提出的生成方法。下面展示三个关键组件：
  
  Algorithm 1: generate_organic_mask  — 有机掩膜生成 (GRF 高斯随机场)
  Algorithm 2: generate_vq_then_fuse  — 消融基线 (先量化, 再硬拼接)
  Algorithm 3: generate_fuse_then_vq  — 我的方法 (先融合, 再量化)  ★
  
  核心洞察：
    在 64 维连续潜空间中做凸组合融合，而不是在离散码本索引上二选一。
    类比：频域混合两个信号 vs 时域硬拼接两个信号。
================================================================================
"""
import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

# ============================================================================
#  Algorithm 1: 有机掩膜生成 — 四步流水线
#  ============================================================================
#  目的：生成一张 32×32 的软掩膜 M，值在 [0,1] 连续渐变
#        决定每个位置"多少来自岩心A、多少来自岩心B"
#
#  Step 1: 白噪声 → Step 2: 大σ低通滤波 (形成团块) →
#  Step 3: 中位数二值化 → Step 4: 小σ柔化边缘 (消除锯齿)
# ============================================================================

def generate_organic_mask(h=32, w=32, cloud_sigma=4.0, soften_sigma=1.0):
    """
    生成有机云雾状软掩膜。
    
    类比信号处理：
      - Step 1-2：白噪声通过高斯低通滤波器 → 有色噪声 (空间相关的团块)
      - Step 3：  硬限幅器 / 1-bit 量化
      - Step 4：  高斯窗平滑 → 消除硬切换产生的吉布斯震荡
    
    Returns:
        soft_mask: (1, 32, 32, 1)  float32 ∈ [0, 1], 渐变过渡
    """
    # Step 1: 独立高斯白噪声 (32×32)
    noise = np.random.randn(h, w).astype(np.float32)

    # Step 2: 大 σ 高斯模糊 → 引入空间相关性 → "云雾"团块
    cloud = gaussian_filter(noise, sigma=cloud_sigma)

    # Step 3: 中位数二值化 → 保证覆盖率约 50%
    threshold = np.median(cloud)
    binary = (cloud > threshold).astype(np.float32)

    # Step 4: 小 σ 柔化 → 硬边界 → 渐变过渡带 ★ 消除锯齿的关键
    soft_mask = gaussian_filter(binary, sigma=soften_sigma)

    # 扩展为 4D 张量以便广播: (1, H, W, 1)
    return soft_mask[np.newaxis, :, :, np.newaxis], binary


# ============================================================================
#  Algorithm 2: VQ-then-Fuse — 消融基线
#  ============================================================================
#  策略: 岩心A、B → 各自通过 Encoder + VQ → 得到离散码本索引
#        → 按硬掩膜 (只有 0/1) 在索引层面二选一 → 解码
#
#  问题: 每个位置对应 8×8 像素的感受野
#        边界处相邻位置可能分属码字 #237 和 #89 → 8×8 方块伪影
# ============================================================================

def generate_vq_then_fuse(encoder, decoder, vq_layer, img_A, img_B, mask_hard):
    """
    消融基线: 先离散量化，再在码本索引层面硬拼接。
    
    mask_hard 只有 0 或 1 (没有中间值)
    → fused_indices[i,j] = mask==1 ? indices_B[i,j] : indices_A[i,j]
    → 每个位置"非A即B"，没有渐变
    → 解码后产生方块边界
    """
    # 编码到连续潜空间
    z_eA = encoder(tf.constant(np.expand_dims(img_A, 0)), training=False)  # (1,32,32,64)
    z_eB = encoder(tf.constant(np.expand_dims(img_B, 0)), training=False)

    # 各自 VQ 量化 → 获取离散索引 (0~511 整数)
    indices_A = get_vq_indices(vq_layer, z_eA)  # (1, 32, 32)
    indices_B = get_vq_indices(vq_layer, z_eB)

    # ── 关键行：在整数索引层面硬切换 ──
    # mask_2d ∈ {0, 1}，没有中间值！
    # 0.0 → 用 A 的码字；1.0 → 用 B 的码字
    mask_2d = mask_hard[0, :, :, 0]
    fused_indices = np.where(mask_2d > 0.5, indices_B, indices_A)
    # ─────────────────────────────────────

    # 查表 + 解码
    z_q = indices_to_zq(vq_layer, fused_indices.astype(np.int32))
    generated = decoder(z_q, training=False)
    return np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)


# ============================================================================
#  Algorithm 3: Fuse-then-VQ — 我的方法 ★
#  ============================================================================
#  策略: 岩心A、B → Encoder → Z_eA, Z_eB (连续64维向量)
#        → 在 64 维空间中按掩膜比例做凸组合
#        → Z_fused = Z_eA × (1-M) + Z_eB × M
#        → 统一 VQ 量化 → 解码
#
#  优势: 掩膜 M ∈ [0,1] 连续值 → 每个位置可以做"60%A + 40%B"的混合
#        → VQ 码本为每个混合特征找最佳匹配码字
#        → 相邻位置混合比例渐变 → 自然平滑过渡
# ============================================================================

def generate_fuse_then_vq(encoder, decoder, vq_layer, img_A, img_B, mask_soft):
    """
    我的方法: 先在连续潜空间中柔性融合，再统一量化。
    
    类比: 不是在时域拼接两个音频信号 (会有"咔嗒"断点)，
          而是在频域中按比例混合每个频率分量 → 平滑过渡。
          
    编码器 ≈ FFT，潜空间 ≈ 频域系数，融合 = 频域凸组合，解码器 ≈ IFFT。
    """
    # Step 1: 编码到连续潜空间 Z_e (VQ 量化之前!)
    z_eA = encoder(tf.constant(np.expand_dims(img_A, 0)), training=False)  # (1,32,32,64)
    z_eB = encoder(tf.constant(np.expand_dims(img_B, 0)), training=False)

    mask_t = tf.constant(mask_soft, dtype=tf.float32)  # (1,32,32,1)

    # ── ★ 核心公式：64 维凸组合 (convex combination) ★ ──
    # Z_fused[i,j,:] = Z_eA[i,j,:] × (1 - M[i,j]) + Z_eB[i,j,:] × M[i,j]
    #
    # M[i,j] ∈ [0,1] 连续值 — 这是关键！
    #   0.0 → 完全来自 A
    #   0.3 → 70%A + 30%B  (A 和 B 的遗传混合)
    #   0.5 → 各一半
    #   0.7 → 30%A + 70%B
    #   1.0 → 完全来自 B
    z_fused = z_eA * (1.0 - mask_t) + z_eB * mask_t
    # ──────────────────────────────────────────────

    # Step 2: VQ 量化 — 为"混合基因"找最匹配的码字
    # 512 个码字中总有一个最能代表这个混合特征
    z_q = vq_layer(z_fused, training=False)

    # Step 3: 解码 → 256×256 新岩心
    generated = decoder(z_q, training=False)
    return np.clip(generated.numpy()[0, :, :, 0], 0.0, 1.0)


# ============================================================================
#  辅助函数
# ============================================================================

def get_vq_indices(vq_layer, z_e):
    """VQ 最近邻查找，返回码本索引 (0~511)"""
    input_shape = tf.shape(z_e)
    flattened = tf.reshape(z_e, [-1, vq_layer.embedding_dim])
    # ||z - e||² = ||z||² + ||e||² - 2·z·e
    similarity = tf.matmul(flattened, vq_layer.embeddings)
    flat_sq = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(
        tf.reduce_sum(vq_layer.embeddings ** 2, axis=0), [1, -1])
    distances = flat_sq + emb_sq - 2.0 * similarity
    encoding_indices = tf.argmin(distances, axis=1)
    return tf.reshape(encoding_indices, input_shape[:3]).numpy().astype(np.int32)


def indices_to_zq(vq_layer, indices):
    """码本索引 → 量化特征 Z_q (查表)"""
    H, W = indices.shape
    one_hot = np.eye(vq_layer.num_embeddings)[indices.reshape(-1)]
    embeddings_np = vq_layer.embeddings.numpy()  # (64, 512)
    z_q_flat = one_hot @ embeddings_np.T  # (H×W, 64)
    return tf.constant(z_q_flat.reshape(1, H, W, -1).astype(np.float32))


# ============================================================================
#  两段核心代码的对比总结
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  离散掩膜融合算法 — 答辩核心代码")
    print("=" * 65)
    print()
    print("  ┌─ 消融基线 (VQ-then-Fuse) ──────────────────────┐")
    print("  │  fused_indices = mask>0.5 ? idx_B : idx_A      │")
    print("  │  → 整数二选一, 每个位置非A即B, 边界处有锯齿     │")
    print("  └────────────────────────────────────────────────┘")
    print()
    print("  ┌─ 我的方法 (Fuse-then-VQ) ★ ─────────────────────┐")
    print("  │  Z_fused = Z_eA×(1-M) + Z_eB×M                 │")
    print("  │  → 64维凸组合, M∈[0,1]连续, 边界处渐变平滑     │")
    print("  └────────────────────────────────────────────────┘")
    print()
    print("  本质区别: 连续向量插值 vs 离散索引切换")
    print("  类比:    频域混合信号 vs 时域硬拼接信号")
