# 图 3-8 ~ 3-11 代码原理详解（汇报用）

> 所有脚本位于 `thesis_code/figure_scripts/`，生成的图片位于 `results/evaluation/`。

---

## 图 3-8: PixelCNN vs NanoGPT 训练损失对比折线图

**脚本**: [`figure_3_8_loss_comparison.py`](../thesis_code/figure_scripts/figure_3_8_loss_comparison.py)
**输出**: `results/evaluation/Figure_3_8_Loss_Comparison.png`

### 数据来源

两条曲线的 loss 历史来自两个独立训练过程：

| 模型 | 数据文件 | 训练脚本 |
|------|----------|----------|
| PixelCNN (18 ResBlocks) | `pixelcnn_retrain_history.npz` | `train_pixelcnn_retrain.py` |
| NanoGPT V2 (4L/4H/256D) | `gpt_prior_v2_history.npz` | `train_gpt_prior.py` |

### 核心代码逻辑

```python
# 1. 加载两个模型的历史 loss
pixelcnn = np.load(PIXELCNN_HISTORY)     # train_loss, val_loss
gpt = np.load(GPT_HISTORY)               # train, val

# 2. 截断到前 25 个 epoch
MAX_EPOCH = 25
pcnn_train = pixelcnn['train_loss'][:MAX_EPOCH]
pcnn_val   = pixelcnn['val_loss'][:MAX_EPOCH]
gpt_train  = gpt['train'][:MAX_EPOCH]
gpt_val    = gpt['val'][:MAX_EPOCH]
```

### 四条曲线的含义

- **PixelCNN Train Loss (蓝色实线)**: 加权交叉熵损失，在训练集上的自回归下一个词预测
- **PixelCNN Val Loss (蓝色虚线)**: 验证集上的损失，反映泛化能力
- **NanoGPT Train Loss (红色实线)**: 同样的加权交叉熵损失，但模型是 Transformer
- **NanoGPT Val Loss (红色虚线)**: Transformer 在验证集上的表现

### 为什么 NanoGPT 收敛更快/更低

1. **全局感受野**: Transformer 的自注意力机制能看到所有已生成的 token，而 PixelCNN 受限于卷积感受野（即使 18 个 ResBlock，RF = 7 + 18×2 = 43，刚好覆盖 32×32）
2. **参数效率**: 4L/4H/256D 的 Micro-GPT 约 1.7M 非嵌入参数，比 PixelCNN 更参数高效
3. **位置编码**: 2D 行列位置编码 (`wpe_row + wpe_col`) 提供了显式的空间结构信息

---

## 图 3-9 (上): PixelCNN 训练损失收敛曲线

**脚本**: [`figure_3_9_pixelcnn_loss.py`](../thesis_code/figure_scripts/figure_3_9_pixelcnn_loss.py)
**输出**: `results/evaluation/Figure_3_9_PixelCNN_Loss.png`

### 数据来源

- 来自 `results/checkpoints/pixelcnn_history.npz`（原始 PixelCNN 训练历史）
- 与图 3-8 不同，这里用的是**原始 PixelCNN**（非 retrained），展示的是单一模型的收敛性

### 核心标注逻辑

```python
# 在 epoch=25 处标注最终收敛值
# train_loss 标注在点下方（因为 train < val）
ax.annotate(f'{train_loss[-1]:.4f}',
            xy=(25, train_loss[-1]),
            xytext=(22.5, train_loss[-1] - 0.10),  # Y 向下偏移
            ...)
# val_loss 标注在点上方（因为 val > train）
ax.annotate(f'{val_loss[-1]:.4f}',
            xy=(25, val_loss[-1]),
            xytext=(22.5, val_loss[-1] + 0.12),  # Y 向上偏移
            ...)
```

### 关键观察

- 训练损失和验证损失之间的 gap 很小（约 0.02），说明 PixelCNN **没有过拟合**
- 训练曲线单调下降，约在 epoch 15 后趋于平稳

---

## 图 3-9 (下): 离散自回归生成网格拼接伪影对比

**脚本**: [`figure_3_9_3_11_combined.py`](../thesis_code/figure_scripts/figure_3_9_3_11_combined.py) 中的 `figure_3_9_artifacts()`
**输出**: `results/evaluation/Figure_3_9_Stitching_Artifacts.png`

### 原理：为什么会有拼接伪影？

#### PixelCNN 自回归生成（核心代码来自 `figure_4_5_retrained.py:179-246`）

```python
# Stage 1: 在 32×32 码本网格上随机撒种子 (seed patches)
# num_seeds=3, seed_patch_size=3 → 3个3×3的真实patch
for (sr, sc) in chosen:
    for dr in range(seed_patch_size):
        for dc in range(seed_patch_size):
            code_map[0, sr+dr, sc+dc] = real_code_map[0, sr+dr, sc+dc]

# Stage 2: 光栅扫描顺序逐像素生成 (从上到下，从左到右)
for h in range(32):
    for w in range(32):
        if seed_mask[0, h, w]:
            continue
        logits = pixelcnn(code_map, training=False)  # 只看"上方+左方"上下文
        # Top-P nucleus sampling with T=0.5, top_p=0.5
        sorted_logits, sorted_indices = tf.math.top_k(logits_hw, k=512)
        probs = tf.nn.softmax(sorted_logits / 0.5)
        cumsum = tf.cumsum(probs)
        nucleus_mask = cumsum < 0.5  # 只保留累积概率 < 0.5 的 top tokens
        sampled_idx = sample_from(nucleus_mask)
        code_map[0, h, w] = sampled_idx
```

**伪影成因**:
- PixelCNN 因果掩码只允许看到**上方和左方**的像素（自回归约束）
- 信息流是单向的（左上→右下），无法利用全局上下文
- 当遇到未见过的码本模式时，模型只能在有限上下文中"猜测"，导致**阶梯状不连续**（staircase artifact）
- 种子 patch 之间的区域没有真实信息桥接，产生**棋盘格错位**

#### NanoGPT V2 生成（核心代码来自 `nanogpt.py:247-308`）

```python
# 2D Seed: 前 2 行共 64 个真实 token 作为 prompt
seed_flat = seed_codes[:64].flatten()  # 2 rows × 32 cols
start_idx = torch.tensor(seed_flat, dtype=torch.long).unsqueeze(0)

# 自回归生成剩余 1024-64=960 个 token
gen_idx = gpt_model.generate(
    start_idx,
    max_new_tokens=1024 - 64,
    temperature=1.3,      # 高温 → 多样性
    top_p=0.95,           # 核采样 → 避免退化
    valid_tokens=valid_tokens_tensor  # 绝对废号屏蔽
)
```

**NanoGPT 的优势与局限**:
- **Transformer 自注意力**: 每个 token 能看到所有前面已生成的 token（全局感受野），不存在"阶梯"问题
- **但是**: 1D 因果注意力的顺序仍然是从左到右、从上到下的扁平化扫描，2D 空间结构被"压扁"成 1D
- **2D 位置编码缓解**: `wpe_row(pos//32) + wpe_col(pos%32)` 提供了行列位置信息，但不是完美的 2D 结构保持
- T=1.3 高温 + top_p=0.95 带来多样性，但也可能引入**棋盘格/网格状伪影**

### 红色方框标注

红框标注了人工选定的伪影明显区域，并附 2.5× 放大插图展示细节。

---

## 图 3-10: 连续特征张量提取与高斯软掩膜矩阵可视化

**脚本**: [`figure_3_10_tensor_mask.py`](../thesis_code/figure_scripts/figure_3_10_tensor_mask.py)
**输出**: `results/evaluation/Figure_3_10_Tensor_Mask.png`

### 三栏设计

#### 左侧: 3D 张量方块 (Continuous Latent Tensor Z_e)

```python
# 构建 H'×W'×D 的 3D 线框
H, W, D = 4, 4, 8  # 符号化维度: H'=32, W'=32, D=64
# 绘制所有三维网格线
for x in range(H+1):
    for y in range(W+1):
        ax_left.plot([x,x], [y,y], [0,D], color='#5BA0D0')  # Z 方向线
# ... X, Y 方向同理

# 用梯度色块填充每个通道层
for z in range(D):
    channel_color = plt.cm.Blues(0.3 + 0.5 * z/D)
    for x, y in grid_points:
        ax_left.plot([x+0.5], [y+0.5], [z+0.5], 's',
                     color=channel_color, markersize=18,
                     alpha=0.3 + 0.5*z/D)  # 越深层越深
```

**含义**: 编码器输出的连续特征张量 `Z_e ∈ R^(H'×W'×D)`，其中 H'=W'=32（空间维度），D=64（通道/特征维度）。这是 VQ-VAE 量化之前的状态。

#### 中间: 融合箭头与公式

```python
ax_mid.annotate('', xy=(0.95,0.5), xytext=(0.05,0.5),
    arrowprops=dict(arrowstyle='->', lw=3.5, color='#CC3333'))
ax_mid.text(0.5, 0.25,
    r'$\mathbf{Z}_{fused} = \mathbf{M} \odot \mathbf{Z}_{eB} + '
    r'(1-\mathbf{M}) \odot \mathbf{Z}_{eA}$')
```

**含义**: 核心融合公式 —— 两张图的连续潜特征按软掩膜 M 加权混合。

#### 右侧: 高斯软掩膜热力图 (Gaussian Soft Mask M)

```python
# 生成 2D 高斯衰减场
x = np.linspace(-2.5, 2.5, 200)
X, Y = np.meshgrid(x, x)
sigma = 1.2
M = np.exp(-(X**2 + Y**2) / (2 * sigma**2))  # 高斯核

# 灰度热力图 + 等值线叠加
im = ax_right.imshow(M, cmap='gray', vmin=0.0, vmax=1.0)
ax_right.contour(M, levels=[0.1,0.3,0.5,0.7,0.9],
                 colors='#FF6644', linewidths=0.8)  # 红橙色等值线
```

**含义**:
- `M(x,y) = exp(-(x² + y²)/(2σ²))` 其中 σ=1.2
- 中心 M≈1.0（图 B 的孔隙特征占主导）
- 边缘 M→0.0（图 A 的基质特征占主导）
- 等值线展示平滑的"均质→多孔"过渡

### 实际应用中的有机掩膜生成

在 `batch_generate_random_rocks.py:120-154` 中，实际使用的高斯软掩膜更加复杂：

```python
def generate_organic_mask(h, w, cloud_sigma=4.0, coverage_threshold=0.75):
    # Step 1: 高斯随机场 → 模拟天然岩石的有机纹理
    noise = np.random.randn(h, w)
    cloud = gaussian_filter(noise, sigma=cloud_sigma)      # 大 sigma 平滑

    # Step 2: 百分位阈值二值化 → 控制孔隙度
    thresh_val = np.percentile(cloud, coverage_threshold * 100)
    pore_binary = (cloud > thresh_val).astype(np.float32)

    # Step 3: 边缘软化 → 消除硬边界
    pore_soft = gaussian_filter(pore_binary, sigma=1.0)

    # matrix_mask: 1.0=基质(parent), pore_mask: 1.0=孔隙(donor)
    matrix_mask = 1.0 - pore_soft
    return matrix_mask, pore_soft
```

---

## 图 3-11: 潜空间掩膜融合 —— 离散拼接 vs 连续融合

**脚本**: [`figure_3_9_3_11_combined.py`](../thesis_code/figure_scripts/figure_3_9_3_11_combined.py) 中的 `figure_3_11_fusion()`
**输出**: `results/evaluation/Figure_3_11_Fusion_Comparison.png`

### 核心创新: "先向量量化再融合" vs "先融合再向量量化"

这是本文（论文）的核心方法论贡献。两者都发生在**潜空间（Latent Space）**，解码后看起来相似是因为共享相同的源图、掩膜和解码器，但细节边界有本质差异。

### 左侧 (a): 离散特征直接拼接 — "VQ-then-Fuse"

```python
# Step 1: 分别对 Z_eA 和 Z_eB 进行向量量化（最近邻查找码本索引）
indices_A = get_indices_from_z(z_eA)  # 将连续向量映射到离散码本ID
indices_B = get_indices_from_z(z_eB)
code_A = indices_A.numpy().astype(np.int32)[0]  # (32, 32)
code_B = indices_B.numpy().astype(np.int32)[0]

# Step 2: 在离散码本空间进行硬切换（硬二值拼接）
code_fused_discrete = np.where(
    pore_mask[0,:,:,0] > 0.5,   # 掩膜位置
    code_B,                       # 孔隙区: 用图B的码本ID
    code_A                        # 基质区: 用图A的码本ID
)
before_img = codes_to_image(decoder, vq_layer, code_fused_discrete)
```

**问题**: 码本 ID 是离散的（0~511），在边界处发生**突变**。即使掩膜是软的，码本 ID 的硬切换导致解码后在图像空间出现**锯齿台阶**（jagged steps），如红色放大插图所示。

### 右侧 (b): 连续潜空间特征掩膜融合 — "Fuse-then-VQ"

```python
# Step 1: 在连续潜空间进行软加权混合（核心创新）
z_fused = z_eA * matrix_mask + z_eB * pore_mask
# z_fused[h,w,:] = (1-M)·Z_eA[h,w,:] + M·Z_eB[h,w,:]
# 每个 64 维向量都是两个源的连续插值！

# Step 2: 融合后的连续向量再统一进行向量量化
indices_fused = get_indices_from_z(z_fused)
code_fused = indices_fused.numpy().astype(np.int32)[0]
after_img = codes_to_image(decoder, vq_layer, code_fused)
```

**优势**: 在 64 维连续空间中，掩膜边界处的向量是**逐渐过渡**的（例如 0.3·Z_A + 0.7·Z_B），VQ 量化时能找到最匹配这种"中间态"的码本向量，解码后呈现**丝滑过渡**。

### 最近邻码本查找 (get_indices_from_z)

这是将连续特征映射到离散码本的核心操作，用于两种融合方式：

```python
def get_indices_from_z(z_e):
    """L2 最近邻查找：对每个位置 (h,w)，在 512 个码本向量中找最近的那个"""
    flattened = tf.reshape(z_e, [-1, 64])              # (32*32, 64)

    # 距离计算: ||z - e_j||² = ||z||² + ||e_j||² - 2·z·e_j
    similarity = tf.matmul(flattened, vq_layer.embeddings)   # (1024, 512)
    flat_sq = tf.reduce_sum(flattened ** 2, axis=1, keepdims=True)
    emb_sq = tf.reshape(tf.reduce_sum(vq_layer.embeddings ** 2, axis=0), [1, -1])
    distances = flat_sq + emb_sq - 2.0 * similarity           # (1024, 512)

    encoding_indices = tf.argmin(distances, axis=1)           # (1024,)
    return tf.reshape(encoding_indices, [1, 32, 32])          # → 码本ID网格
```

### 为什么两张图整体看起来非常相似？

1. **共享源图**: 图 A 和图 B 是同一对源图，展示了相同的地质结构
2. **共享掩膜**: 使用相同的有机高斯掩膜 `M`，孔隙区域位置完全一致
3. **共享解码器**: VQ-VAE 解码器权重相同，码本向量相同
4. **差异只在边界**: 两种方式的差异集中在掩膜 M≈0.5 的过渡区域，需要通过 3× 放大才能观察到

### 红色虚线框 + 放大插图

- 通过计算掩膜梯度幅值 `grad_mag = sqrt(grad_x² + grad_y²)` 自动定位最陡边界
- 3× 放大显示边界细节：左侧"锯齿台阶"，右侧"丝滑过渡"
- `mark_inset(loc1=2, loc2=3)` 用相邻角（左下→右上）绘制矩形连接线（而非对角线的三角形）

---

## 总结：四张图之间的逻辑关系

```
图 3-8 ──→ 先验模型训练收敛（定量）
    │
    ├── PixelCNN 收敛轨迹 (蓝线)
    └── NanoGPT 收敛轨迹 (红线)

图 3-9 ──→ 生成质量问题（定性）
    │
    ├── PixelCNN 损失曲线 (上) — 模型收敛性证明
    └── 离散自回归伪影 (下) — PixelCNN vs NanoGPT 生成对比

图 3-10 ──→ 融合方法论（理论可视化）
    │
    ├── 连续特征张量 Z_e (3D 线框)
    ├── 融合公式 Z_fused = M·Z_eB + (1-M)·Z_eA
    └── 高斯软掩膜 M (2D 热力图)

图 3-11 ──→ 融合效果验证（对偶消融）
    │
    ├── VQ-then-Fuse (离散拼接) — 锯齿边界
    └── Fuse-then-VQ (连续融合) — 丝滑过渡  ★ 核心创新
```

**核心故事线**: 图 3-8 证明两个先验模型都成功收敛 → 图 3-9 揭示离散自回归生成的拼接伪影问题 → 图 3-10 提出连续潜空间掩膜融合的方案 → 图 3-11 通过消融对比证明"先融合再量化"的优越性。
