# 基于 VQ-VAE 的碳酸盐岩 CT 切片重构与生成

这个项目用深度学习处理碳酸盐岩的二维 CT 切片，主要完成两件事：学习并重建真实岩石的灰度与孔隙结构，以及融合不同切片的潜在特征，生成新的岩石切片图像。代码用于毕业论文实验，包含 VQ-VAE 主模型、连续潜空间融合方法、标准 VAE 和自回归先验的对比实验，以及论文绘图与诊断工具。

这是整个仓库的统一使用说明。主要 Python 代码位于 `Carbonate_VAE_Project/`；运行命令前，需要先进入该目录。

## 个人工作

在导师提供研究方向与 CT 数据的基础上，主要完成以下工作：

- **数据处理与模型训练**：完成 TIFF 切片读取、256×256 单通道预处理与归一化，开展 VAE、VQ-VAE 训练和推理，调整学习率、batch size 与联合损失配置，保存权重和训练记录。
- **连续潜空间融合**：提出在量化前的连续特征上进行软掩膜融合的思路，结合直方图匹配与掩膜参数调整，观察生成切片的边界和纹理变化。
- **重构评估与结果整理**：结合 MAE、MS-SSIM、平均灰度差和 Sobel 边缘差分析重构表现，整理训练曲线及原图、重构图、融合图。

模型与实验脚本基于已有 VAE/VQ-VAE 方法实现，开发过程中使用 AI 辅助编码与调试。

## 项目在做什么

主流程如下：

```text
真实 TIFF 切片
    → 调整为 256×256 单通道图像并归一化
    → 编码器提取 32×32×64 的连续特征
    → VQ 码本量化
    → 解码器重建 256×256 图像
```

生成新切片时，先对两张真实切片进行灰度分布匹配，再分别编码，用随机高斯平滑掩膜混合量化前的连续特征，最后经过 VQ 量化与解码。PixelCNN 和 GPT 用于离散先验采样的对比实验，主融合方法本身不依赖这两个先验网络。

模型目前处理二维灰度图像。训练中命名为 `porosity_loss` 的损失计算的是平均灰度差；它与阈值分割得到的孔隙面积比例需要分别解释。

## 效果展示

以下图件来自已有实验输出。

### 原图与重建图

![真实切片与 VQ-VAE 重建对比](Carbonate_VAE_Project/results/evaluation/high_res_comparison.png)

### 连续潜空间融合

![不同真实切片的连续特征融合示例](Carbonate_VAE_Project/results/evaluation/perfect_fusion_generation.png)

训练收敛与权重敏感性分析也随仓库提供：

- [主模型训练收敛曲线](Carbonate_VAE_Project/results/evaluation/training_convergence_curve.png)
- [平均灰度差约束的权重敏感性图](Carbonate_VAE_Project/results/evaluation/Figure_3_3_Sensitivity.png)（历史图件中的 `porosity` 标签按下文解释）

## 实验结果如何理解

- **训练记录**：`results/training_history.npz` 保存 50 轮训练指标；首页收敛图由 `figure_scripts/plot_training_curves.py` 读取该文件绘制。训练损失反映对训练数据的拟合情况。
- **评估范围**：`eval_sensitivity_models.py` 和 `eval_one_model.py` 从与训练相同的 CT 数据目录中抽取至多 200 张切片，固定随机种子为 999；未排除训练样本。这些结果属于抽样重构评估，不能作为独立测试集上的泛化成绩。正式验证需要先按岩样或扫描体积划分互不重叠的数据，再重新训练和评估。
- **指标含义**：代码及历史图件中的 `porosity_loss`、`por_dev` 计算的是平均灰度差，不等于经孔隙分割或物理测量得到的孔隙度误差。融合示例展示视觉效果，尚不能据此证明生成切片的物理性质。
- **历史绘图入口**：旧版 `final_evaluation_charts.py` 的 `generate_loss_curves()` 曾使用模拟数据绘制曲线。本次将该函数改为读取真实 `training_history.npz`，并移除没有统计依据的阴影带；旧版脚本生成的曲线不能作为训练记录。本次没有重新训练模型，也没有用新图替换历史论文图件。

## 目录说明

```text
仓库根目录/
├── README.md                         # 本说明
├── .gitignore                        # Git 上传忽略规则
└── Carbonate_VAE_Project/
    ├── thesis_code/
    │   ├── config.py                 # 根据代码位置定位项目目录
    │   ├── models/                   # VQ-VAE、标准 VAE、GPT
    │   ├── utils/                    # TIFF 解码和数据加载
    │   ├── preprocessing/            # 离散潜码提取
    │   ├── training/                 # 主模型、先验、基线与敏感性训练
    │   ├── generation/               # 融合生成与采样演示
    │   ├── evaluation/               # 重建与指标评估
    │   ├── figure_scripts/           # 论文绘图和架构示意图
    │   └── diagnostics/              # 环境检查和历史诊断
    ├── docs/                         # 图件、答辩材料与技术核查记录
    ├── examples/mnist/               # 独立的 VAE 教学实验
    ├── notebooks/                    # 探索性笔记本
    ├── data/                         # 本地 CT 数据，不随 Git 上传
    ├── results/                      # 仅上传精选图件、训练记录和最终主模型权重
    ├── nanogpt.py                    # 旧 GPT 权重的模块名兼容入口
    ├── requirements.txt              # 主流程依赖
    └── requirements-gpt.txt          # GPT 实验附加依赖
```

## 第一次使用

### 1 进入代码目录

用 VS Code 打开仓库根目录，在“终端 → 新建终端”中执行：

```powershell
cd Carbonate_VAE_Project
```

后面的 Python 命令均在这个目录运行。若已经打开了 `Carbonate_VAE_Project` 文件夹，可以直接执行后面的命令。

### 2 准备 Python 环境

参考现有实验环境使用 Python 3.10、TensorFlow 2.10 和 NumPy 1.x。已有可用环境时，可以直接激活它；也可以使用 Conda 新建环境：

```powershell
conda create -n carbonate-vae python=3.10 -y
conda activate carbonate-vae
python -m pip install -r requirements.txt
python -m thesis_code.diagnostics.check_env
```

环境检查会显示依赖版本、项目路径和 TensorFlow 检测到的 GPU 数量。部分现有生成和先验训练脚本会强制检查 GPU，检测不到时会停止。

只有运行 GPT 对比实验时才需要安装附加依赖：

```powershell
python -m pip install -r requirements-gpt.txt
```

### 3 准备数据与权重

当前提交副本的 `data/` 目录为空，需要自行准备 CT 图像。仓库随附第 50 轮 VQ-VAE 的编码器、解码器和码本；标准 VAE、PixelCNN、GPT 等其他实验权重仍需另行取得或自行训练。资源放置位置如下：

| 资源 | 放置位置 | 用途 |
| --- | --- | --- |
| 主 CT 数据集，TIFF 格式 | `data/raw/CT/`，允许嵌套子目录 | 主训练、重建和融合 |
| 历史实验小样本集 | `data/real_images/` | 部分论文绘图与标准 VAE 快训 |
| VQ-VAE 编码器权重 | `results/checkpoints/encoder_ep50.weights.h5` | 重建与融合 |
| VQ-VAE 解码器权重 | `results/checkpoints/decoder_ep50.weights.h5` | 重建与融合 |
| VQ 码本 | `results/checkpoints/vq_layer_ep50.npy` | 重建与融合 |

主数据加载器会检查至少 4000 张 TIFF，因此使用该加载器的训练和重建入口需要满足这个数量要求。手动读取切片的融合入口不经过这一数量检查。

克隆仓库后，随附的三份 VQ-VAE 文件已位于脚本预期的位置，合计约 2.35 MiB。推理仍需要真实 CT 输入图像；本项目没有自动下载 CT 数据或其他模型权重的入口。

### 4 生成或重建切片

具备 CT 数据、上述三份权重和可用 GPU 后，先运行主融合入口：

```powershell
python -m thesis_code.generation.generate_fusion_thesis
```

生成的对比图保存为：

```text
results/evaluation/perfect_fusion_generation.png
```

使用主数据集检查原图与重建图：

```powershell
python -m thesis_code.evaluation.evaluate_thesis
```

输出为 `results/evaluation/high_res_comparison.png`。这些脚本会写入固定的结果路径，重复运行会覆盖同名图件。

### 5 从头训练主模型

没有权重时，准备主 CT 数据集后运行：

```powershell
python -m thesis_code.training.train_thesis
```

现有默认设置是 batch size 4、训练 50 轮、码本大小 512、特征维度 64。损失包含 VQ、MAE、MS-SSIM、平均灰度差和 Sobel 边缘差。训练产生的权重、过程图和历史数组保存在 `results/` 中；参数保留在训练脚本内。

## 训练参数与随附文件

主模型参数以 `thesis_code/training/train_thesis.py` 和 `thesis_code/models/vae_model.py` 为准：

| 参数 | 值 |
| --- | --- |
| 输入图像 | 256×256，单通道，逐切片 Min-Max 归一化 |
| Batch size | 4 |
| 训练轮次 | 50 |
| 优化器与学习率 | Adam；初始学习率 `1e-3`，CosineDecay，`alpha=0.01` |
| 连续特征 | 32×32×64 |
| 码本规模 | 512 个向量，每个 64 维 |
| 承诺损失系数 β | 0.25 |
| 损失权重 | VQ：1；MAE：10；MS-SSIM：5；平均灰度差：20；Sobel 边缘差：10 |

仓库只提供以下 9 个结果文件，合计约 4.64 MiB。它们保留原来的运行路径，不需要重命名或复制到其他目录。

| 类别 | 文件 |
| --- | --- |
| 最终编码器 | [`encoder_ep50.weights.h5`](Carbonate_VAE_Project/results/checkpoints/encoder_ep50.weights.h5) |
| 最终解码器 | [`decoder_ep50.weights.h5`](Carbonate_VAE_Project/results/checkpoints/decoder_ep50.weights.h5) |
| 最终 VQ 码本 | [`vq_layer_ep50.npy`](Carbonate_VAE_Project/results/checkpoints/vq_layer_ep50.npy) |
| 主模型训练记录 | [`training_history.npz`](Carbonate_VAE_Project/results/training_history.npz) |
| 基线训练记录 | [`baseline_history.npz`](Carbonate_VAE_Project/results/baseline_history.npz) |
| 代表性图件 | 上面的两张效果图、训练收敛曲线与权重敏感性图 |

训练记录可用 NumPy 的 `np.load(..., allow_pickle=False)` 读取。中间 epoch 权重、其他模型、批量出图与诊断日志继续留在本地，不随普通 Git 提交。

## 其他实验怎么运行

| 实验 | 命令 | 前置条件 |
| --- | --- | --- |
| 提取离散潜码 | `python -m thesis_code.preprocessing.extract_codes` | 主数据与 VQ-VAE 权重 |
| PixelCNN 先验训练 | `python -m thesis_code.training.train_pixelcnn` | `results/checkpoints/latent_codes.npy` |
| 加强版 PixelCNN 训练 | `python -m thesis_code.training.train_pixelcnn_retrain` | 同上 |
| GPT 先验训练 | `python -m thesis_code.training.train_gpt_prior` | 潜码与 PyTorch 依赖 |
| 视觉损失基线 | `python -m thesis_code.training.train_baseline` | 主 CT 数据集 |
| 标准 VAE 快训 | `python -m thesis_code.training.train_standard_vae_quick` | `data/real_images/` |
| 孔隙度权重敏感性训练 | `python -m thesis_code.training.train_sensitivity --lambda_porosity 10.0` | 主 CT 数据集 |
| 参数多样性图 | `python -m thesis_code.figure_scripts.figure_4_6_diversity` | 小样本数据与主模型权重 |
| 标准 VAE 与 VQ-VAE 对比图 | `python -m thesis_code.figure_scripts.figure_4_2_comparison` | 两类模型的权重与小样本数据 |

潜码、先验权重、标准 VAE 权重之间不能互相替代。论文脚本编号保留了实验阶段的命名，部分与终稿图号不同，具体对应关系见下方的技术核查记录。

## 哪些 Python 文件需要保留

| 类别 | 对应位置 | 如何理解 |
| --- | --- | --- |
| 主流程 | `models/vae_model.py`、`utils/data_loader.py`、`config.py`、主训练与主融合入口 | 训练、重建和融合的基础 |
| 论文对比实验 | 标准 VAE、基线、敏感性训练、PixelCNN、GPT 及潜码提取 | 主融合运行时未必使用，但论文复现需要 |
| 论文绘图 | `figure_scripts/` | 用于绘制论文图件，部分脚本还相互导入函数 |
| 环境与兼容 | `diagnostics/check_env.py`、根目录兼容文件 `nanogpt.py`、各包的 `__init__.py` | 保证环境检查、旧权重加载和包导入 |
| 历史诊断与旧方案 | 其余 `diagnostics/`、部分重复绘图和演示入口 | 可归档，不应仅因为没有被其他模块导入就判断无用 |
| 教学示例 | `examples/mnist/` | 不属于论文主实验，可独立保留或归档 |

因此，有些 `.py` 对日常生成不必要，但仍用于论文对照、历史复核或答辩。本次只统一说明和上传规则，没有删除 Python 源码。逐文件取舍见[论文对照清理审查](Carbonate_VAE_Project/docs/THESIS_CLEANUP_AUDIT.md)。

## gitignore 是做什么的

`.gitignore` 是 Git 的上传忽略规则，决定未被跟踪的哪些文件不进入提交，不会删除本地文件，也不会判断代码是否有用。

- `__pycache__/`：忽略 Python 自动生成的缓存目录。
- `*.py[cod]`：匹配 `.pyc`、`.pyo` 和 `.pyd`，不匹配普通 `.py` 源码。
- 项目内的 `data/`：保留在本地。
- 项目内的 `results/`：默认忽略，仅对白名单中的 4 张图、2 份训练记录和 3 份最终主模型权重开放上传。
- 仓库根目录的 Word、PPT、PDF：论文与答辩原文件留在本地；项目 `docs/figures/` 内的 PDF 示意图仍可上传。

已经被 Git 跟踪的文件不会因为新增忽略规则而自动退出版本管理。此次没有把任何 `.py` 源码加入忽略名单。

## 进一步了解

- [项目结构与旧文件迁移对照](Carbonate_VAE_Project/docs/PROJECT_STRUCTURE.md)
- [论文图件的代码原理](Carbonate_VAE_Project/docs/FIGURE_PRINCIPLES_REPORT.md)
- [论文与代码对应及可归档清单](Carbonate_VAE_Project/docs/THESIS_CLEANUP_AUDIT.md)

这些文件是技术记录，项目简介和上手步骤统一以本 README 为入口。
