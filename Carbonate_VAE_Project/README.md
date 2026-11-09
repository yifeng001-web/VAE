# 碳酸盐岩 CT 图像重建与生成

本项目使用 VQ-VAE 学习碳酸盐岩二维 CT 切片的结构，通过直方图匹配和连续潜空间融合生成新切片，并保留 PixelCNN、GPT 先验以及标准 VAE 的对比实验。

当前提交副本的 `data/` 目录没有 CT 图像文件。运行训练、重建或融合生成前，需要将主数据补充到 `data/raw/CT/`；部分历史绘图脚本另需 `data/real_images/` 小样本集。现有 `results/` 中的权重和实验结果保持原样。

## 目录

```text
Carbonate_VAE_Project/
├── thesis_code/              # Python 代码包
│   ├── config.py             # 项目与资源路径
│   ├── models/               # VQ-VAE、标准 VAE、GPT
│   ├── utils/                # 数据加载
│   ├── preprocessing/        # 潜码提取
│   ├── training/             # 模型、基线与先验训练
│   ├── generation/           # 融合、采样与批量生成
│   ├── evaluation/           # 重建、物理指标与消融评估
│   ├── figure_scripts/       # 论文绘图及 diagrams/ 示意图
│   └── diagnostics/          # 环境与历史实验诊断
├── data/
│   ├── raw/CT/               # 主 CT 数据集
│   └── real_images/          # 历史小样本实验数据
├── results/                  # 原有权重、训练记录与实验图件
├── docs/                     # 目录说明、答辩材料、figures/ 图件
├── examples/mnist/           # 独立 VAE 教学示例
├── notebooks/               # 探索性笔记本
├── requirements.txt          # TensorFlow 主流程依赖
└── requirements-gpt.txt      # GPT 实验的附加依赖
```

完整迁移对照见 [项目目录与迁移说明](docs/PROJECT_STRUCTURE.md)，论文图件说明见 [绘图原理](docs/FIGURE_PRINCIPLES_REPORT.md)，文件取舍见 [终稿对照清理审查](docs/THESIS_CLEANUP_AUDIT.md)。

根目录 `nanogpt.py` 用于兼容旧 GPT 检查点保存的模块名，实际实现位于 `thesis_code/models/nanogpt.py`。

## 环境

参考现有实验环境使用 Python 3.10、TensorFlow 2.10 和 NumPy 1.x。依赖文件用于新环境安装；已有可用的训练环境可以直接进行环境检查。

```powershell
python -m pip install -r requirements.txt
python -m thesis_code.diagnostics.check_env
```

GPT 实验另需 PyTorch：

```powershell
python -m pip install -r requirements-gpt.txt
```

部分原有训练和生成脚本会在启动时强制检查 GPU；环境检查会报告 TensorFlow 实际检测到的设备。GPU 驱动和运行库继续使用原实验环境的配置。

## 主流程

所有命令均在项目根目录执行，使用模块方式启动，无需进入 `thesis_code` 或设置 `PYTHONPATH`。路径由 `thesis_code/config.py` 定位，移动整个项目后无需修改盘符。

1. 将主数据集放入 `data/raw/CT/`，允许分层子目录。当前主加载器要求至少 4000 张 TIFF。
2. 训练 VQ-VAE：

   ```powershell
   python -m thesis_code.training.train_thesis
   ```

3. 使用第 50 轮权重进行融合或重建：

   ```powershell
   python -m thesis_code.generation.generate_fusion_thesis
   python -m thesis_code.evaluation.evaluate_thesis
   ```

已有权重时可以跳过训练。融合与重建需要 `results/checkpoints/` 下的 `encoder_ep50.weights.h5`、`decoder_ep50.weights.h5` 和 `vq_layer_ep50.npy`。结果继续写入 `results/` 下原有位置，重复运行可能覆盖同名图件或权重。

## 其他实验入口

| 用途 | 命令 |
| --- | --- |
| 批量融合 | `python -m thesis_code.generation.batch_generate_random_rocks` |
| 导师演示 | `python -m thesis_code.generation.demo_for_advisor` |
| 提取离散潜码 | `python -m thesis_code.preprocessing.extract_codes` |
| PixelCNN 训练 | `python -m thesis_code.training.train_pixelcnn` |
| GPT 先验训练 | `python -m thesis_code.training.train_gpt_prior` |
| PixelCNN 无条件采样 | `python -m thesis_code.generation.generate_unconditional_thesis` |
| 视觉损失基线 | `python -m thesis_code.training.train_baseline` |
| 标准 VAE 对比 | `python -m thesis_code.training.train_standard_vae` |
| 参数敏感性训练 | `python -m thesis_code.training.train_sensitivity` |
| 孔隙比例与 S2 图件 | `python -m thesis_code.figure_scripts.run_fig_4_3_4` |
| 架构示意图 | `python -m thesis_code.figure_scripts.diagrams.draw_vqvae_architecture` |

PixelCNN 和 GPT 训练均依赖 `results/checkpoints/latent_codes.npy`。它们是额外的生成路线，连续潜空间融合本身无需先验模型。具体参数仍保留在各实验脚本中。

论文绘图保存到 `results/evaluation/`；架构和流程示意图保存到 `docs/figures/`。MATLAB 绘图脚本仍位于 `thesis_code/figure_scripts/`，使用脚本位置推导项目路径。

`diagnostics/` 中的历史诊断可能会加载大模型或生成输出，应按需运行；文件名中的 `check` 不代表自动化测试。`docs/archive/` 的旧清单只用于历史追溯。

## 指标口径

模型处理二维灰度切片。主训练脚本中的 `porosity_loss` 实际计算平均灰度差；论文物理评估脚本另用阈值分割计算孔隙面积比例，二者应分别解释。

## 语法检查

```powershell
python -m compileall -q thesis_code examples docs/defense
```

此命令只检查 Python 语法，不启动训练或重新生成实验结果。
