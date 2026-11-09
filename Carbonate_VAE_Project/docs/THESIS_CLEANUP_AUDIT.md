## Material Passport

- 对照文档：`E:\lyf\bishe\刘一锋\毕业论文终稿-刘一锋.docx`。
- 审查项目：`E:\lyf\bishe\刘一锋\Carbonate_VAE_Project`。论文与图件对应清单沿用已核对的同版原项目；当前提交副本没有 CT 图像文件，本次未重新训练或重算实验数值。
- 范围：终稿正文与公式、图题、嵌入图像、当前代码导入关系、权重与训练记录；未改动论文，未运行训练，未删除项目文件。
- 方法：本地读取 DOCX 的 OOXML，提取 306 个含正文文本的非空段落、4 张表、25 个媒体文件；读取公式文本补充参数；对项目图件进行字节哈希、可处理图像的感知指纹筛选与人工缩略图核对，对同大小的权重文件比较 SHA-256。
- 证据边界：没有发现与嵌入图像字节完全相同的项目文件；下文区分视觉对应、算法与布局对应、尚未确定来源。部分超大图件未解码做感知比较，因此不能把“没有匹配”解释为“没有使用”。未独立复现实验数值。

# 论文对照清理审查

结论：可以从主项目中移出教学示例、空笔记本、一次性诊断、未采用的绘图版本和多数中间检查点。标准 VAE、基线、敏感性实验、PixelCNN 和 NanoGPT 都进入了终稿的论证，应保留对应实现、最终权重与数据记录。

文中的“可归档”表示不必放在精简交付目录中，建议保留一份研究备份；“待核对”表示现有证据不足以支持删除。下面的文件名均按当前整理后的目录给出。

## 终稿与项目的主要对应关系

| 论文内容 | 应保留的代码或资源 | 核对结果 |
| --- | --- | --- |
| 第 2 章 VQ-VAE、复合损失、连续潜空间融合 | `models/vae_model.py`、`training/train_thesis.py`、`generation/generate_fusion_thesis.py` | 算法主体对应；这里以及下表省略的代码前缀为 `thesis_code/` |
| 3.1 数据读取与图 3.2 对比度拉伸 | `utils/data_loader.py`、`figure_scripts/figure_3_2_contrast_stretch.m`、`export_for_matlab.py`、`figure_2_2_histogram.m`；两套原始数据目录 | 终稿四张子图与 `Figure_3_2a/b/c/d_*.png` 的内容对应；灰度变换版本仍需区分 |
| 图 3.4 多分量训练曲线 | `figure_scripts/plot_training_curves.py`、`results/training_history.npz` | 终稿图像视觉对应 `results/evaluation/training_convergence_curve.png`，不是仅凭脚本编号判断 |
| 3.3.3 与图 3.5，权重 10、20、40 的敏感性实验 | `training/train_sensitivity.py`、`evaluation/eval_sensitivity_models.py`、`evaluation/eval_one_model.py`、`figure_scripts/figure_3_3_sensitivity.py` | 图像视觉对应 `Figure_3_3_Sensitivity.png`；20 的模型来自主实验，10 和 40 位于 `sensitivity_checkpoints/` |
| 图 3.6 基础与复合模型残差 | `training/train_baseline.py`、`figure_scripts/figure_3_6_residual.py`；基线和主模型最终权重 | 终稿图像视觉对应 `Figure_3_6_Residual.png` |
| 图 3.7 两模型损失对比 | `figure_scripts/figure_3_7_loss.py`、`results/baseline_history.npz`、`results/training_history.npz` | 终稿图像视觉对应 `Figure_3_7_Loss_Curves.png` |
| 3.4.1 与图 3.8 先验训练 | 两个 PixelCNN 训练版本、`training/train_gpt_prior.py`、`preprocessing/extract_codes.py`、`models/nanogpt.py` | 终稿文字写两个先验，但嵌入图视觉对应单独的 `Figure_3_9_PixelCNN_Loss.png`，见后文 |
| 图 3.9 PixelCNN 与 NanoGPT 伪影 | `figure_scripts/figure_3_9_3_11_combined.py`、`figure_scripts/figure_4_5_retrained.py` | 后者被前者导入，用于构建与采样；即使文件名带旧图号，也不能删除 |
| 图 3.10 连续张量与软掩膜 | `figure_scripts/figure_3_10_tensor_mask.py` | 题意与输出 `Figure_3_10_Tensor_Mask.png` 对应，超大嵌入图未逐像素确认 |
| 图 4.1 四面板训练曲线 | `figure_scripts/final_evaluation_charts.py`、`results/training_history.npz` | 终稿视觉对应 `results/evaluation/loss_curves.png`；同编号的 `figure_4_1_training_loss.py` 是另一版布局 |
| 图 4.2 标准 VAE 与 VQ-VAE | `models/standard_vae.py`、`figure_scripts/figure_4_2_comparison.py`、标准 VAE 的两份最终权重 | 终稿视觉对应 `Figure_4_2_VAE_vs_VQVAE_Comparison.png`，不能删除标准 VAE 分支 |
| 图 4.3 固定 σ=4、thr=0.75 的 16 个样本 | `figure_scripts/figure_4_5_fusion_novel_rocks.py` | 参数、布局和方法对应，但现存 PNG 的随机样本与终稿不同，不能声称是原图的字节复现 |
| 图 4.4 多尺度和阈值 3×3 对比 | `figure_scripts/figure_4_6_diversity.py` | σ 为 2、4、6，thr 为 0.65、0.75、0.85，参数与布局对应；现存 PNG 的样本也与终稿不同 |

第 2 章和第 3 章的架构示意图有后期制图或排版痕迹；尚未一一确认源脚本。`docs/figures/` 和 `figure_scripts/diagrams/` 宜作为制图素材保留，不能按“模型运行不需要”直接删除。

## 可以从主交付目录移出的内容

### 明确不属于论文主体的材料

- `examples/mnist/`：MNIST 教学实验，终稿没有使用 MNIST 数据或其输出。
- `notebooks/visualization.ipynb`：只有一个空代码单元，无运行结果。
- `thesis_code/diagnostics/` 中除 `check_env.py` 外的一次性诊断：权重结构打印、旧 VAE 名称排查、坍塌排查、PixelCNN 分布诊断等。当前终稿对应的绘图和训练入口不导入这些工具。
- `thesis_code/models/manual_load_vae.py`：旧权重修复工具；当前标准 VAE 对比图直接加载 `standard_vae_*_beta.weights.h5`，无需该工具。与旧诊断一起归档。
- `results/sanity_test/`、`results/topology_test/`、`results/v6_sanity/`：早期调试图，非终稿展示结果。
- `__pycache__/`、`*.pyc`：可重新生成的 Python 缓存。
- `docs/archive/temp_weight_output.txt`：旧权重打印文本；历史文件清单也只作备查。

### 旧方案或重复绘图版本

以下脚本可以移入历史实验归档；这不等于否定其研究价值：

- `generation/generate_novel_thesis.py`：单图潜变量扰动方案；`generate_unconditional_thesis.py`：旧 PixelCNN 无条件采样。终稿主要融合生成及图 3.9 的复现不通过这两个入口。
- `generation/demo_for_advisor.py`、`batch_generate_random_rocks.py`：额外演示入口；终稿图 4.3、4.4 采用的是专用论文绘图方案。
- `figure_scripts/figure_2_1_latent_space.py`：额外的潜空间散点示意，和终稿图 2.1 的 VAE 流程示意不同。
- `figure_scripts/figure_2_2_minmax_stretch.py`、`figure_2_2_cn.py`：对比度图的其他版本；已保留 MATLAB 四子图来源后可归档。
- `figure_scripts/figure_3_4_loss_curves.py`：另一个训练曲线版本；终稿图 3.4 与 `plot_training_curves.py` 的输出对应。
- `figure_scripts/figure_3_X_sensitivity.py`：包含 0、10、20、40 四个权重；终稿采用 10、20、40 三组的 `figure_3_3_sensitivity.py`。
- `figure_scripts/figure_3_10_smoothing.py`：边缘软化对比图；终稿图 3.10 是张量与掩膜示意。
- `figure_scripts/figure_4_1_training_loss.py`：和终稿四面板训练图不同的布局。
- `figure_scripts/figure_4_2_feature_learning.py`：仅 VQ-VAE 的表达对比，终稿采用包含标准 VAE 的对比图。
- `figure_scripts/figure_4_5_rescue_gridsearch.py`：旧 PixelCNN 温度与筛选参数救援搜索。
- `figure_scripts/figure_4_7_ablation.py`：额外的拼接与融合边界实验，终稿没有独立的图 4.7。

旧绘图输出、`matlab_*.log` 和诊断图可以随对应脚本归档。训练日志、原始数值 `.npz` 和最终权重应优先保留，不能把它们当作普通调试日志整批移除。`results/generated_images/` 的逐轮网格图未作为终稿的核心图件，可保留少数里程碑后归档其余版本。

## 不宜整批删除的内容

- `data/raw/CT/` 与 `data/real_images/`：前者用于主训练，后者被多份终稿绘图脚本直接使用。
- `results/checkpoints/encoder_ep50.weights.h5`、`decoder_ep50.weights.h5`、`vq_layer_ep50.npy`：主模型。
- `results/baseline_checkpoints/` 的最终模型，以及 `baseline_history.npz`：图 3.6、3.7 的证据。
- `results/sensitivity_checkpoints/lam_10.0/`、`lam_40.0/` 的最终模型、`history.npz`、`eval_result.npz`，以及 `results/evaluation/sensitivity_eval.npz`：图 3.5 的复核基础。
- `results/checkpoints/standard_vae_enc_beta.weights.h5`、`standard_vae_dec_beta.weights.h5`：图 4.2 需要。现有最终文件与第 15 轮快训文件字节相同，快训脚本也会生成这些文件；保留 `train_standard_vae_quick.py`，全量训练脚本作为备选来源保留，不应直接以“重复训练”删除两者。
- `results/checkpoints/pixelcnn_history.npz`、`pixelcnn_retrain_history.npz`、`gpt_prior_v2_history.npz`：终稿图题与实际图像存在版本差异，应先保全两套 PixelCNN 证据。
- `pixelcnn_retrain.weights.h5`、`gpt_prior_v2.pt`、`valid_tokens.npy`、`latent_codes.npy`：图 3.9 的依赖。原版 `pixelcnn.weights.h5` 暂留，用于核对图 3.8 及早期采样来源。
- `figure_scripts/figure_4_5_retrained.py`：虽然旧文件名容易误判，但图 3.9 的组合脚本直接导入其中两个函数。
- `figure_scripts/figure_4_3_4_5_evaluation.py`、`figure_ablation.py`、`run_fig_4_3_4.py` 等 S2 评估代码：终稿没有对应的第 4 章定量图，却在 2.5 和 4.3 讨论 S2 与尺度标定。先作为补充证据保留，尚不能证明这些叙述已通过现有脚本完成端到端标定。
- `docs/defense/`：不参与模型运行，但用于答辩讲解，适合单独保留。
- 根目录 `nanogpt.py`：旧 GPT 权重的反序列化兼容入口，不能作为重复模型实现删除。

## 大文件精简的候选范围

优先保留最终权重和数值记录，再考虑下列历史权重。具体文件和容量列在后面的自动盘点表中，容量使用 MiB。

| 候选组 | 条件与建议 |
| --- | --- |
| `gpt_prior_ep*.pt` | 旧 V1 的中间快照，可归档；暂留 `gpt_prior.pt` 便于来源追溯 |
| `gpt_prior_v2_ep*.pt` | V2 的中间快照；保留实际被加载的 `gpt_prior_v2.pt` 与历史数组后可归档 |
| `standard_vae_*_beta_ep*.weights.h5` | 保留无 epoch 后缀的最终编码器、解码器后，可归档这些中间副本；第 15 轮副本已确认与最终文件完全相同 |
| 主模型、基线及敏感性实验的非最终 epoch 权重 | 终稿未展示逐 epoch 模型效果，保留最终模型与损失记录后可归档；归档意味着不再就地复核该轮模型输出 |
| `encoder.weights.h5`、`decoder.weights.h5`、`vae_carbonate.weights.h5` | 是早期标准 VAE 权重，不是主 VQ-VAE 第 50 轮权重的另一种文件名；不要误当成主模型重复文件。确认无需旧 VAE 修复实验后可另行归档 |

另发现 11 组字节相同的检查点。部分基线和敏感性脚本分别使用带 epoch 与不带 epoch 的名字，因此不能只按哈希相同就任删一个路径。下文保留每组的全部路径，便于后续统一加载方式后再去重。

## 清理前需保留证据的几处差异

1. **图 3.8 图题与插图不一致。** 终稿图题写 PixelCNN 与 NanoGPT 对比；实际嵌入图可见标题与图例只有 PixelCNN 的训练和验证曲线，视觉对应 `Figure_3_9_PixelCNN_Loss.png`。`Figure_3_8_Loss_Comparison.png` 才是当前项目中的两模型对比图。不要把前者当作“旧图号文件”删掉。
2. **学习率记录不同。** 论文 3.2.2 写初始学习率 `1e-4`；主训练和基线脚本使用 `CosineDecay(1e-3, ...)`。本次仅记录差异，没有改论文或训练参数。
3. **框架与取特征方式不同。** 论文 3.4.2 写 PyTorch 与 Hook；实际主融合是 TensorFlow/Keras 的编码器输出，再经 VQ 与解码器。PyTorch 在这里用于 GPT 先验，不能据此删掉 TensorFlow 依赖。
4. **部分插图是不同随机运行的产物。** 图 4.3、4.4 与专用融合脚本的参数、布局一致，但终稿中实际样本不同于当前目录 PNG。应保留原论文中的插图，不能以重运行的新图替换并声称完全复现。
5. **“孔隙度”口径要保留说明。** 论文 2.2.3 本身说明训练约束采用平均灰度差；这不等同于独立阈值分割得到的面孔率。敏感性图脚本固化了数值，因此对应评估程序、数值数组和原始数据比额外美化图更应保留。

## 本次附带修复

上次目录整理生成的 `PROJECT_STRUCTURE.md` 中文被替换为问号，已修复。另检查到已有 GPT 检查点序列化引用 `nanogpt.GPTConfig`；本次补充根目录兼容模块，使旧权重可在新包结构下解析。没有改变模型实现或重新保存权重。

## 逐文件与容量盘点

以下附录由当前目录、代码导入与文件大小生成。`KEEP` 表示论文复现或项目基础依赖；`ARCHIVE` 表示可从主交付移出；`REVIEW` 表示补充证据、待确认来源或可选工具。空的包初始化文件不单列，随所属代码包保留。

| 类别 | 数量 |
| --- | --- |
| KEEP | 32 |
| ARCHIVE | 28 |
| REVIEW | 15 |

| 状态 | 文件 | 对应内容或理由 |
| --- | --- | --- |
| KEEP | `thesis_code/config.py` | project paths |
| REVIEW | `thesis_code/diagnostics/check_env.py` | supporting evidence / illustration / alternative tool |
| ARCHIVE | `thesis_code/diagnostics/check_standard_vae.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/check_weight_loading.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/diagnose_code_map.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/diagnose_pixelcnn.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/diagnose_vae_collapse.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/inspect_h5.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/inspect_legacy_vae.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/inspect_legacy_weights.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/inspect_vae_weight_names.py` | one-off diagnostic; no paper entry imports this |
| ARCHIVE | `thesis_code/diagnostics/inspect_vqvae_weights.py` | one-off diagnostic; no paper entry imports this |
| KEEP | `thesis_code/evaluation/eval_one_model.py` | Section 3.3.3 numeric evidence |
| REVIEW | `thesis_code/evaluation/eval_physics.py` | supporting evidence / illustration / alternative tool |
| KEEP | `thesis_code/evaluation/eval_sensitivity_models.py` | Section 3.3.3 numeric evidence |
| REVIEW | `thesis_code/evaluation/evaluate_thesis.py` | supporting evidence / illustration / alternative tool |
| REVIEW | `thesis_code/evaluation/verify_ablation.py` | supporting evidence / illustration / alternative tool |
| ARCHIVE | `thesis_code/figure_scripts/diagnose_histogram_shape.m` | one-off histogram diagnostic |
| REVIEW | `thesis_code/figure_scripts/diagrams/draw_feature_fusion_hd.py` | supporting evidence / illustration / alternative tool |
| REVIEW | `thesis_code/figure_scripts/diagrams/draw_fusion_pipeline_hd.py` | supporting evidence / illustration / alternative tool |
| REVIEW | `thesis_code/figure_scripts/diagrams/draw_original_style_pipeline.py` | supporting evidence / illustration / alternative tool |
| REVIEW | `thesis_code/figure_scripts/diagrams/draw_pipeline_flowchart.py` | supporting evidence / illustration / alternative tool |
| REVIEW | `thesis_code/figure_scripts/diagrams/draw_vqvae_architecture.py` | supporting evidence / illustration / alternative tool |
| KEEP | `thesis_code/figure_scripts/export_for_matlab.py` | Fig 3.2 export candidate |
| ARCHIVE | `thesis_code/figure_scripts/figure_2_1_latent_space.py` | unused latent scatter layout |
| ARCHIVE | `thesis_code/figure_scripts/figure_2_2_cn.py` | alternative contrast layout |
| KEEP | `thesis_code/figure_scripts/figure_2_2_histogram.m` | Fig 3.2 histogram source candidate |
| ARCHIVE | `thesis_code/figure_scripts/figure_2_2_minmax_stretch.py` | alternative contrast layout |
| ARCHIVE | `thesis_code/figure_scripts/figure_3_10_smoothing.py` | alternative smoothing illustration |
| KEEP | `thesis_code/figure_scripts/figure_3_10_tensor_mask.py` | Fig 3.10 source candidate |
| KEEP | `thesis_code/figure_scripts/figure_3_2_contrast_stretch.m` | Fig 3.2 four-panel source candidate |
| KEEP | `thesis_code/figure_scripts/figure_3_3_sensitivity.py` | Fig 3.5 |
| ARCHIVE | `thesis_code/figure_scripts/figure_3_4_loss_curves.py` | alternative loss layout |
| KEEP | `thesis_code/figure_scripts/figure_3_6_residual.py` | Fig 3.6 |
| KEEP | `thesis_code/figure_scripts/figure_3_7_loss.py` | Fig 3.7 |
| REVIEW | `thesis_code/figure_scripts/figure_3_8_loss_comparison.py` | supporting evidence / illustration / alternative tool |
| KEEP | `thesis_code/figure_scripts/figure_3_9_3_11_combined.py` | Fig 3.9; extra Fig 3.11 |
| KEEP | `thesis_code/figure_scripts/figure_3_9_pixelcnn_loss.py` | embedded Fig 3.8 |
| ARCHIVE | `thesis_code/figure_scripts/figure_3_X_sensitivity.py` | four-weight alternative |
| ARCHIVE | `thesis_code/figure_scripts/figure_4_1_training_loss.py` | alternative loss layout |
| KEEP | `thesis_code/figure_scripts/figure_4_2_comparison.py` | Fig 4.2 |
| ARCHIVE | `thesis_code/figure_scripts/figure_4_2_feature_learning.py` | alternative comparison without standard VAE |
| REVIEW | `thesis_code/figure_scripts/figure_4_3_4_5_evaluation.py` | supporting evidence / illustration / alternative tool |
| KEEP | `thesis_code/figure_scripts/figure_4_5_fusion_novel_rocks.py` | Fig 4.3 algorithm/layout |
| ARCHIVE | `thesis_code/figure_scripts/figure_4_5_rescue_gridsearch.py` | prior rescue search |
| KEEP | `thesis_code/figure_scripts/figure_4_5_retrained.py` | imported by Fig 3.9 implementation |
| KEEP | `thesis_code/figure_scripts/figure_4_6_diversity.py` | Fig 4.4 algorithm/layout |
| ARCHIVE | `thesis_code/figure_scripts/figure_4_7_ablation.py` | additional standalone ablation |
| REVIEW | `thesis_code/figure_scripts/figure_ablation.py` | supporting evidence / illustration / alternative tool |
| REVIEW | `thesis_code/figure_scripts/figure_vq_schematic.py` | supporting evidence / illustration / alternative tool |
| KEEP | `thesis_code/figure_scripts/final_evaluation_charts.py` | embedded Fig 4.1; extra evaluation panels |
| KEEP | `thesis_code/figure_scripts/plot_training_curves.py` | Fig 3.4 |
| REVIEW | `thesis_code/figure_scripts/run_fig_4_3_4.py` | supporting evidence / illustration / alternative tool |
| ARCHIVE | `thesis_code/generation/batch_generate_random_rocks.py` | additional batch demo |
| ARCHIVE | `thesis_code/generation/demo_for_advisor.py` | advisor demo |
| KEEP | `thesis_code/generation/generate_fusion_thesis.py` | continuous fusion |
| ARCHIVE | `thesis_code/generation/generate_novel_thesis.py` | single-image latent mutation alternative |
| ARCHIVE | `thesis_code/generation/generate_unconditional_thesis.py` | original PixelCNN sampling alternative |
| ARCHIVE | `thesis_code/models/manual_load_vae.py` | legacy VAE weight recovery |
| KEEP | `thesis_code/models/nanogpt.py` | Section 3.4.1; Fig 3.9 |
| KEEP | `thesis_code/models/standard_vae.py` | Fig 4.2 |
| KEEP | `thesis_code/models/vae_model.py` | core VQ-VAE |
| KEEP | `thesis_code/preprocessing/extract_codes.py` | prior input codes |
| KEEP | `thesis_code/training/train_baseline.py` | Fig 3.6 and Fig 3.7 baseline |
| KEEP | `thesis_code/training/train_gpt_prior.py` | Section 3.4.1; Fig 3.9 |
| KEEP | `thesis_code/training/train_pixelcnn.py` | embedded Fig 3.8 original history |
| KEEP | `thesis_code/training/train_pixelcnn_retrain.py` | Fig 3.9 retrained prior |
| KEEP | `thesis_code/training/train_sensitivity.py` | Fig 3.5 |
| REVIEW | `thesis_code/training/train_standard_vae.py` | supporting evidence / illustration / alternative tool |
| KEEP | `thesis_code/training/train_standard_vae_quick.py` | Fig 4.2; ep15 matches final weights |
| KEEP | `thesis_code/training/train_thesis.py` | main training |
| KEEP | `thesis_code/utils/data_loader.py` | Section 3.1 |
| KEEP | `nanogpt.py` | legacy serialized GPTConfig module compatibility |
| ARCHIVE | `examples/mnist/mnist_vae.py` | MNIST teaching example |
| ARCHIVE | `notebooks/visualization.ipynb` | one empty code cell |

### 中间权重容量

| 分组 | 文件数 | MiB |
| --- | --- | --- |
| GPT V1 intermediate | 4 | 513.44 |
| GPT V2 intermediate | 9 | 338.97 |
| Standard VAE intermediate | 6 | 499.85 |
| Main VQ-VAE intermediate | 30 | 23.44 |
| Baseline intermediate | 30 | 23.44 |
| Sensitivity intermediate | 60 | 46.88 |
| Total (non-overlapping candidates) | 139 | 1446.02 |

### 字节相同的检查点

| 组 | 全部路径 |
| --- | --- |
| 1 | `results/baseline_checkpoints/decoder.weights.h5`<br>`results/baseline_checkpoints/decoder_ep50.weights.h5` |
| 2 | `results/sensitivity_checkpoints/lam_10.0/decoder.weights.h5`<br>`results/sensitivity_checkpoints/lam_10.0/decoder_ep50.weights.h5` |
| 3 | `results/sensitivity_checkpoints/lam_40.0/decoder.weights.h5`<br>`results/sensitivity_checkpoints/lam_40.0/decoder_ep50.weights.h5` |
| 4 | `results/baseline_checkpoints/encoder.weights.h5`<br>`results/baseline_checkpoints/encoder_ep50.weights.h5` |
| 5 | `results/sensitivity_checkpoints/lam_10.0/encoder.weights.h5`<br>`results/sensitivity_checkpoints/lam_10.0/encoder_ep50.weights.h5` |
| 6 | `results/sensitivity_checkpoints/lam_40.0/encoder.weights.h5`<br>`results/sensitivity_checkpoints/lam_40.0/encoder_ep50.weights.h5` |
| 7 | `results/checkpoints/standard_vae_dec_beta.weights.h5`<br>`results/checkpoints/standard_vae_dec_beta_ep15.weights.h5` |
| 8 | `results/checkpoints/standard_vae_enc_beta.weights.h5`<br>`results/checkpoints/standard_vae_enc_beta_ep15.weights.h5` |
| 9 | `results/baseline_checkpoints/vq_layer.npy`<br>`results/baseline_checkpoints/vq_layer_ep50.npy` |
| 10 | `results/sensitivity_checkpoints/lam_10.0/vq_layer.npy`<br>`results/sensitivity_checkpoints/lam_10.0/vq_layer_ep50.npy` |
| 11 | `results/sensitivity_checkpoints/lam_40.0/vq_layer.npy`<br>`results/sensitivity_checkpoints/lam_40.0/vq_layer_ep50.npy` |
