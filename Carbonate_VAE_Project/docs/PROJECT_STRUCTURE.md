# 项目目录与迁移说明

本次整理调整目录、导入和资源定位，保留原有模型结构、损失函数、训练参数与实验结果。

## 目录职责

- `thesis_code/models/`：VQ-VAE、标准 VAE、GPT 与历史权重加载工具。
- `thesis_code/utils/`：数据读取等共享工具。
- `thesis_code/preprocessing/`：潜码提取。
- `thesis_code/training/`：主训练、基线、先验和敏感性训练。
- `thesis_code/generation/`：融合生成、先验采样、批量生成和演示。
- `thesis_code/evaluation/`：重建、物理指标和消融评估。
- `thesis_code/figure_scripts/`：论文绘图；`diagrams/` 存放架构与流程示意图脚本。
- `thesis_code/diagnostics/`：环境、权重和中间结果诊断，这些脚本不是自动化测试套件。
- `docs/`：说明、答辩讲解源码与静态图件。
- `docs/archive/`：历史清单和调试文本，仅作追溯，不作为当前目录指南。
- `examples/mnist/`：独立 MNIST 教学实验。
- `data/`、`results/`、`notebooks/`：沿用原有资源位置。

## 运行约定

在项目根目录使用 `python -m thesis_code.<子包>.<脚本名>`。例如，原来的 `thesis_code/train_thesis.py` 改为 `python -m thesis_code.training.train_thesis`。

`thesis_code/config.py` 统一定位项目根目录。主数据仍在 `data/raw/CT/`；历史小样本脚本仍使用 `data/real_images/`。

根目录的 `nanogpt.py` 是有意保留的兼容入口：已有 PyTorch 权重以 `nanogpt.GPTConfig` 保存配置对象，反序列化仍需要这个模块名。实际模型实现位于 `thesis_code/models/nanogpt.py`。

终稿与代码的取舍建议见 [论文对照清理审查](THESIS_CLEANUP_AUDIT.md)。

## 文件迁移清单

| 原位置 | 新位置 |
| --- | --- |
| `Feature_Fusion_HD.png` | `docs/figures/Feature_Fusion_HD.png` |
| `Figure_3_1_Architecture_HD.png` | `docs/figures/Figure_3_1_Architecture_HD.png` |
| `Figure_3_1_Architecture_Vector.pdf` | `docs/figures/Figure_3_1_Architecture_Vector.pdf` |
| `Figure_3_2_Fusion_Pipeline_HD.png` | `docs/figures/Figure_3_2_Fusion_Pipeline_HD.png` |
| `Figure_Original_Style_Pipeline_HD.png` | `docs/figures/Figure_Original_Style_Pipeline_HD.png` |
| `PROJECT_FILE_INVENTORY.txt` | `docs/archive/PROJECT_FILE_INVENTORY_2026-05-13.txt` |
| `VQ_VAE_Architecture_HD.png` | `docs/figures/VQ_VAE_Architecture_HD.png` |
| `VQ_VAE_Architecture_Vector.pdf` | `docs/figures/VQ_VAE_Architecture_Vector.pdf` |
| `check_env.py` | `thesis_code/diagnostics/check_env.py` |
| `defense_algorithm_explained.py` | `docs/defense/defense_algorithm_explained.py` |
| `defense_key_algorithm.py` | `docs/defense/defense_key_algorithm.py` |
| `draw_feature_fusion_hd.py` | `thesis_code/figure_scripts/diagrams/draw_feature_fusion_hd.py` |
| `draw_fusion_pipeline_hd.py` | `thesis_code/figure_scripts/diagrams/draw_fusion_pipeline_hd.py` |
| `draw_original_style_pipeline.py` | `thesis_code/figure_scripts/diagrams/draw_original_style_pipeline.py` |
| `draw_pipeline_flowchart.py` | `thesis_code/figure_scripts/diagrams/draw_pipeline_flowchart.py` |
| `draw_vqvae_architecture.py` | `thesis_code/figure_scripts/diagrams/draw_vqvae_architecture.py` |
| `results/view_h5.py` | `thesis_code/diagnostics/inspect_h5.py` |
| `temp_check_weights.py` | `thesis_code/diagnostics/inspect_legacy_weights.py` |
| `temp_inspect_vae.py` | `thesis_code/diagnostics/inspect_legacy_vae.py` |
| `temp_weight_output.txt` | `docs/archive/temp_weight_output.txt` |
| `textbook_reproduce/mnist_vae.py` | `examples/mnist/mnist_vae.py` |
| `textbook_reproduce/outputs/test1.png` | `examples/mnist/outputs/test1.png` |
| `thesis_code/batch_generate_random_rocks.py` | `thesis_code/generation/batch_generate_random_rocks.py` |
| `thesis_code/demo_for_advisor.py` | `thesis_code/generation/demo_for_advisor.py` |
| `thesis_code/diagnose_code_map.py` | `thesis_code/diagnostics/diagnose_code_map.py` |
| `thesis_code/eval_one_model.py` | `thesis_code/evaluation/eval_one_model.py` |
| `thesis_code/eval_physics.py` | `thesis_code/evaluation/eval_physics.py` |
| `thesis_code/eval_sensitivity_models.py` | `thesis_code/evaluation/eval_sensitivity_models.py` |
| `thesis_code/evaluate_thesis.py` | `thesis_code/evaluation/evaluate_thesis.py` |
| `thesis_code/extract_codes.py` | `thesis_code/preprocessing/extract_codes.py` |
| `thesis_code/figure_scripts/FIGURE_PRINCIPLES_REPORT.md` | `docs/FIGURE_PRINCIPLES_REPORT.md` |
| `thesis_code/figure_scripts/diagnose_pixelcnn.py` | `thesis_code/diagnostics/diagnose_pixelcnn.py` |
| `thesis_code/generate_fusion_thesis.py` | `thesis_code/generation/generate_fusion_thesis.py` |
| `thesis_code/generate_novel_thesis.py` | `thesis_code/generation/generate_novel_thesis.py` |
| `thesis_code/generate_unconditional_thesis.py` | `thesis_code/generation/generate_unconditional_thesis.py` |
| `thesis_code/manual_load_vae.py` | `thesis_code/models/manual_load_vae.py` |
| `thesis_code/nanogpt.py` | `thesis_code/models/nanogpt.py` |
| `thesis_code/run_fig_4_3_4.py` | `thesis_code/figure_scripts/run_fig_4_3_4.py` |
| `thesis_code/temp_check_vqvae.py` | `thesis_code/diagnostics/inspect_vqvae_weights.py` |
| `thesis_code/temp_diag_vae_collapse.py` | `thesis_code/diagnostics/diagnose_vae_collapse.py` |
| `thesis_code/temp_diagnose_names.py` | `thesis_code/diagnostics/inspect_vae_weight_names.py` |
| `thesis_code/test_standard_vae.py` | `thesis_code/diagnostics/check_standard_vae.py` |
| `thesis_code/test_weight_loading.py` | `thesis_code/diagnostics/check_weight_loading.py` |
| `thesis_code/train_baseline.py` | `thesis_code/training/train_baseline.py` |
| `thesis_code/train_gpt_prior.py` | `thesis_code/training/train_gpt_prior.py` |
| `thesis_code/train_pixelcnn.py` | `thesis_code/training/train_pixelcnn.py` |
| `thesis_code/train_pixelcnn_retrain.py` | `thesis_code/training/train_pixelcnn_retrain.py` |
| `thesis_code/train_sensitivity.py` | `thesis_code/training/train_sensitivity.py` |
| `thesis_code/train_standard_vae.py` | `thesis_code/training/train_standard_vae.py` |
| `thesis_code/train_standard_vae_quick.py` | `thesis_code/training/train_standard_vae_quick.py` |
| `thesis_code/train_thesis.py` | `thesis_code/training/train_thesis.py` |
| `thesis_code/verify_ablation.py` | `thesis_code/evaluation/verify_ablation.py` |
