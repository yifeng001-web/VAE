% figure_2_2_histogram.m
% MATLAB 脚本：画图 2-2 下方两个直方图 (c) 和 (d)
% 关键设计：两个直方图共享相同的归一化 [0,1] 横坐标，可直接对比拉伸前后分布变化

clear; clc;

% ===== 路径配置 =====
project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mat_path = fullfile(project_root, 'results', 'evaluation', 'figure_2_2_hist_data.mat');
output_dir = fullfile(project_root, 'results', 'evaluation');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% ===== 加载数据 =====
fprintf('[MATLAB] 加载数据: %s\n', mat_path);
data = load(mat_path);
img_float = data.img_float;
img_stretched = data.img_stretched;
p_low = data.p_low;
p_high = data.p_high;
img_min = data.img_min;
img_max = data.img_max;

fprintf('[MATLAB] 数据加载完成\n');
fprintf('         img_float: %dx%d, min=%.1f, max=%.1f\n', ...
    size(img_float,1), size(img_float,2), img_min, img_max);
fprintf('         img_stretched: min=%.4f, max=%.4f\n', ...
    min(img_stretched(:)), max(img_stretched(:)));
fprintf('         P2=%.1f, P98=%.1f\n', p_low, p_high);

% ===== 关键：原始图像也做 min-max 归一化到 [0,1] =====
% 这样两个直方图共享相同的横坐标范围，可直接对比分布变化
img_normalized = (img_float - img_min) / (img_max - img_min);
p_low_norm = (p_low - img_min) / (img_max - img_min);
p_high_norm = (p_high - img_min) / (img_max - img_min);
fprintf('         P2归一化=%.4f, P98归一化=%.4f\n', p_low_norm, p_high_norm);

% ===== 共用参数 =====
N_BINS = 80;
BIN_RANGE = [0.0, 1.0];
XLIM = [-0.02, 1.02];

% ===== 设置 MATLAB 图形参数 =====
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultAxesFontName', 'Helvetica');
set(0, 'DefaultTextFontName', 'Helvetica');

% ===== 绘制 (c) 原始图像 min-max 归一化后直方图 =====
figure('Position', [100, 100, 750, 420], 'Visible', 'off');

h1 = histogram(img_normalized(:), N_BINS, 'Normalization', 'pdf', ...
    'BinLimits', BIN_RANGE, ...
    'FaceColor', [0.36, 0.42, 0.75], ...  % #5c6bc0
    'EdgeColor', [0.16, 0.23, 0.43], ...   % #2a3a6e
    'LineWidth', 0.5);
hold on;

% 标注 P2/P98 截断位置（垂直虚线）
yl = ylim;
plot([p_low_norm, p_low_norm], [0, yl(2)], 'r--', 'LineWidth', 1.5);
plot([p_high_norm, p_high_norm], [0, yl(2)], 'r--', 'LineWidth', 1.5);
text(p_low_norm + 0.02, yl(2)*0.92, sprintf('P_2=%.3f', p_low_norm), ...
    'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
text(p_high_norm + 0.02, yl(2)*0.92, sprintf('P_{98}=%.3f', p_high_norm), ...
    'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
hold off;

xlabel('归一化像素值', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('概率密度', 'FontSize', 14, 'FontWeight', 'bold');
title('(c) 原始图像直方图', 'FontSize', 14, 'FontWeight', 'bold');
xlim(XLIM);
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
set(gca, 'Layer', 'top');
% 去掉黑色外框
set(gca, 'Box', 'off');
set(gca, 'TickDir', 'in');
set(gca, 'TickLength', [0.005, 0.005]);

output_path_c = fullfile(output_dir, 'Figure_2_2_MATLAB_hist_c.png');
print(gcf, '-dpng', '-r300', output_path_c);
fprintf('[MATLAB] (c) 已保存: %s\n', output_path_c);

% ===== 绘制 (d) 拉伸后直方图（相同 bins、相同横坐标）=====
figure('Position', [100, 100, 750, 420], 'Visible', 'off');

h2 = histogram(img_stretched(:), N_BINS, 'Normalization', 'pdf', ...
    'BinLimits', BIN_RANGE, ...
    'FaceColor', [0.18, 0.49, 0.20], ...   % #2e7d32
    'EdgeColor', [0.10, 0.29, 0.12], ...   % #1a4a1e
    'LineWidth', 0.5);

xlabel('归一化像素值', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('概率密度', 'FontSize', 14, 'FontWeight', 'bold');
title('(d) 自适应对比度拉伸后直方图', 'FontSize', 14, 'FontWeight', 'bold');
xlim(XLIM);
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
set(gca, 'Layer', 'top');
% 去掉黑色外框
set(gca, 'Box', 'off');
set(gca, 'TickDir', 'in');
set(gca, 'TickLength', [0.005, 0.005]);

output_path_d = fullfile(output_dir, 'Figure_2_2_MATLAB_hist_d.png');
print(gcf, '-dpng', '-r300', output_path_d);
fprintf('[MATLAB] (d) 已保存: %s\n', output_path_d);

% ===== 同时保存一张完整对比图（1×2 直接比较）=====
figure('Position', [100, 100, 1100, 400], 'Visible', 'off');

subplot(1, 2, 1);
h1b = histogram(img_normalized(:), N_BINS, 'Normalization', 'pdf', ...
    'BinLimits', BIN_RANGE, ...
    'FaceColor', [0.36, 0.42, 0.75], ...
    'EdgeColor', [0.16, 0.23, 0.43], ...
    'LineWidth', 0.5);
hold on;
yl2 = ylim;
plot([p_low_norm, p_low_norm], [0, yl2(2)], 'r--', 'LineWidth', 1.5);
plot([p_high_norm, p_high_norm], [0, yl2(2)], 'r--', 'LineWidth', 1.5);
text(p_low_norm + 0.02, yl2(2)*0.92, sprintf('P_2=%.3f', p_low_norm), ...
    'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
text(p_high_norm + 0.02, yl2(2)*0.92, sprintf('P_{98}=%.3f', p_high_norm), ...
    'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
hold off;
xlabel('归一化像素值', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('概率密度', 'FontSize', 13, 'FontWeight', 'bold');
title('(c) 原始图像直方图', 'FontSize', 14, 'FontWeight', 'bold');
xlim(XLIM);
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
set(gca, 'Layer', 'top');
box on;

subplot(1, 2, 2);
h2b = histogram(img_stretched(:), N_BINS, 'Normalization', 'pdf', ...
    'BinLimits', BIN_RANGE, ...
    'FaceColor', [0.18, 0.49, 0.20], ...
    'EdgeColor', [0.10, 0.29, 0.12], ...
    'LineWidth', 0.5);
xlabel('归一化像素值', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('概率密度', 'FontSize', 13, 'FontWeight', 'bold');
title('(d) 自适应对比度拉伸后直方图', 'FontSize', 14, 'FontWeight', 'bold');
xlim(XLIM);
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);
set(gca, 'Layer', 'top');
box on;

output_path_full = fullfile(output_dir, 'Figure_2_2_MATLAB_histograms.png');
print(gcf, '-dpng', '-r300', output_path_full);
fprintf('[MATLAB] 完整直方图已保存: %s\n', output_path_full);

fprintf('[MATLAB] === 全部完成 ===\n');
fprintf('        关键设计: 两个直方图共享归一化 [0,1] 横坐标 + 相同 bins + 相同 Y 轴\n');
fprintf('        原始图: P2=%.4f, P98=%.4f (红色虚线标记截断位置)\n', p_low_norm, p_high_norm);
close all;
exit;
