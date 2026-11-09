% figure_3_2_contrast_stretch.m
% 图 3.2: 碳酸盐岩数字切片自适应对比度拉伸对比图
% ============================================================================
%
% 【数据来源】
%   数据文件: data/real_images/1633-1663-1203-50um0000.tif
%   数据类型: 16-bit 灰度 TIFF（碳酸盐岩 CT 扫描切片）
%   扫描分辨率: 50 μm/像素
%   切片数量: 0000 ~ 0090, 0096 共 96 张连续切片
%
%   ——请填写以下元信息（论文中数据来源小节需要引用）——
%   岩心来源: [待填写: 地层/深度/采样位置]
%   岩性描述: [待填写: 如 四川盆地XX组碳酸盐岩]
%   CT 设备:   [待填写: 型号/厂商]
%   扫描参数: [待填写: 电压/电流/曝光时间/放大倍率]
%   数据提供: [待填写: 实验室/导师/合作单位]
%
% ============================================================================
% 核心红线：
%   1. 读取真实 16-bit .tif，绝不伪造数据
%   2. stretchlim + 线性映射，忽略 1% 极值噪点后拉伸
%   3. 四张子图各自独立输出，无标题，保留横纵坐标
%   4. 直方图：imhist + stem 蓝色针状细线 (64 bins)
%   5. Y 轴禁用科学计数法 (YAxis.Exponent = 0)
%   6. 统一 xlim([0, 1])
% ============================================================================

clear; clc; close all;

% ===== 0. 项目路径 =====
project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
output_dir = fullfile(project_root, 'results', 'evaluation');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% ===== 1. 读取真实 16-bit 图像 =====
actual_file = fullfile(project_root, 'data', 'real_images', '1633-1663-1203-50um0000.tif');

if ~isfile(actual_file)
    error('数据文件不存在: %s', actual_file);
end

f = imread(actual_file);
fprintf('[MATLAB] 图像读取成功: %dx%d, dtype=%s\n', ...
    size(f,1), size(f,2), class(f));
fprintf('         min=%d, max=%d, mean=%.1f\n', ...
    min(f(:)), max(f(:)), mean(f(:)));

% ===== 2. 核心算法：stretchlim + 线性映射 =====
f = im2double(f);

limits = stretchlim(f, [0.01 0.99]);
a = limits(1);
b = limits(2);

fprintf('[MATLAB] 绝对 min/max:      [%.6f, %.6f]\n', min(f(:)), max(f(:)));
fprintf('[MATLAB] stretchlim 边界:   [%.6f, %.6f]\n', a, b);

g = (f - a) / (b - a);
g(g < 0) = 0;
g(g > 1) = 1;

fprintf('[MATLAB] 拉伸后范围:        [%.6f, %.6f]\n', min(g(:)), max(g(:)));

% ===== 3. 分四张图独立导出 =====

% --- Figure 3.2a: 原始切片图像 ---
fig_a = figure('Position', [100, 100, 600, 500]);
imshow(f, []);
set(gca, 'XTick', [], 'YTick', []);
exportgraphics(fig_a, fullfile(output_dir, 'Figure_3_2a_original_image.png'), 'Resolution', 600);
fprintf('[MATLAB] 图 3.2a 已保存\n');
close(fig_a);

% --- Figure 3.2b: 拉伸后图像 ---
fig_b = figure('Position', [100, 100, 600, 500]);
imshow(g, []);
set(gca, 'XTick', [], 'YTick', []);
exportgraphics(fig_b, fullfile(output_dir, 'Figure_3_2b_stretched_image.png'), 'Resolution', 600);
fprintf('[MATLAB] 图 3.2b 已保存\n');
close(fig_b);

% --- Figure 3.2c: 原始切片灰度直方图 ---
fig_c = figure('Position', [100, 100, 800, 500]);
[counts_f, x_f] = imhist(f, 256);
stem(x_f, counts_f, 'Color', 'b', 'Marker', 'none', 'LineWidth', 0.5);
xlim([0, 1]);
ax_c = gca;
ax_c.YAxis.Exponent = 0;
xlabel('归一化像素值', 'FontSize', 12);
ylabel('频数', 'FontSize', 12);
set(gca, 'FontSize', 12);
exportgraphics(fig_c, fullfile(output_dir, 'Figure_3_2c_original_histogram.png'), 'Resolution', 600);
fprintf('[MATLAB] 图 3.2c 已保存\n');
close(fig_c);

% --- Figure 3.2d: 拉伸后切片灰度直方图 ---
fig_d = figure('Position', [100, 100, 800, 500]);
[counts_g, x_g] = imhist(g, 256);
stem(x_g, counts_g, 'Color', 'b', 'Marker', 'none', 'LineWidth', 0.5);
xlim([0, 1]);
ax_d = gca;
ax_d.YAxis.Exponent = 0;
xlabel('归一化像素值', 'FontSize', 12);
ylabel('频数', 'FontSize', 12);
set(gca, 'FontSize', 12);
exportgraphics(fig_d, fullfile(output_dir, 'Figure_3_2d_stretched_histogram.png'), 'Resolution', 600);
fprintf('[MATLAB] 图 3.2d 已保存\n');
close(fig_d);

fprintf('[MATLAB] 四张独立子图全部导出完毕。\n');
exit;
