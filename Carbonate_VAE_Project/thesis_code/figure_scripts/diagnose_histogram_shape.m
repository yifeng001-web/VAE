% diagnose_histogram_shape.m
% 快速诊断：检查 16-bit CT 数据直方图的真实多峰结构
clear; clc;

project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
filepath = fullfile(project_root, 'data', 'real_images', '1633-1663-1203-50um0000.tif');

f = imread(filepath);
f_d = im2double(f);  % [0.014267, 0.626368]

% stretchlim 有效范围
limits = stretchlim(f_d, [0.01 0.99]);
a = limits(1); b = limits(2);
g = (f_d - a) / (b - a);
g(g<0)=0; g(g>1)=1;

fprintf('原始 16-bit 灰度范围: %d ~ %d (共 %d 个不同灰度级)\n', ...
    min(f(:)), max(f(:)), length(unique(f(:))));
fprintf('归一化后范围: [%.4f, %.4f]\n', min(f_d(:)), max(f_d(:)));
fprintf('stretchlim 有效边界: [%.4f, %.4f]\n', a, b);

% 诊断图: 用不同 bins 数量画直方图，检查多峰结构
figure('Position', [100, 100, 1400, 900]);

bins_list = [32, 64, 128, 256, 512, 1024];
for i = 1:6
    subplot(2, 3, i);
    [cnt, x] = imhist(f_d, bins_list(i));
    bar(x, cnt, 'BarWidth', 0.8, 'FaceColor', 'b', 'EdgeColor', 'none');
    xlim([0, 1]);
    title(sprintf('%d bins', bins_list(i)), 'FontSize', 13);
    ylabel('频数', 'FontSize', 11);
    xlabel('归一化像素值', 'FontSize', 11);
    ax = gca; ax.YAxis.Exponent = 0; ax.FontSize = 11;
end

sgtitle(sprintf('原始切片直方图多尺度诊断 (%.0f 万像素, %d 个不同灰度级)', ...
    numel(f_d)/10000, length(unique(f(:)))), 'FontSize', 14, 'FontWeight', 'bold');

output_path = fullfile(project_root, 'results', 'evaluation', 'diagnose_histogram_multiscale.png');
exportgraphics(gcf, output_path, 'Resolution', 300);
fprintf('诊断图已保存: %s\n', output_path);

% 同时输出统计摘要
fprintf('\n=== 分位数摘要 ===\n');
pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99];
vals = prctile(f_d(:), pcts);
for i = 1:length(pcts)
    fprintf('  P%02d = %.4f\n', pcts(i), vals(i));
end

close all;
exit;
