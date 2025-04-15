%% 数据增强示例
% 清空环境变量
clear; clc; close all;

% 定义文件路径并加载数据
file_path = "D:\红茶数据2024.0423\工业相机\自然\DATA\tp.mat";
load(file_path);  % 假设文件中变量名为 X (140x39) 和 Y (140x1)

% 检查数据尺寸
disp("原始 X 大小：");
disp(size(X));  % 应为 140x39
disp("原始 Y 大小：");
disp(size(Y));  % 应为 140x1

%% 数据增强参数设置
num_aug = 3;         % 每个样本生成 3 个增强样本
noise_level = 0.05;  % 噪声比例：可理解为每个特征扰动的标准差占原始数值的比例

% 初始化增强数据，先包含原始数据
X_aug = X;
Y_aug = Y;

%% 对每个样本进行数据增强
num_samples = size(X, 1);
for i = 1:num_samples
    for j = 1:num_aug
        % 对当前样本添加随机高斯噪声，噪声标准差为 noise_level
        noise = noise_level * randn(1, size(X, 2));
        X_new = X(i, :) + noise;
        Y_new = Y(i);  % 目标值保持不变
        % 将新样本追加到增强数据中
        X_aug = [X_aug; X_new];
        Y_aug = [Y_aug; Y_new];
    end
end

%% 查看增强后的数据尺寸
disp("增强后 X 的大小：");
disp(size(X_aug));  % 期望大小为 140*(1+num_aug) x 39，即 560 x 39
disp("增强后 Y 的大小：");
disp(size(Y_aug));  % 期望大小为 560 x 1

%% 保存增强后的数据（可选）
save("D:\红茶数据2024.0423\工业相机\自然\DATA\tp_aug.mat", "X_aug", "Y_aug");
