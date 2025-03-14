% 指定基础目录
baseDir = 'D:\茶叶干燥过程\茶叶多光谱图像\热风第二批140个样+水分\1鲜叶_processed\';

% 指定输出目录
outputDir = 'D:\茶叶干燥过程\茶叶多光谱图像\热风第二批140个样+水分\GLHD\';

% 如果输出目录不存在，则创建
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 循环遍历文件夹1到20
for folderIndex = 1:20
    % 生成文件夹名
    folderName = num2str(folderIndex);

    % 初始化WL9矩阵
    WL9 = zeros(25, 6);

    % 循环遍历'interestingspace'文件夹中的图像
    for imageIndex = 1:25
        % 生成文件名
        filename = fullfile(baseDir, folderName, '25个波段对应的图像\interestingspace', [num2str(imageIndex), '.bmp']);

        % 使用提供的代码读取和处理图像
        A = imread(filename); % 读取图像
        A = im2gray(A); % 转换为灰度图像
        p = imhist(A); % 计算直方图
        p = p ./ numel(A); % 归一化直方图
        L = length(p); % 灰度级数
        [v, mu] = statmoments(p, 3); % 计算统计矩
        t(1) = mu(1); % 均值
        t(2) = mu(2) ^ 0.5; % 标准差
        varn = mu(2) / (L - 1) ^ 2; % 归一化方差
        t(3) = 1 - 1 / (1 + varn); % 纹理度量
        t(4) = mu(3) / (L - 1) ^ 2; % 三阶中心矩
        t(5) = sum(p.^2); % 能量
        t(6) = -sum(p .* (log2(p + eps))); % 熵
        T = [t(1) t(2) t(3) t(4) t(5) t(6)]; % 特征向量
        WL9(imageIndex, :) = T; % 存储特征
    end

    % 将结果保存到Excel文件
    outputFilename = fullfile(outputDir, sprintf('%03d.xlsx', folderIndex+0));
    writematrix(WL9, outputFilename);
end