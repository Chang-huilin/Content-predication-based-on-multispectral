%% 


figure; % 创建新的图形窗口%%  绘制散点图
sz = 25;
c = 'r';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('训练集真实值');
ylabel('训练集预测值');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('训练集预测值 vs. 训练集真实值')
% 

figure
scatter(T_test, T_sim2, sz, 'filled', c, 'Marker', '^')  % 使用红色实心三角作为点的标记
hold on
plot(xlim, ylim, '--k')
xlabel('测试集真实值');
ylabel('测试集预测值');
legend('R_c=0.9732 RMSEP=0.3101','R_p=0.9396 RMSEP=0.4509','Location', 'Northwest','FontWeight', 'bold');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('测试集预测值 vs. 测试集真实值')
% 

plot(T_train, T_sim1, 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'k'); % 黑色实心方块表示校正集
hold on; 
%绘制预测集的散点图（用红色实心三角形表示）
plot(T_test, T_sim2, '^r', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % 红色实心三角形表示预测集
hold on;
hold on;
x=0:0.05:0.2;
y=x;
plot(x,y);
hold on;
line([0, 90], [0, 90], 'Color', 'red', 'LineStyle', '--');%对角线，从（0，0)到（90，90）
xlabel('Measured value (%)','FontWeight', 'bold');        %加粗，'FontWeight', 'bold'
ylabel('Predicted value (%)','FontWeight', 'bold');
legend('R_c=0.9732 RMSEP=0.3101','R_p=0.9396 RMSEP=0.4509','Location', 'Northwest','FontWeight', 'bold');