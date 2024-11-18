import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 导入数据
file_path = r"C:\Users\79365\Desktop\图像-叶绿素\叶绿素\matlab数据\35.mat"
data = sio.loadmat(file_path)

# 提取X和Y
X = data['X']
Y = data['Y'][:, 2]  # 第三列的Y数据

# 划分训练集和测试集
num_total = 140
X_train = np.concatenate([X[0:num_total:5, :], X[2:num_total:5, :], X[4:num_total:5, :]], axis=0)
X_test = np.concatenate([X[1:num_total:5, :], X[3:num_total:5, :]], axis=0)

Y_train = np.concatenate([Y[0:num_total:5], Y[2:num_total:5], Y[4:num_total:5]], axis=0)
Y_test = np.concatenate([Y[1:num_total:5], Y[3:num_total:5]], axis=0)

# 数据归一化
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_test = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).unsqueeze(1)  # 增加维度适应Conv2d
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# 构建卷积神经网络
# Update the model initialization
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((1, 1))  # Change to (1, 1) or adjust according to input size

        self.conv2 = nn.Conv2d(16, 32, (3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(32)
        
        self.fc = nn.Linear(32 * X_train.shape[1], 1)  # Adjust if necessary

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# 初始化模型
model = ConvNet()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs.flatten(), Y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型预测
model.eval()
with torch.no_grad():
    Y_train_pred = model(X_train_tensor).flatten()
    Y_test_pred = model(X_test_tensor).flatten()

# 数据反归一化
Y_train_pred = scaler_Y.inverse_transform(Y_train_pred.numpy().reshape(-1, 1)).flatten()
Y_test_pred = scaler_Y.inverse_transform(Y_test_pred.numpy().reshape(-1, 1)).flatten()
Y_train_actual = scaler_Y.inverse_transform(Y_train_tensor.numpy().reshape(-1, 1)).flatten()
Y_test_actual = scaler_Y.inverse_transform(Y_test_tensor.numpy().reshape(-1, 1)).flatten()

# 均方根误差 (RMSE)
train_rmse = np.sqrt(mean_squared_error(Y_train_actual, Y_train_pred))
test_rmse = np.sqrt(mean_squared_error(Y_test_actual, Y_test_pred))

print(f'训练集RMSE: {train_rmse:.4f}')
print(f'测试集RMSE: {test_rmse:.4f}')

# 绘制预测结果
plt.figure()
plt.plot(Y_train_actual, 'r-*', label='真实值')
plt.plot(Y_train_pred, 'b-o', label='预测值')
plt.title(f'训练集预测结果对比 (RMSE = {train_rmse:.4f})')
plt.legend()
plt.grid()

plt.figure()
plt.plot(Y_test_actual, 'r-*', label='真实值')
plt.plot(Y_test_pred, 'b-o', label='预测值')
plt.title(f'测试集预测结果对比 (RMSE = {test_rmse:.4f})')
plt.legend()
plt.grid()

plt.show()

# R²系数计算
train_r2 = 1 - np.sum((Y_train_actual - Y_train_pred) ** 2) / np.sum((Y_train_actual - np.mean(Y_train_actual)) ** 2)
test_r2 = 1 - np.sum((Y_test_actual - Y_test_pred) ** 2) / np.sum((Y_test_actual - np.mean(Y_test_actual)) ** 2)

print(f'训练集R²: {train_r2:.4f}')
print(f'测试集R²: {test_r2:.4f}')

# MAE和MBE计算
train_mae = mean_absolute_error(Y_train_actual, Y_train_pred)
test_mae = mean_absolute_error(Y_test_actual, Y_test_pred)

train_mbe = np.mean(Y_train_pred - Y_train_actual)
test_mbe = np.mean(Y_test_pred - Y_test_actual)

print(f'训练集MAE: {train_mae:.4f}')
print(f'测试集MAE: {test_mae:.4f}')
print(f'训练集MBE: {train_mbe:.4f}')
print(f'测试集MBE: {test_mbe:.4f}')
