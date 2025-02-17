import torch
from torch.utils.data import DataLoader
from models.mad_mil import MAD_MIL

def train_loop(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        
        optimizer.zero_grad()
        logits, Y_prob, Y_hat, _ = model(data)
        
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def main():
    # 超参数设置
    n_classes = 2
    L = 512  # 特征维度
    D = 128  # 注意力隐层维度
    n_heads = 3  # 注意力头数量
    lr = 1e-4
    weight_decay = 1e-6
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MAD_MIL(n_classes=n_classes, L=L, D=D, n_heads=n_heads).to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch}, Loss: {train_loss:.4f}') 