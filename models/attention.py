import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(AttentionModule, self).__init__()
        self.L = L  # 输入特征维度
        self.D = D  # 注意力隐层维度
        self.K = K  # 注意力头数
        
        # 注意力网络
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, K)
        )
        
    def forward(self, x):
        # x: [B, N, L], B是批次大小, N是实例数量, L是特征维度
        A = self.attention(x)  # [B, N, K]
        A = torch.transpose(A, 1, 2)  # [B, K, N]
        A = F.softmax(A, dim=2)  # 在实例维度上做softmax
        
        M = torch.bmm(A, x)  # [B, K, L]
        
        return M, A 