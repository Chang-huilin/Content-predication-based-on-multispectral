import torch
import torch.nn as nn
from .attention import AttentionModule

class MAD_MIL(nn.Module):
    def __init__(self, n_classes=2, L=512, D=128, n_heads=3):
        super(MAD_MIL, self).__init__()
        self.L = L
        self.D = D
        self.n_heads = n_heads
        self.n_classes = n_classes
        
        # 多个注意力头
        self.attention_heads = nn.ModuleList([
            AttentionModule(L=L, D=D) for _ in range(n_heads)
        ])
        
        # 特征转换层
        self.feature_extractor = nn.Sequential(
            nn.Linear(L, L),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # 分类器
        self.classifier = nn.Linear(L * n_heads, n_classes)
        
    def forward(self, x):
        # x: [B, N, L]
        x = self.feature_extractor(x)
        
        # 收集每个头的输出
        head_outputs = []
        attention_maps = []
        
        for attention_head in self.attention_heads:
            M, A = attention_head(x)  # M: [B, 1, L], A: [B, 1, N]
            head_outputs.append(M)
            attention_maps.append(A)
        
        # 合并所有头的输出
        M = torch.cat(head_outputs, dim=1)  # [B, n_heads, L]
        M = M.view(M.size(0), -1)  # [B, n_heads*L]
        
        # 分类预测
        logits = self.classifier(M)
        Y_prob = torch.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)
        
        return logits, Y_prob, Y_hat, attention_maps 