import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Networks(nn.Module):
    def __init__(self, latent_dim=128):
        super(Networks, self).__init__()

        ##这里的维度可能要改后面

        # --- View 1: HOG (Input: 1984) ---
        self.encoder1 = nn.Sequential(
            nn.Linear(1984, 512),   # 第一层降维到 512
            nn.ReLU(),
            nn.Linear(512, latent_dim),    # 第二层压缩到 128 (Z)
        )

        # --- View 2: LBP (Input: 928) ---
        self.encoder2 = nn.Sequential(
            nn.Linear(928, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        # --- View 3: GIST (Input: 512) ---
        self.encoder3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        ##自适应视图融合权重
        # 初始权重均为 1，通过网络反向传播自动学习哪个视图更重要
        self.view_weights = nn.Parameter(torch.ones(3))  
    
    def forward(self, input1, input2, input3):
        # 1. 独立提取各视图的隐特征 Z
        z1 = self.encoder1(input1)
        z2 = self.encoder2(input2)
        z3 = self.encoder3(input3)

        # 2. 计算归一化的视图权重 (Softmax 保证权重和为 1)
        # 对应公式里的 alpha_v
        weights = F.softmax(self.view_weights, dim=0)  # 归一化权重


        # 3. 融合得到全局共识特征 Z_global
        z_global = weights[0] * z1 + weights[1] * z2 + weights[2] * z3

        # 返回各个视图的特征 (用于多视图重构 Loss) 和 全局特征 (用于算图 C 和分类)
        return  z1, z2, z3, z_global



