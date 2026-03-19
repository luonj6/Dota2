import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Networks(nn.Module):
    def __init__(self, latent_dim=128, num_samples=2386, num_clusters=20):
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

        # === 新增：Decoders (用于预训练重构) ===
        self.decoder1 = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, 1984))
        self.decoder2 = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, 928))
        self.decoder3 = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, 512))



        ##自适应视图融合权重
        # 初始权重均为 1，通过网络反向传播自动学习哪个视图更重要
        self.view_weights = nn.Parameter(torch.ones(3))  

        ##全局自表达矩阵C
        #必须极其微小地初始化，否则一开始容易梯度爆炸
        self.C = nn.Parameter(1.0e-4 * torch.randn(num_samples, num_samples))
    
        #极简聚类分类头f_cn
        self.cluster_head = nn.Sequential(
            nn.Linear(latent_dim, num_clusters),
            nn.Softmax(dim=1) # 输出概率分布
        )


    # === 新增：专门用于 AE 预训练的前向传播 ===
    def forward_ae(self, input1, input2, input3):
        z1 = self.encoder1(input1)
        z2 = self.encoder2(input2)
        z3 = self.encoder3(input3)

        rec1 = self.decoder1(z1)
        rec2 = self.decoder2(z2)
        rec3 = self.decoder3(z3)

        return rec1, rec2, rec3


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

        # 强行把 C 的对角线变成 0（自己不能用自己来重构，必须逼迫它去寻找真正的邻居）
        C_diag_zero = self.C - torch.diag(torch.diag(self.C))

        # 得到下方的语义预测概率 H_IN
        H_IN = self.cluster_head(z_global)

        # 返回各个视图的特征 (用于多视图重构 Loss) 和 全局特征 (用于算图 C 和分类)
        return  z1, z2, z3, z_global, C_diag_zero, weights, H_IN
