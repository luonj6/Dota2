import torch
# from dataloader_uci import data_loader_train
from network import Networks
import metrics as metrics
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from se import SE_block
import torch.nn as nn
from munkres import Munkres
from torch.nn import functional as F
from dataset_caltech import Caltech20

import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 为了保证绝对一致，可能会牺牲一点点速度，但为了复现实验是值得的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'==> Random Seed Set to: {seed}')


final_accs, final_nmis, final_aris = [], [], []
final_fscores, final_precs, final_recalls = [], [], []
# 2. 定义你要跑的几个种子
seeds = [1] 
# seeds = [1, 42, 2024, 3407, 888]

# 3. 加这行循环，把你原来的所有代码缩进进去
for seed in seeds:
    print(f"\n{'='*20} Start Running with Seed: {seed} {'='*20}")
    
    # ===> 关键一步：在所有操作开始前设置种子 <===
    setup_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]='1' 



    dataset = Caltech20()
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=2386, shuffle=False)
    mat = sio.loadmat('Caltech101-20.mat')
    label_true = mat['Y'].flatten() # 或者是 mat['truth']，看具体文件名
    # label_true = label_true - 1 # 如果标签是 1-20，通常要转成 0-19
    label_true = label_true # 如果标签是 1-20，通常要转成 0-19


    # learning_rate = 0.001# acc=98.75

    # model = Networks()
    # model = model.to(device)

    #################Add

    # for data in data_loader_train:
    #     train_imga, train_imgb, train_imgc = data
    #     input1 = train_imga.float().to(device)
    #     input2 = train_imgb.float().to(device)
    #     input3 = train_imgc.float().to(device)
        
    #     # 跑通我们新写的前向传播
    #     z1, z2, z3, z_global = model(input1, input2, input3)
        
    #     print(f"Input 1 shape: {input1.shape}")
    #     print(f"Latent z1 shape: {z1.shape}")
    #     print(f"Fused z_global shape: {z_global.shape}")
    #     print(f"Learned view weights (alpha): {torch.nn.functional.softmax(model.view_weights, dim=0).detach().cpu().numpy()}")
        
    #     print("✅ Phase 1: Encoder & Fusion module runs perfectly!")
    #     break # 测一个 Batch 证明代码通了就行了
    
    # print("="*40 + "\n")
    
    # # 退出程序，后面的 AE KMeans 测试先注释掉，因为网络还没训练
    # import sys
    # sys.exit()

    # print("\n" + "="*40)
    # print("Testing Self-Expression Graph C & Loss...")
    # model.train()

    # for data in data_loader_train:
    #     train_imga, train_imgb, train_imgc = data
    #     input1 = train_imga.float().to(device)
    #     input2 = train_imgb.float().to(device)
    #     input3 = train_imgc.float().to(device)
        
    #     # 1. 前向传播拿到所有东西
    #     z1, z2, z3, z_global, C_graph, weights = model(input1, input2, input3)
        
    #     # 2. 计算多视图共识重构 Loss (Z_v = C * Z_v)
    #     # 用 torch.matmul(C_graph, z) 来用别人的特征重构自己
    #     recon_loss1 = weights[0] * torch.sum((z1 - torch.matmul(C_graph, z1)) ** 2)
    #     recon_loss2 = weights[1] * torch.sum((z2 - torch.matmul(C_graph, z2)) ** 2)
    #     recon_loss3 = weights[2] * torch.sum((z3 - torch.matmul(C_graph, z3)) ** 2)
        
    #     total_recon_loss = recon_loss1 + recon_loss2 + recon_loss3
        
    #     # 3. 计算图 C 的正则化项 (防止 C 里的权重变得无穷大)
    #     lambda_C = 1.0 # 这个超参数后续可以调
    #     reg_loss_C = lambda_C * torch.sum(C_graph ** 2)
        
    #     # 4. 阶段二总 Loss
    #     loss_stage2 = total_recon_loss + reg_loss_C
        
    #     print(f"Graph C shape: {C_graph.shape}")
    #     print(f"Diag of C (should be 0): {C_graph[0,0].item():.6f}, {C_graph[100,100].item():.6f}")
    #     print(f"Recon Loss: {total_recon_loss.item():.4f} | Reg Loss: {reg_loss_C.item():.4f}")
    #     print(f"Total Loss: {loss_stage2.item():.4f}")
        
    #     print("✅ Phase 2: Self-Expression Matrix C is fully integrated!")
    #     break

    # print("\n" + "="*40)
    # print("Testing Clustering Head, Top-K & Repulsive Loss...")
    # model.train() 

    # for data in data_loader_train:
    #     train_imga, train_imgb, train_imgc = data
    #     input1 = train_imga.float().to(device)
    #     input2 = train_imgb.float().to(device)
    #     input3 = train_imgc.float().to(device)
        
    #     # 1. 前向传播
    #     z1, z2, z3, z_global, C_graph, weights, H_IN = model(input1, input2, input3)
        
    #     # 2. 对称化图 C，并用 Top-K 提取正样本集合 P
    #     # 公式: A = (|C| + |C^T|) / 2
    #     C_abs = torch.abs(C_graph)
    #     A_sym = (C_abs + C_abs.t()) / 2.0
        
    #     K = 5 # 设置 Top-K 邻居数量
    #     topk_weights, topk_indices = torch.topk(A_sym, k=K, dim=1)
        
    #     # 3. 计算下方网络的语义排斥力 D_IN
    #     # 概率分布的内积可以衡量相似度: 极其相似则趋近 1，极不相似趋近 0
    #     sim_H = torch.matmul(H_IN, H_IN.t()) 
    #     D_IN = 1.0 - sim_H # 异类趋近 1，同类趋近 0
        
    #     # 4. 计算排斥性自表达的物理剪刀 Loss (C \odot D_IN)
    #     # 注意: 加 detach() 是为了防止算图 C 的时候，错误地更新下方分类器的参数
    #     beta = 10.0 # 斥力权重
    #     repulsive_loss = beta * torch.sum((C_graph * D_IN.detach()) ** 2)
        
    #     print(f"H_IN shape: {H_IN.shape} (Should be 2386, 20)")
    #     print(f"Top-{K} Indices shape: {topk_indices.shape} (Should be 2386, 5)")
    #     print(f"D_IN sample value: {D_IN[0,1].item():.4f}")
    #     print(f"Repulsive Loss: {repulsive_loss.item():.4f}")
        
    #     print("✅ Phase 3 & 4: Clustering head, Top-K P and D_IN logic verified!")
    #     break

    # print("\n" + "="*40)
    # print("Testing FINAL BOSS: Multi-Positive InfoNCE Loss...")
    # model.train() 

    # for data in data_loader_train:
    #     train_imga, train_imgb, train_imgc = data
    #     input1 = train_imga.float().to(device)
    #     input2 = train_imgb.float().to(device)
    #     input3 = train_imgc.float().to(device)
        
    #     # 1. 前向传播
    #     z1, z2, z3, z_global, C_graph, weights, H_IN = model(input1, input2, input3)
        
    #     # 2. 对称化图 C，并用 Top-K 提取正样本集合 P
    #     C_abs = torch.abs(C_graph)
    #     A_sym = (C_abs + C_abs.t()) / 2.0
    #     K = 5 
    #     _, topk_indices = torch.topk(A_sym, k=K, dim=1)
        
    #     # 3. 计算排斥力 D_IN (用于优化上方图 C)
    #     sim_H = torch.matmul(H_IN, H_IN.t()) 
    #     D_IN = 1.0 - sim_H 
        
    #     # =======================================================
    #     # 4. [新增] 计算多正样本 InfoNCE Loss (用于优化下方网络)
    #     # =======================================================
    #     tau = 0.5 # 温度系数，通常设为 0.1 到 0.5 之间
    #     N = z_global.shape[0]
        
    #     # 4.1 特征 L2 归一化 (对比学习的标配，让内积等于余弦相似度)
    #     z_norm = F.normalize(z_global, dim=1)
        
    #     # 4.2 计算全场两两相似度矩阵，并除以温度系数
    #     sim_matrix = torch.matmul(z_norm, z_norm.t()) / tau
        
    #     # 4.3 极其关键的防崩溃设计：屏蔽自己与自己的对比！
    #     # 把对角线（自己和自己）的相似度设为极小值，这样在 Softmax 时概率就变成 0
    #     sim_matrix.fill_diagonal_(-1e9)
        
    #     # 4.4 巧妙利用 log_softmax 算出每个样本在全局中的相对 log 概率
    #     # 这完美等价于 InfoNCE 公式里的：分子 / (全场分母求和) 再取 Log！
    #     log_prob = F.log_softmax(sim_matrix, dim=1)
        
    #     # 4.5 构建正样本 Mask (从 topk_indices 生成)
    #     # 生成一个 N x N 的全 0 矩阵，然后用 scatter_ 把对应兄弟的坐标填成 1
    #     mask = torch.zeros((N, N)).to(device)
    #     mask.scatter_(1, topk_indices, 1.0)
        
    #     # 4.6 算出 InfoNCE Loss
    #     # 用 mask 挑出正样本的 log_prob 求和，除以正样本数量 K，最后取负数求平均
    #     loss_contrastive = - (mask * log_prob).sum(dim=1) / K
    #     loss_contrastive = loss_contrastive.mean()
        
    #     print(f"sim_matrix shape: {sim_matrix.shape}")
    #     print(f"Mask shape: {mask.shape}, sum per row (should be {K}): {mask[0].sum().item()}")
    #     print(f"InfoNCE Contrastive Loss: {loss_contrastive.item():.4f}")
        
    #     print("✅ Phase 5: Multi-Positive InfoNCE Masterpiece complete!")
    #     break

    model = Networks(latent_dim=128, num_samples=2386, num_clusters=20)
    model = model.to(device)

    # ========================================================
    # 阶段零：Autoencoder 预训练 (打地基)
    # ========================================================
    print("\n" + "="*40)
    print("🔥 Stage 0: Starting AE Pretraining...")
    
    # 巧妙过滤：只优化 Encoder 和 Decoder，绝对不要动图 C 和分类头
    ae_params = [p for n, p in model.named_parameters() if 'encoder' in n or 'decoder' in n]
    optimizer_AE = torch.optim.Adam(ae_params, lr=1e-3)
    criterion_mse = torch.nn.MSELoss()

    n_epochs_pretrain = 100 # 预训练打底 100 轮
    
    for epoch in range(n_epochs_pretrain):
        model.train()
        for data in data_loader_train:
            train_imga, train_imgb, train_imgc = data
            input1 = train_imga.float().to(device)
            input2 = train_imgb.float().to(device)
            input3 = train_imgc.float().to(device)

            # 调用专属的 forward_ae
            rec1, rec2, rec3 = model.forward_ae(input1, input2, input3)
            
            # 计算纯粹的 MSE 重构损失
            loss_ae = criterion_mse(rec1, input1) + criterion_mse(rec2, input2) + criterion_mse(rec3, input3)
            
            optimizer_AE.zero_grad()
            loss_ae.backward()
            optimizer_AE.step()
            
        if (epoch + 1) % 20 == 0:
            print(f"Pretrain Epoch [{epoch+1}/{n_epochs_pretrain}] | MSE Loss: {loss_ae.item():.4f}")

    print("✅ AE Pretraining Completed! Latent space initialized.")




    # ==========================================
    # 核心创新：交替优化的双优化器配置
    # ==========================================
    # 1. 专门用于优化图 C 的优化器
    optimizer_C = torch.optim.Adam([model.C], lr=1e-3, weight_decay=1e-4)

    # 2. 专门用于优化网络权重（Encoder, Fusion, 聚类头）的优化器
    # 过滤掉 C，只把其他参数放进来
    net_params = [p for n, p in model.named_parameters() if n != 'C']
    optimizer_Net = torch.optim.Adam(net_params, lr=1e-3, weight_decay=1e-4)

    # 超参数设定 (你可以后期微调)
    lambda_C = 1.0   # 图 C 正则化权重
    beta = 10.0      # 排斥自表达权重 (物理剪刀)
    gamma = 1.0      # InfoNCE 权重 (对比拉力)
    tau = 0.5        # InfoNCE 温度系数
    K = 5            # Top-K 正样本数量

    n_epochs = 100
    print("\n" + "="*40)
    print("🚀 Starting Alternating Optimization Training...")

    for epoch in range(n_epochs):
        model.train()
        for data in data_loader_train:
            train_imga, train_imgb, train_imgc = data
            input1 = train_imga.float().to(device)
            input2 = train_imgb.float().to(device)
            input3 = train_imgc.float().to(device)
            
            N = input1.shape[0]

            # ====================================================
            # Phase 1: 冻结网络，更新图 C (拓扑发现与剪枝)
            # ====================================================
            optimizer_C.zero_grad()
            z1, z2, z3, z_global, C_graph, weights, H_IN = model(input1, input2, input3)
            
            # 1.1 共识重构 Loss (注意：这里用 z.detach() 防止梯度传给网络)
            recon_loss = weights[0] * torch.sum((z1.detach() - torch.matmul(C_graph, z1.detach())) ** 2) + \
                         weights[1] * torch.sum((z2.detach() - torch.matmul(C_graph, z2.detach())) ** 2) + \
                         weights[2] * torch.sum((z3.detach() - torch.matmul(C_graph, z3.detach())) ** 2)
            
            # 1.2 排斥自表达 Loss (物理剪断)
            sim_H = torch.matmul(H_IN.detach(), H_IN.detach().t()) 
            D_IN = 1.0 - sim_H 
            repulsive_loss = beta * torch.sum((C_graph * D_IN) ** 2)
            
            # 1.3 基础正则化
            reg_loss = lambda_C * torch.sum(C_graph ** 2)
            
            # 1.4 图 C 总 Loss 并反向传播
            loss_C = recon_loss + repulsive_loss + reg_loss
            loss_C.backward()
            optimizer_C.step()

            # ====================================================
            # Phase 2: 冻结图 C，更新网络 (多视图提取与语义对齐)
            # ====================================================
            optimizer_Net.zero_grad()
            # 重新 Forward，获取更新了 C 之后的计算图
            z1, z2, z3, z_global, C_graph, weights, H_IN = model(input1, input2, input3)
            
            # 2.1 提取确信正样本 P (来自冻结的图 C)
            C_abs = torch.abs(C_graph.detach())
            A_sym = (C_abs + C_abs.t()) / 2.0
            _, topk_indices = torch.topk(A_sym, k=K, dim=1)
            
            # 2.2 共识重构 Loss (拉扯 Encoder)
            recon_loss_net = weights[0] * torch.sum((z1 - torch.matmul(C_graph.detach(), z1)) ** 2) + \
                             weights[1] * torch.sum((z2 - torch.matmul(C_graph.detach(), z2)) ** 2) + \
                             weights[2] * torch.sum((z3 - torch.matmul(C_graph.detach(), z3)) ** 2)
            
            # 2.3 InfoNCE Loss (拉拢兄弟，推开路人)
            z_norm = F.normalize(z_global, dim=1)
            sim_matrix = torch.matmul(z_norm, z_norm.t()) / tau
            sim_matrix.fill_diagonal_(-1e9)
            log_prob = F.log_softmax(sim_matrix, dim=1)
            
            mask = torch.zeros((N, N)).to(device)
            mask.scatter_(1, topk_indices, 1.0)
            loss_contrastive = - (mask * log_prob).sum(dim=1) / K
            loss_contrastive = loss_contrastive.mean()
            
            # 2.4 网络总 Loss 并反向传播
            loss_Net = recon_loss_net + gamma * loss_contrastive
            loss_Net.backward()
            optimizer_Net.step()

        # 打印 Epoch 日志
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | Loss C: {loss_C.item():.4f} | Loss Net: {loss_Net.item():.4f} | InfoNCE: {loss_contrastive.item():.4f}")

    print("✅ Training Completed!")

    # ========================================================
    # 终极阶段：聚类性能双轨评估 (Evaluation)
    # ========================================================
    print("\n" + "="*40)
    print("📊 Evaluating Final Clustering Performance...")
    model.eval() 
    
    with torch.no_grad():
        for data in data_loader_train:
            train_imga, train_imgb, train_imgc = data
            input1 = train_imga.float().to(device)
            input2 = train_imgb.float().to(device)
            input3 = train_imgc.float().to(device)
            
            # 拿到训练完成后的终极输出
            z1, z2, z3, z_global, C_graph, weights, H_IN = model(input1, input2, input3)
            
            # ------------------------------------------------
            # 评估方法 1：直接使用底层聚类头 H_IN (End-to-End)
            # ------------------------------------------------
            # 这是最纯粹的深度聚类输出，不需要借助任何外部机器学习算法
            preds_network = torch.argmax(H_IN, dim=1).cpu().numpy()
            
            acc_net = metrics.acc(label_true, preds_network)
            nmi_net = metrics.nmi(label_true, preds_network)
            ari_net = metrics.ari(label_true, preds_network)
            
            print(f"🌟 Method 1: Network Direct Prediction (H_IN)")
            print(f"   ACC: {acc_net:.4f} | NMI: {nmi_net:.4f} | ARI: {ari_net:.4f}")
            
            # ------------------------------------------------
            # 评估方法 2：在全局共识特征上跑 K-Means (Latent Space)
            # ------------------------------------------------
            # 证明 InfoNCE 和图 C 把隐空间 Z_global 塑造得非常线性可分
            z_global_np = z_global.cpu().numpy()
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=20, n_init=20, random_state=seed)
            preds_kmeans = kmeans.fit_predict(z_global_np)
            
            acc_km = metrics.acc(label_true, preds_kmeans)
            nmi_km = metrics.nmi(label_true, preds_kmeans)
            ari_km = metrics.ari(label_true, preds_kmeans)
            
            print(f"\n🌟 Method 2: K-Means on Fused Z_global")
            print(f"   ACC: {acc_km:.4f} | NMI: {nmi_km:.4f} | ARI: {ari_km:.4f}")
            
            # 打印最终学到的视图权重
            final_weights = torch.nn.functional.softmax(model.view_weights, dim=0).cpu().numpy()
            print(f"\nFinal Learned View Weights: HOG={final_weights[0]:.4f}, LBP={final_weights[1]:.4f}, GIST={final_weights[2]:.4f}")
            
            break 
    print("="*40 + "\n")