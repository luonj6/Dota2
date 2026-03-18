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


    learning_rate = 0.001# acc=98.75

    model = Networks()
    model = model.to(device)

    #################Add

    for data in data_loader_train:
        train_imga, train_imgb, train_imgc = data
        input1 = train_imga.float().to(device)
        input2 = train_imgb.float().to(device)
        input3 = train_imgc.float().to(device)
        
        # 跑通我们新写的前向传播
        z1, z2, z3, z_global = model(input1, input2, input3)
        
        print(f"Input 1 shape: {input1.shape}")
        print(f"Latent z1 shape: {z1.shape}")
        print(f"Fused z_global shape: {z_global.shape}")
        print(f"Learned view weights (alpha): {torch.nn.functional.softmax(model.view_weights, dim=0).detach().cpu().numpy()}")
        
        print("✅ Phase 1: Encoder & Fusion module runs perfectly!")
        break # 测一个 Batch 证明代码通了就行了
    
    print("="*40 + "\n")
    
    # 退出程序，后面的 AE KMeans 测试先注释掉，因为网络还没训练
    import sys
    sys.exit()