import torch.utils.data as data
import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Caltech20(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        # 加载数据
        # 假设文件名为 Caltech101-20.mat
        try:
            data_0 = scipy.io.loadmat('Caltech101-20.mat')
        except FileNotFoundError:
            print("Error: Caltech101-20.mat not found!")
            return

        # 获取数据 (通常在 'X' 这个 key 下，是一个 cell array)
        # X[0][0]...X[0][5] 分别对应 6 个视图
        X = data_0['X'] 
        
        # 获取标签 (通常在 'Y' 这个 key 下)
        # 这一步是为了训练时如果有需要可以验证，或者存下来供 best_map 使用
        if 'Y' in data_0:
            self.labels = data_0['Y']
            if self.labels.shape[0] == 1: self.labels = self.labels.T # 转置成 [N, 1]
            self.labels = self.labels.flatten()
            self.train_num = len(self.labels)
        else:
            self.train_num = 2386 # Caltech101-20 标准数量

        # --- 提取特定的三个视图 ---
        # 根据标准 Caltech101-20 数据集结构：
        # View 3: HOG  (1984 维)
        # View 4: GIST (512 维)
        # View 5: LBP  (928 维)
        
        # 注意：MATLAB索引从1开始，Python从0开始
        # 我们需要根据维度确认一下，以防不同版本的 mat 文件顺序不同
        
        views = []
        for i in range(X.shape[1]):
            view_data = X[0][i]
            dim = view_data.shape[1]
            views.append(view_data)
            # 打印一下方便确认
            # print(f"Raw View {i} shape: {view_data.shape}")

        # 手动绑定视图 (按 HOG, LBP, GIST 的顺序)
        # 务必检查你的 mat 文件里维度是否对应！
        # 这里假设: HOG=1984, LBP=928, GIST=512
        raw_data1 = None # HOG
        raw_data2 = None # LBP
        raw_data3 = None # GIST
        
        for v in views:
            d = v.shape[1]
            if d == 1984: raw_data1 = v # HOG
            elif d == 928:  raw_data2 = v # LBP
            elif d == 512:  raw_data3 = v # GIST

        if raw_data1 is None or raw_data2 is None or raw_data3 is None:
            raise ValueError("找不到对应的维度 (1984, 928, 512)，请检查 .mat 文件内容")

        # # --- 数据预处理 (StandardScaler) ---
        # # 对于高维特征，StandardScaler (均值0，方差1) 通常比 MinMaxScaler 更好
        # # 因为它可以消除异常值的影响，并且让 ReLU 更容易激活
        
        # View 1: HOG (1984)
        scaler1 = StandardScaler()
        self.data1 = scaler1.fit_transform(raw_data1.astype(np.float32))

        # View 2: LBP (928)
        scaler2 = StandardScaler()
        self.data2 = scaler2.fit_transform(raw_data2.astype(np.float32))

        # View 3: GIST (512)
        scaler3 = StandardScaler()
        self.data3 = scaler3.fit_transform(raw_data3.astype(np.float32))

        # self.data1 = raw_data1.astype(np.float32)
        # self.data2 = raw_data2.astype(np.float32)
        # self.data3 = raw_data3.astype(np.float32)



        print("\nData Loading & Preprocessing Done (Caltech101-20):")
        print(f"Total Samples: {self.train_num}")
        print(f"View 1 (HOG)  Shape: {self.data1.shape}, Range: [{self.data1.min():.2f}, {self.data1.max():.2f}]")
        print(f"View 2 (LBP)  Shape: {self.data2.shape}, Range: [{self.data2.min():.2f}, {self.data2.max():.2f}]")
        print(f"View 3 (GIST) Shape: {self.data3.shape}, Range: [{self.data3.min():.2f}, {self.data3.max():.2f}]")
        print("-" * 50)

    def __getitem__(self, index):
        img_train1 = self.data1[index, :]
        img_train2 = self.data2[index, :]
        img_train3 = self.data3[index, :]
        return img_train1, img_train2, img_train3

    def __len__(self):
        return self.train_num