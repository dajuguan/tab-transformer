# radical loss & stacking line data
import numpy as np
import torch
import sys
sys.path.append("..")
from mfnn.xydata import XYDataSet, XYZDataSet

def loadData(shuffle=False):
    x_low = np.loadtxt("./data/x_features.txt")
    y_low = np.loadtxt("./data/loss_2d.txt")
    y_high = np.loadtxt("./data/loss_3d.txt")
    # y_high_encoded = np.loadtxt("./data/loss_3d_encoded.txt")
    r_2d = np.loadtxt("./data/r_2d.txt")
    r_3d = np.loadtxt("./data/r_3d.txt")

    x_low = torch.from_numpy(x_low).to(torch.double)
    x_low = (x_low - x_low.min() ) / ( x_low.max() - x_low.min())
    y_low = torch.from_numpy(y_low).to(torch.double) * 100.0
    y_high = torch.from_numpy(y_high).to(torch.double) * 100
    # y_high_encoded = torch.from_numpy(y_high_encoded).to(torch.double)

    # random sample test cases
    torch.manual_seed(123)
    len_x = len(x_low)
    test_indices = torch.randperm(len_x)[:int(len_x/4)]
    xt_low, yt_low, yt_high = x_low[test_indices],y_low[test_indices],y_high[test_indices]

    # train cases
    train_len = 135
    x_low, y_low, y_high = x_low[:train_len], y_low[:train_len], y_high[:train_len]

    loader_high = torch.utils.data.DataLoader(XYZDataSet(x_low, y_low, y_high), batch_size=len(x_low), shuffle=shuffle)
    return x_low,y_low, y_high, loader_high, r_2d, r_3d, xt_low, yt_low, yt_high

def loadStackingData():
    # 1. mockdata
    def D1():
        data = np.loadtxt(r"/home/po/phd/avalanche/3d/data_3d.csv", delimiter=",", dtype=np.float32)
        # data[:,1] = data[:,1]/100
        data[:,2] = data[:,2]/40
        data[:,3] = data[:,3]/5
        data[:,4] = data[:,4]/5
        X = torch.from_numpy(data[:, :3])
        Y = torch.from_numpy(data[:, 3:])
        dim_rope_seq = 0
        return X, Y, dim_rope_seq

    # 2. 二维特征+三维径向损失=> 三维最佳弯掠
    def D2():
        data = np.loadtxt(r"/home/po/phd/avalanche/data.csv", delimiter=",",skiprows=1, dtype=np.float32)
        indexs = data[:,0] - 1
        x_low = np.loadtxt("./data/x_features.txt", dtype=np.float32)
        mask = np.in1d(np.arange(0, len(x_low)),indexs)  # 有些三维数据丢失了，需要过滤下
        radical_loss_3d = np.loadtxt("./data/loss_3d.txt", dtype=np.float32)
       
        X = torch.from_numpy(np.concatenate((radical_loss_3d, x_low), axis=1))
        X = X[mask]
        Y = torch.from_numpy(data[:,13:17]) / 5
        dim_rope_seq = radical_loss_3d.shape[-1]
        return X, Y, dim_rope_seq


    # 3. 二维特征+二维径向损失=> 三维最佳弯掠
    def D3():
        data = np.loadtxt(r"/home/po/phd/avalanche/data.csv", delimiter=",",skiprows=1, dtype=np.float32)
        indexs = data[:,0] - 1
        x_low = np.loadtxt("./data/x_features.txt", dtype=np.float32)
        mask = np.in1d(np.arange(0, len(x_low)),indexs)  # 有些三维数据丢失了，需要过滤下
        radical_loss_2d = np.loadtxt("./data/loss_2d.txt", dtype=np.float32)
       
        X = torch.from_numpy(np.concatenate((radical_loss_2d, x_low), axis=1))
        X = X[mask]
        Y = torch.from_numpy(data[:,13:17]) / 5
        dim_rope_seq = radical_loss_2d.shape[-1]
        return X, Y, dim_rope_seq

    X, Y, dim_rope_seq = D3()
    print("dim_rope_seq=======>:", dim_rope_seq)
    dataloader = torch.utils.data.DataLoader(XYDataSet(X, Y), batch_size=len(X))
    test_size = 10
    xt, yt = X[:test_size], Y[:test_size]
    return X, Y, dataloader, dim_rope_seq, xt, yt