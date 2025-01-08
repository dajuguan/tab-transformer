# radical loss & stacking line data
import numpy as np
import torch
import sys
sys.path.append("..")
from mfnn.xydata import XYDataSet, XYZDataSet

def loadData(shuffle=False, indices_to_remove=None, resampleIndex=None,scale=100.0, start=0):
    x_low = np.loadtxt("./data/x_features.txt")
    y_low = np.loadtxt("./data/loss_2d.txt")
    y_high = np.loadtxt("./data/loss_3d.txt")
    # y_high_encoded = np.loadtxt("./data/loss_3d_encoded.txt")
    r_2d = np.loadtxt("./data/r_2d.txt")
    r_3d = np.loadtxt("./data/r_3d.txt")

    x_low = torch.from_numpy(x_low).to(torch.double)
    x_low = (x_low - x_low.min() ) / ( x_low.max() - x_low.min())
    y_low = torch.from_numpy(y_low).to(torch.double) * scale
    y_high = torch.from_numpy(y_high).to(torch.double) * scale
    # y_high_encoded = torch.from_numpy(y_high_encoded).to(torch.double)
    if indices_to_remove != None:
        mask = torch.ones(len(x_low), dtype=torch.bool)
        mask[indices_to_remove] = False
        x_low = x_low[mask]
        y_low = y_low[mask]
        y_high = y_high[mask]
    if resampleIndex != None:
        x_low = torch.cat((x_low[:resampleIndex],x_low[resampleIndex: resampleIndex+ 1],x_low[resampleIndex: resampleIndex+ 1], x_low[resampleIndex:]))
        y_low = torch.cat((y_low[:resampleIndex],y_low[resampleIndex: resampleIndex+ 1],y_low[resampleIndex: resampleIndex+ 1], y_low[resampleIndex:]))
        y_high = torch.cat((y_high[:resampleIndex],y_high[resampleIndex: resampleIndex+ 1],y_high[resampleIndex: resampleIndex+ 1], y_high[resampleIndex:]))
    # random sample test cases
    torch.manual_seed(123)
    len_x = len(x_low)
    test_indices = torch.randperm(len_x)[:int(len_x/4)]
    xt_low, yt_low, yt_high = x_low[test_indices],y_low[test_indices],y_high[test_indices]

    # train cases
    train_len = len(x_low) # 超过实际的134个数据
    if start != 0:
        train_len = 60
    x_low, y_low, y_high = x_low[start:train_len], y_low[start:train_len], y_high[start:train_len]

    batch_size = len(x_low)
    # batch_size = 32 # 实测减小之后效果更不好
    loader_high = torch.utils.data.DataLoader(XYZDataSet(x_low, y_low, y_high), batch_size=batch_size, shuffle=shuffle)
    return x_low,y_low, y_high, loader_high, r_2d, r_3d, xt_low, yt_low, yt_high

def loadStackingData(datatype="d3", indices_to_remove=None, max_len=None, shuffle=False, resampleIndex=None, startIndex=None):
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


    # 3. 二维特征 + 二维径向损失=> 三维最佳弯掠(弯高统一的，只有叶根的弯角、掠角，叶尖弯角、掠角和弯高，这5个参数; 弯高当时也去了，只有4个)
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

    # 4. 二维特征 + 二维径向损失=> 三维最佳弯掠(包含叶根弯角、掠角、弯高、掠高，和叶尖的共8个参数)
    def D4():
        data = np.loadtxt(r"./data/data_new.csv", delimiter=",",skiprows=1, dtype=np.float32)
        indexs = data[:,0] - 1
        x_low = np.loadtxt("./data/x_features.txt", dtype=np.float32)
        mask = np.in1d(np.arange(0, len(x_low)),indexs)  # 有些三维数据丢失了，需要过滤下
        radical_loss_2d = np.loadtxt("./data/loss_2d.txt", dtype=np.float32)
       
        X = torch.from_numpy(np.concatenate((radical_loss_2d, x_low), axis=1))
        X = X[mask]
        Y = torch.from_numpy(data[:,13:21])
        Y[:, 0:4] = Y[:, 0:4] /10  #角度缩放10，与弯高量纲一致
        dim_rope_seq = radical_loss_2d.shape[-1]
        return X, Y, dim_rope_seq

    if datatype == "d1":
        X, Y, dim_rope_seq = D1()
    elif datatype == "d2":
        X, Y, dim_rope_seq = D2()
    elif datatype == "d3":
        X, Y, dim_rope_seq = D3()
    elif datatype == "d4":
        X, Y, dim_rope_seq = D4()
    else:
        raise ValueError("datatype must be 'd1', 'd2', 'd3', or 'd4'")

    if resampleIndex != None:
        X = torch.cat((X,X[resampleIndex: resampleIndex+ 1]))
        Y = torch.cat((Y,Y[resampleIndex: resampleIndex+ 1]))

    if indices_to_remove != None:
        print("removing indices:", indices_to_remove)
        mask = torch.ones(len(X), dtype=torch.bool)
        mask[indices_to_remove] = False
        X = X[mask]
        Y = Y[mask]
    
    if max_len != None:
        dataloader = torch.utils.data.DataLoader(XYDataSet(X[:max_len], Y[:max_len]), batch_size=len(X), shuffle=shuffle)
    elif startIndex != None:
        dataloader = torch.utils.data.DataLoader(XYDataSet(X[startIndex:], Y[startIndex:]), batch_size=startIndex, shuffle=shuffle)
    else:
        dataloader = torch.utils.data.DataLoader(XYDataSet(X[:], Y[:]), batch_size=len(X), shuffle=shuffle)
    test_size = 10
    xt, yt = X[:test_size], Y[:test_size]
    # 注意D4的角度，缩小了10倍，还原的时候需要×10
    return X, Y, dataloader, dim_rope_seq, xt, yt