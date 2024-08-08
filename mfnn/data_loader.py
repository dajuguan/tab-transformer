import numpy as np
import torch
import sys
sys.path.append("..")
from mfnn.xydata import XYDataSet

def loadData():
    x_low = np.loadtxt("./data/x_features.txt")
    y_low = np.loadtxt("./data/loss_2d.txt")
    y_high = np.loadtxt("./data/loss_3d.txt")
    r_2d = np.loadtxt("./data/r_2d.txt")
    r_3d = np.loadtxt("./data/r_3d.txt")

    x_low = torch.from_numpy(x_low).to(torch.double)
    x_low = (x_low - x_low.min() ) / ( x_low.max() - x_low.min())
    y_low = torch.from_numpy(y_low).to(torch.double) * 100.0
    y_high = torch.from_numpy(y_high).to(torch.double) * 100.0

    loader_high = torch.utils.data.DataLoader(XYDataSet(x_low, y_low, y_high), batch_size=len(x_low))
    return x_low,y_low, y_high, loader_high, r_2d, r_3d