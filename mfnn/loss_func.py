import torch
import numpy as np

pi = torch.tensor(np.pi)
def gaussian_nll(means, variances, ys):
    outdim = means.size()[1]
    nll = 0
    for i in range(0, outdim):
        variance = variances[:, i]
        y = ys[:,i]
        mean = means[:,i]
        nll +=  torch.mean((torch.log(variance) * 0.5) + ((0.5 * (y - mean).square()) / variance)) + torch.log(2 * pi)
    nll = nll/outdim
    return nll