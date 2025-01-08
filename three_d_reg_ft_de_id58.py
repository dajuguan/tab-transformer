# 带并行版本
# in domain
from mfnn.mfnn_de import FCNN_DE
from mfnn.xydata import XYDataSet
import torch
import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
from mfnn.data_loader import loadData
from mfnn.loss_func import gaussian_nll
import random

SCALE=100
START=57
x_low,y_low, y_high, loader_high, r_2d, r_3d, xt_low, yt_low, yt_high = loadData(start=START,scale=SCALE) 
# print("x_low", x_low.shape, y_high.shape, y_low.shape)

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 
def de_model():
    model = FCNN_DE(
        x_low.size()[-1],
        y_high.size()[-1] * 2, 
        [128], 
        0, 
        torch.nn.LeakyReLU, 
        low_fidelity_features=y_low.size()[-1],
        enable_embedding=False,
        const_variance=1e-9
        )
    model.to(torch.double)
    return model

def train(i, critrion, steps, device="cpu"):
    print(f"{i}-th training start on {device}...")
    seed = random.randint(0, 99999)
    torch.manual_seed(seed)

    model = de_model()
    model.to(device)
    model_path = fr"./data/ft_de{i}.id.model"
    loss_path = fr"./data/ft_de{i}.id.loss"

    _,_, _, dataloader, _, _, _, _, _ = loadData(shuffle=True, resampleIndex=57,scale=SCALE,start=START) 

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5*steps, 0.8*steps])

    "Train the model by given dataloader."
    t_start = time.time()
    train_loss, test_loss = [], []
    _xt_low, _yt_low = xt_low.to(device), yt_low.to(device)
    model.train()
    epochs_iter = tqdm.tqdm(range(steps), desc=f"Epoch{i}", position=i)
    tq = tqdm.tqdm(dataloader, desc="train", ncols=None, leave=False, unit="batch")
    # User defined preprocess
    for epoch in epochs_iter:
        runningLoss = 0.0
        for x, y_low, y in tq:
            x = x.to(device)
            y_low = y_low.to(device)
            y = y.to(device)

            # compute prediction error
            y_pred, y_variance = model(x, y_low)
            loss = critrion(y_pred, y_variance, y)
            runningLoss += float(loss.item())
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.set_postfix(loss=f"{loss.item():.4e}")

        epochs_iter.set_postfix(loss=f"{runningLoss:.4e}")
        # record results
        # run test
        # yt_pred, yt_variance = model(_xt_low, _yt_low)
        # loss_test = critrion(yt_pred.to("cpu"), yt_variance.to("cpu"),yt_high).item()
        # train_loss.append(runningLoss)
        # test_loss.append(loss_test)

        scheduler.step()
    print(f"{i}-th training complete", f'wall time = {time.time()- t_start:.2f}s')
    torch.save(model.state_dict(), model_path)
    # torch.save({"train": train_loss, "test": test_loss}, loss_path)


import threading
import multiprocessing as mp
STEPS = 600
N_MODELS = 5
# critrion = torch.nn.L1Loss()
# critrion = torch.nn.MSELoss()
# critrion = torch.nn.SmoothL1Loss()
critrion = gaussian_nll
def de_train_cpu():
    threads = []
    for i in range(N_MODELS):
        t = threading.Thread(target=train, args={i, critrion, STEPS, device})
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def de_train_gpu():
    torch.multiprocessing.set_start_method('spawn')
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(N_MODELS)
    for i in range(N_MODELS):
        device = torch.device(f"cuda:{i%2}") 
        pool.apply_async(train, args=[i, critrion, STEPS, device])
    pool.close()
    pool.join()

def predict(tensor_x, tensor_y_low):
    models = []
    means = []
    variances = []
    for i in [0,3]:
    # for i in range(N_MODELS):
        model = de_model()
        model.eval()
        # print("device:", next(model.parameters()).device, tensor_x.device, tensor_y_low.device)
        model_path = fr"./data/ft_de{i}.id.model"
        model.load_state_dict(state_dict=torch.load(model_path))
        models.append(model)
    
        mu, vr  = model(tensor_x, tensor_y_low)
        means.append(mu)
        variances.append(vr)

    means_all = torch.stack(means)
    variances_all = torch.stack(variances)

    mean_avg = means_all.mean(0)
    variance_avg = variances_all.mean(0)
    variance_avg = (variance_avg +  mean_avg.pow(2)).mean(0) - mean_avg.pow(2)
    sigma_avg = torch.sqrt(variance_avg)

    upper = mean_avg + sigma_avg
    lower = mean_avg - sigma_avg

    mean_avg = mean_avg.to("cpu").detach().numpy()
    sigma_avg = sigma_avg.to("cpu").detach().numpy()
    lower = lower.to("cpu").detach().numpy()
    upper = upper.to("cpu").detach().numpy()
    return mean_avg, sigma_avg, lower, upper

import scienceplots
defaultTicks =  {'xtick.top':False,'ytick.right':False}
plt.style.use(['science','ieee','no-latex',defaultTicks])
if __name__ == "__main__":
    ## train
    # de_train_gpu()

    ## eval
    N = 1
    x_test = x_low[:N]
    y_low_test = y_low[:N]
    y_pred, y_sigma, y_lower, y_upper = predict(x_test, y_low_test)

    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print("y_pred:\n", y_pred[:, :7],y_pred[:, -7:])
    print("y_true:\n", y_high[:N].detach().numpy()[:, :7], y_high[:N].detach().numpy()[:, -7:])

    for i in range(58-START,59-START):
    # for i in [4, 16, 58]:
        x_test = x_low[i-1:i]
        y_low_test = y_low[i-1:i]

        y_pred, y_sigma, y_lower, y_upper = predict(x_test, y_low_test)
        y_pred = y_pred[0]
        y_lower = y_lower[0]
        y_upper = y_upper[0]

        y_true = y_high[i-1].detach().numpy()

        plt.plot(y_low[i-1]/SCALE, r_2d, "-b", label="Quasi 3D")
        plt.plot(y_true/SCALE, r_3d, '--k',label="3D")
        plt.plot(y_pred/SCALE, r_3d, '-r',label="DET predicted")
        plt.fill_betweenx(r_3d, y_lower/SCALE, y_upper/SCALE, alpha=0.2, color="red",label="$\sigma$")
        plt.xlabel("${Y_p}$")
        plt.ylabel("Normalized span")
        plt.legend()
        plt.savefig(fr"./imgs/ft_withpaper/{i}.id.png")
        plt.close()
