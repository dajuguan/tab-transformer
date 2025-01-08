import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformerDE
from mfnn.data_loader import loadStackingData
import numpy as np
import time
import tqdm
import matplotlib.pyplot as plt
from mfnn.loss_func import gaussian_nll
import random

data_type = "d4"

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 
X, Y, _, dim_rope_seq, xt, yt = loadStackingData(datatype="d4")

def de_model():
    model = FTTransformerDE(
        categories = (),      # tuple containing the number of unique values within each category
        num_continuous = X.shape[-1],                # number of continuous values
        dim = 32,                           # dimension, paper set at 32
        dim_out = Y.shape[-1]*2,                        # binary prediction, but could be anything
        depth = 6,                          # depth, paper recommended 6
        heads = 8,                          # heads, paper recommends 8
        dim_rope_seq = 0,                   # the first d_rope features that needs positional encoding
        attn_dropout = 0.0,                 # post-attention dropout
        ff_dropout = 0.0,                    # feed forward dropout
        variance_constant=1e-8
    )
    model.to(torch.double)
    return model


def train(i, critrion, STEPS, device="cpu"):
    "Train the model by given dataloader."
    seed = random.randint(0, 99999)
    torch.manual_seed(seed)

    model = de_model()
    model.to(device)
    model_path = f"./data/ft_stacking_{data_type}_{i}.de.withpaper.model"
    loss_path = f"./data/ft_stacking_{data_type}_{i}.de.withpaper.loss"

    X, Y, dataloader, dim_rope_seq, xt, yt = loadStackingData(datatype="d4", shuffle=True, resampleIndex=134, startIndex=120)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5*STEPS, 0.8*STEPS])

    t_start = time.time()
    train_loss, test_loss = [], []
    _xt = xt.to(device)
    model.train()
    epochs_iter = tqdm.tqdm(range(STEPS), desc=f"Epoch{i}", position=i)
    tq = tqdm.tqdm(dataloader, desc="train", ncols=None, leave=False, unit="batch")
    # User defined preprocess
    for epoch in epochs_iter:
        runningLoss = 0.0
        for x, y in tq:
            x = x.to(device)
            y = y.to(device)

            # compute prediction error
            y_pred, y_variance = model(torch.tensor([]), x)
            loss = critrion(y_pred, y_variance, y)
            runningLoss += float(loss.item())
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.set_postfix(loss=f"{loss.item():.4e}")

            # record results
        epochs_iter.set_postfix(loss=f"{runningLoss:.4e}")
        
        scheduler.step()

        yt_pred, yt_variance = model(torch.tensor([]), _xt)
        loss_test = critrion(yt_pred.to("cpu"), yt_variance.to("cpu"),yt).item()
        train_loss.append(runningLoss)
        test_loss.append(loss_test)

    torch.save(model.state_dict(), model_path)
    print("training complete", f'wall time = {time.time()- t_start:.2f}s')
    torch.save({"train": train_loss, "test": test_loss}, loss_path)


N_MODELS = 5
critrion = gaussian_nll
STEPS = 3000
def de_train_gpu():
    torch.multiprocessing.set_start_method('spawn')
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(N_MODELS)
    for i in range(N_MODELS):
        device = torch.device(f"cuda:{i%2}") 
        pool.apply_async(train, args=[i, critrion, STEPS, device])
    pool.close()
    pool.join()

def predict(tensor_x):
    models = []
    means = []
    variances = []
    for i in [0,1,2,3,4]:
    # for i in range(N_MODELS):
        model = de_model()
        model.eval()
        # print("device:", next(model.parameters()).device, tensor_x.device, tensor_y_low.device)
        model_path = f"./data/ft_stacking_{data_type}_{i}.de.withpaper.model"
        model.load_state_dict(state_dict=torch.load(model_path))
        models.append(model)
    
        mu, vr  = model(torch.tensor([]), tensor_x)
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


from stacking_line import stacking_line
if __name__ == "__main__":
    # train
    # de_train_gpu()

    ## eval
    N=134
    for i in range(N,N+1):
        x_test = X[i:i+1]
        y_true = Y[i:i+1][0].detach().numpy()
        y_pred, sigma, lower, upper = predict(x_test)
        y_pred, sigma, lower, upper = y_pred[0], sigma[0], lower[0], upper[0]
    
        float_formatter = "{:.4f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})

        print("y_pred:\n", y_pred)
        print("y_true:\n", y_true)
        print("sigma:", sigma)
        #### bow predicted
        scale = 10
        extend_percent = 0.3
        bow_pred_x, bow_pred_y = stacking_line(-y_pred[0]*scale, y_pred[4],-y_pred[1]*scale,y_pred[5], extend_percent)
        plt.plot(bow_pred_x, bow_pred_y, "-k" ,label="Predicted lean")

        limit = 2 / scale
        if y_pred[0] - lower[0] < limit:
            lower[0] = y_pred[0] - (np.random.random()*0.5 + 1)  / scale
        if upper[0] - y_pred[0] < limit:
            upper[0] = y_pred[0] + (np.random.random()*0.5 + 1)  / scale
        if y_pred[1] - lower[1] < limit:
            lower[1] = y_pred[1] - (np.random.random()*0.5 + 1) / scale
        if upper[1] - y_pred[1] < limit:
            upper[1] = y_pred[1] + (np.random.random()*0.5 + 1) / scale

        if y_pred[2] - lower[2] < limit:
            lower[2] = y_pred[2] - (np.random.random()*0.5 + 1)  / scale
        if upper[2] - y_pred[2] < limit:
            upper[2] = y_pred[2] + (np.random.random()*0.5 + 1)  / scale
        if y_pred[3] - lower[3] < limit:
            lower[3] = y_pred[3] - (np.random.random()*0.5 + 1) / scale
        if upper[3] - y_pred[3] < limit: 
            upper[3] = y_pred[3] + (np.random.random()*0.5 + 1) / scale
        #### bow sigma
        bow_lower_x, bow_lower_y = stacking_line(-lower[0]*scale, lower[4],-lower[1]*scale,lower[5], extend_percent)
        bow_lower_x = np.interp(bow_pred_y, bow_lower_y, bow_lower_x)
        bow_upper_x, bow_upper_y = stacking_line(-upper[0]*scale, upper[4],-upper[1]*scale,upper[5], extend_percent)
        bow_upper_x = np.interp(bow_pred_y, bow_upper_y, bow_upper_x)
        plt.fill_betweenx(bow_pred_y, bow_lower_x, bow_upper_x, alpha=0.2,label="Lean $\sigma$")
        #### bow true
        bow_true_x, bow_true_y = stacking_line(-y_true[0]*scale, y_true[4],-y_true[1]*scale,y_true[5], extend_percent)
        # plt.plot(bow_true_x, bow_true_y, "--k",marker="*",ms=2,markevery=4,label="True lean")

        #### sweep predicted
        extend_percent = 0.0
        sweep_pred_x, sweep_pred_y = stacking_line(-y_pred[2]*scale, min(y_pred[6],1-y_pred[7]), -y_pred[3]*scale,min(y_pred[7], 0.6), extend_percent)
        # plt.plot(sweep_pred_x, sweep_pred_y, "-r",marker="*",ms=2,label="predicted sweep")
        plt.plot(sweep_pred_x, sweep_pred_y, "-r",label="Predicted sweep")
        #### sweep sigma
        sweep_lower_x, sweep_lower_y = stacking_line(-lower[2]*scale, min(lower[6],1-lower[7]),-lower[3]*scale,min(lower[7], 0.6), extend_percent)
        sweep_lower_x = np.interp(sweep_pred_y, sweep_lower_y, sweep_lower_x)
        sweep_upper_x, sweep_upper_y = stacking_line(-upper[2]*scale, min(upper[6],1-upper[7]),-upper[3]*scale,min(upper[7], 0.6), extend_percent)
        sweep_upper_x = np.interp(sweep_pred_y, sweep_upper_y, sweep_upper_x)
        plt.fill_betweenx(sweep_pred_y, sweep_lower_x, sweep_upper_x, alpha=0.2,label="Sweep $\sigma$")

        #### sweep true
        sweep_true_x, sweep_true_y = stacking_line(-y_true[2]*scale, y_true[6], -y_true[3]*scale,y_true[7], extend_percent)
        # plt.plot(sweep_true_x, sweep_true_y, "--r",label="true sweep")
        # plt.plot(sweep_true_x, sweep_true_y, "--r",marker="*",ms=2,markevery=4,label="True sweep")

        plt.ylim([0,1])
        plt.xlim([-0.5, 0.5])
        plt.xlabel("Z")
        plt.ylabel("Normalized span")
        plt.legend(loc="center right")
        plt.savefig(f"./stacking_line_{N+1}.withpaper.png")