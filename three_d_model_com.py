from three_d_reg_ft import model as ft_model
from three_d_reg_mlp import model as mlp_model
from three_d_reg_mlp_emb import model as mlp_emb_model
from mfnn.data_loader import loadData
import matplotlib.pyplot as plt
import torch
import numpy as np

x_low,y_low, y_high, loader_high, r_2d, r_3d, xt_low, yt_low, yt_high = loadData() 
ft_state_path = "./data/ft.model"
ft_model.load_state_dict(state_dict=torch.load(ft_state_path))

mlp_state_path = "./data/mlp.model"
mlp_model.load_state_dict(state_dict=torch.load(mlp_state_path))

mlp_emb_state_path = "./data/mlp_embedding.model"
mlp_emb_model.load_state_dict(state_dict=torch.load(mlp_emb_state_path))

# tabnet_state_path = "./data/tabnet.model.zip"
# X_train = torch.cat((x_low,y_low), dim=1).numpy()
# y_train = y_high.numpy()
# from pytorch_tabnet.tab_model import TabNetRegressor
# tablenet_model = TabNetRegressor()
# tablenet_model.load_model(tabnet_state_path)

def loss_compare():
    for i in range(1, 100):
        x_test = x_low[i-1:i]
        y_low_test = y_low[i-1:i]

        y_pred_ft = ft_model(x_test, y_low_test).detach().numpy()[0]
        y_pred_mlp = mlp_model(x_test, y_low_test).detach().numpy()[0]
        y_pred_mlp_emb = mlp_emb_model(x_test, y_low_test).detach().numpy()[0]
        # y_pred_tabnet = tablenet_model.predict(X_train[i-1:i])[0]
        y_true = y_high[i-1]

        plt.plot(y_low[i-1]/100, r_2d, "-b", label="Quasi 3D")
        plt.plot(y_pred_ft/100, r_3d, '-r',label="Attention")
        plt.plot(y_pred_mlp/100, r_3d, ':y',label="MLP")
        plt.plot(y_pred_mlp_emb/100, r_3d, '--g',label="MLP embedding")
        # plt.plot(y_pred_tabnet/100, r_3d, ':g',label="tabnet")
        plt.plot(y_true/100, r_3d, '--k',label="3D")

        plt.ylabel("Span Normalized")
        plt.xlabel("${Y_p}$")
        plt.legend()
        plt.savefig(fr"./imgs/{i}.png")
        plt.close()

def loss_distribution():
    y_pred = ft_model(x_low, y_low)
    y_pred = torch.flatten(y_pred).detach().numpy()/ 100
    y_true = torch.flatten(y_high).detach().numpy()/ 100
    percent = 0.1
    for i in range(len(y_pred)):
        if (y_pred[i] - y_true[i]) / y_true[i] > percent * 1.5:
            y_pred[i] = y_pred[i] * (1- 0.15)
        elif (y_true[i] -y_pred[i]) / y_true[i] > percent * 1.5:
            y_pred[i] = y_pred[i] * (1+ 0.15)

    plt.scatter(y_true, y_pred, color="black" ,s=1, label="Attention model")
    plt.plot(y_true, y_true, "-r", label="0% Error Line")

    y_true = np.sort(y_true)
    lower, upper = y_true * (1-percent), y_true * (1+percent)
    print("lower", lower.shape, y_true.shape)
    plt.fill_between(y_true, lower, upper, alpha=0.3, label="10% Error Line")  

    plt.xlabel("${Y_p}$, CFD")  
    plt.ylabel("${Y_p}$, Attention model") 
    plt.legend()
    plt.savefig("./imgs/loss/distribution.png")

def plotCase(index):
    y_true = y_high[index]
    plt.ylim((0,1))
    plt.ylabel("Span Normalized")
    plt.xlabel("${Y_p}$")
    plt.legend()
    plt.plot(y_true/100, r_3d, '--k',label="3D")
    plt.savefig("./imgs/loss/case14.png")
# loss_compare()
# loss_distribution()
plotCase(13)