from three_d_reg_ft import model as ft_model
from three_d_reg_mlp import model as mlp_model
from mfnn.mfnn import FCNN
from mfnn.data_loader import loadData
import matplotlib.pyplot as plt
import torch

x_low,y_low, y_high, loader_high, r_2d, r_3d = loadData() 
ft_state_path = "./data/ft.model"
ft_model.load_state_dict(state_dict=torch.load(ft_state_path))

mlp_state_path = "./data/mlp.model"
mlp_model.load_state_dict(state_dict=torch.load(mlp_state_path))

tabnet_state_path = "./data/tabnet.model.zip"
x_low,y_low, y_high, loader_high, r_2d, r_3d = loadData() 
X_train = torch.cat((x_low,y_low), dim=1).numpy()
y_train = y_high.numpy()
from pytorch_tabnet.tab_model import TabNetRegressor
tablenet_model = TabNetRegressor()
tablenet_model.load_model(tabnet_state_path)

for i in range(1, 50):
    x_test = x_low[i-1:i]
    y_low_test = y_low[i-1:i]

    y_pred_ft = ft_model(x_test, y_low_test).detach().numpy()[0]
    y_pred_mlp = mlp_model(x_test, y_low_test).detach().numpy()[0]
    y_pred_tabnet = tablenet_model.predict(X_train[i-1:i])[0]
    y_true = y_high[i]

    plt.plot(y_low[i]/100, r_2d, "-b", label="2d")
    plt.plot(y_pred_ft/100, r_3d, '-r',label="ft")
    plt.plot(y_pred_mlp/100, r_3d, ':y',label="mlp")
    plt.plot(y_pred_tabnet/100, r_3d, ':g',label="tabnet")
    plt.plot(y_true/100, r_3d, '--k',label="true")
    plt.legend()
    plt.savefig(fr"./imgs/{i}.png")
    plt.close()
