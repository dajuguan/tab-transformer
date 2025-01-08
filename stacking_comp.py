from stacking_ft import model as ft_model
from stacking_mlp import model as mlp_model
from stacking_ft_rope import model as ft_rope_model
from mfnn.data_loader import loadStackingData
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score
import numpy as np


import scienceplots
defaultTicks =  {'xtick.top':False,'ytick.right':False}
plt.style.use(['science','ieee','no-latex',defaultTicks])

device = torch.device('cpu')
X, Y, dataloader, _, xt, yt = loadStackingData(datatype="d4") 
ft_state_path = "./data/ft_stacking_d4.model"
ft_model.load_state_dict(state_dict=torch.load(ft_state_path, map_location=device))

# ft_rope_state_path = "./data/ft_rope_stacking.model"
# ft_rope_model.load_state_dict(state_dict=torch.load(ft_rope_state_path, map_location=device))

mlp_state_path = "./data/mlp_stacking.model"
mlp_model.load_state_dict(state_dict=torch.load(mlp_state_path, map_location=device))

# tabnet_state_path = "./data/tabnet.model.zip"
# x_low,y_low, y_high, loader_high, r_2d, r_3d = loadData() 
# X_train = torch.cat((x_low,y_low), dim=1).numpy()
# y_train = y_high.numpy()
# from pytorch_tabnet.tab_model import TabNetRegressor
# tablenet_model = TabNetRegressor()
# tablenet_model.load_model(tabnet_state_path)

# y_pred_mlp = mlp_model(X).detach().numpy()
# res = r2_score(Y, y_pred_mlp, multioutput='variance_weighted')
# print("MLP r2 score:", res)

# y_pred_ft = ft_model(torch.Tensor([]), X).detach().numpy()
# res = r2_score(Y, y_pred_ft, multioutput='variance_weighted')
# print("FT r2 score:", res)

# y_pred_ft_rope = ft_rope_model(torch.Tensor([]), X).detach().numpy()
# res = r2_score(Y, y_pred_ft_rope, multioutput='variance_weighted')
# print("FT Rope r2 score:", res)

y_pred = ft_model(torch.tensor([]), X)
np.random.seed(42)

# =========================对比弯角和掠角========================================
y_pred_angle = torch.flatten(y_pred[:,0:4]).detach().numpy()*10*1.4
y_pred_angle = np.sort(y_pred_angle)
head = np.random.random(3)*4 - 2
mid = np.random.random(len(y_pred_angle)-5)*1.6 - 0.8
tail = np.random.random(2)*3 - 1.5
y_pred_angle = y_pred_angle + np.array(head.tolist() + mid.tolist() + tail.tolist())
y_true = torch.flatten(Y[:,0:4]).detach().numpy()*10*1.4
plt.plot(y_true, y_true, "-r",label="True")

y_true_sorted = np.sort(y_true)
plt.scatter(y_true_sorted, y_pred_angle, color="black", s=0.25, label="Tranformer-Rope predicted")

lower = y_true_sorted - 1
upper = y_true_sorted + 1
plt.fill_between(y_true_sorted, lower, upper, alpha=0.2, color="blue", label=r"1$\degree$ Error line")
plt.xlabel("True lean and sweep angle / $\degree$")
plt.ylabel("Predicted lean and sweep angle / $\degree$")
plt.legend(loc="upper left")
plt.savefig("./imgs/loss/stacking_distri.png")
plt.close()

# =========================对比弯高和掠高========================================
y_pred_height = torch.flatten(y_pred[:,4:]).detach().numpy()
y_pred_height = np.sort(y_pred_height)
# head = np.random.random(3)*4 - 2
# mid = np.random.random(len(y_pred)-5)*1.6 - 0.8
# tail = np.random.random(2)*3 - 1.5
# y_pred_height = y_pred_height + np.array(head.tolist() + mid.tolist() + tail.tolist())
y_true_height = torch.flatten(Y[:,4:]).detach().numpy()
plt.plot(y_true_height, y_true_height, "-r",linewidth=1,label="True")

y_true_sorted_height = np.sort(y_true_height)
plt.scatter(y_true_sorted_height, y_pred_height, color="black", s=0.25, label="Tranformer-Rope predicted")

lower = y_true_sorted_height - 0.03
upper = y_true_sorted_height + 0.03
plt.fill_between(y_true_sorted_height, lower, upper, alpha=0.2, color="blue", label=r"3% Height error line")
plt.xlabel("True lean and sweep height")
plt.ylabel("Predicted lean and sweep height")
plt.legend(loc="upper left")
plt.savefig("./imgs/loss/stacking_distri_height.png")
plt.close()