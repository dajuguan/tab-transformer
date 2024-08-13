from stacking_ft_rope import model as ft_model
from stacking_mlp import model as mlp_model
from stacking_ft_rope import model as ft_rope_model
from mfnn.data_loader import loadStackingData
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score
import numpy as np

X, Y, dataloader, _, xt, yt = loadStackingData() 
ft_state_path = "./data/ft_stacking.model"
ft_model.load_state_dict(state_dict=torch.load(ft_state_path))

ft_rope_state_path = "./data/ft_rope_stacking.model"
ft_rope_model.load_state_dict(state_dict=torch.load(ft_rope_state_path))

mlp_state_path = "./data/mlp_stacking.model"
mlp_model.load_state_dict(state_dict=torch.load(mlp_state_path))



# tabnet_state_path = "./data/tabnet.model.zip"
# x_low,y_low, y_high, loader_high, r_2d, r_3d = loadData() 
# X_train = torch.cat((x_low,y_low), dim=1).numpy()
# y_train = y_high.numpy()
# from pytorch_tabnet.tab_model import TabNetRegressor
# tablenet_model = TabNetRegressor()
# tablenet_model.load_model(tabnet_state_path)

y_pred_mlp = mlp_model(X).detach().numpy()
res = r2_score(Y, y_pred_mlp, multioutput='variance_weighted')
print("MLP r2 score:", res)

y_pred_ft = ft_model(torch.Tensor([]), X).detach().numpy()
res = r2_score(Y, y_pred_ft, multioutput='variance_weighted')
print("FT r2 score:", res)

y_pred_ft_rope = ft_rope_model(torch.Tensor([]), X).detach().numpy()
res = r2_score(Y, y_pred_ft_rope, multioutput='variance_weighted')
print("FT Rope r2 score:", res)


y_pred = ft_rope_model(torch.tensor([]), X)
y_pred = torch.flatten(y_pred).detach().numpy()*5
y_true = torch.flatten(Y).detach().numpy()*5
plt.plot(y_true, y_true, "-r",label="True")
plt.scatter(y_true, y_pred, color="black", s=8, label="Predicted")

y_true_sorted = np.sort(y_true)
lower = y_true_sorted - 1
upper = y_true_sorted + 1
plt.fill_between(y_true_sorted, lower, upper, alpha=0.8, label="1° Error Line")
plt.xlabel("True bow and lean angle / °")
plt.ylabel("Predicted angle / °")
plt.legend()
plt.savefig("./imgs/loss/stacking_distri.png")