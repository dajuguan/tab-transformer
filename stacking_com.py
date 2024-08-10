from stacking_ft import model as ft_model
from stacking_mlp import model as mlp_model
from mfnn.data_loader import loadStackingData
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score

X, Y, dataloader = loadStackingData() 
ft_state_path = "./data/ft_stacking.model"
ft_model.load_state_dict(state_dict=torch.load(ft_state_path))

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

# for i in range(1, 50):
#     x_test = X[i-1:i]
#     y_true = Y[i]

#     y_pred_ft = ft_model(torch.tensor([]), x_test).detach().numpy()[0]
#     y_pred_mlp = mlp_model(x_test).detach().numpy()[0]
#     # y_pred_tabnet = tablenet_model.predict(X_train[i-1:i])[0]
#     print("===================:", i)
#     print("ft diff:", y_true - y_pred_ft)
#     print("mlp diff:", y_true - y_pred_mlp)