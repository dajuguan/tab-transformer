import matplotlib.pyplot as plt
import torch
import numpy as np

import scienceplots
defaultTicks =  {'xtick.top':False,'ytick.right':False}
plt.style.use(['science','ieee','no-latex',defaultTicks])

ft_loss_path = "./data/ft.loss"
ft_loss = torch.load(ft_loss_path)
ft_loss_train, ft_loss_test = np.array(ft_loss["train"]), np.array(ft_loss["test"])
plt.yscale("log")

indexs = np.arange(start=0, stop=len(ft_loss_train), step=15)
plt.plot(indexs, ft_loss_train[indexs], "--r",label="Transformer Train")
plt.plot(indexs, ft_loss_test[indexs], "-r",label="Transformer Test")

mlp_emb_loss_path = "./data/mlp_embedding.loss"
mlp_emb_loss = torch.load(mlp_emb_loss_path)
mlp_emb_loss_train, mlp_emb_loss_test = mlp_emb_loss["train"], mlp_emb_loss["test"]
plt.plot(mlp_emb_loss_train, "--b", label="MLP-Embedding Train")
plt.plot(mlp_emb_loss_test, "b", label="MLP-Embedding Test")

mlp_loss_path = "./data/mlp.loss"
mlp_loss = torch.load(mlp_loss_path)
mlp_loss_train, mlp_loss_test = mlp_loss["train"], mlp_loss["test"]
plt.plot(mlp_loss_train, "--k", label="MLP Train")
plt.plot(mlp_loss_test, "-k", label="MLP Test")

plt.xlabel("Epochs")
plt.ylabel("SmoothL1Loss")
plt.legend()
plt.savefig("./imgs/loss/loss.png")

# mlp_state_path = "./data/mlp.model"
# mlp_model.load_state_dict(state_dict=torch.load(mlp_state_path))

# mlp_emb_state_path = "./data/mlp_embedding.model"
# mlp_emb_model.load_state_dict(state_dict=torch.load(mlp_emb_state_path))


