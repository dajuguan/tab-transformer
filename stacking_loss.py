import matplotlib.pyplot as plt
import torch
import numpy as np

import scienceplots
defaultTicks =  {'xtick.top':False,'ytick.right':False}
plt.style.use(['science','ieee','no-latex',defaultTicks])

plt.yscale("log")


mlp_loss_path = "./data/mlp_stacking.loss"
mlp_loss = torch.load(mlp_loss_path)
mlp_loss_train, mlp_loss_test = mlp_loss["train"], mlp_loss["test"]
plt.plot(mlp_loss_train, "--k", label="MLP train")
plt.plot(mlp_loss_test, "-k", label="MLP test")

ft_loss_path = "./data/ft_stacking.loss"
ft_loss = torch.load(ft_loss_path)
ft_loss_train, ft_loss_test = np.array(ft_loss["train"]), np.array(ft_loss["test"])
plt.plot(ft_loss_test, "--b", label="Transformer train")
plt.plot(ft_loss_train, "-b", label="Transformer test")


ft_rope_loss_path = "./data/ft_rope_stacking.loss"
ft_rope_loss = torch.load(ft_rope_loss_path)
ft_rope_loss_train, ft_rope_loss_test = np.array(ft_rope_loss["train"]), np.array(ft_rope_loss["test"])
plt.plot(ft_rope_loss_test, "--r", label="Transformer-Rope train")
plt.plot(ft_rope_loss_train, "-r", label="Transformer-Rope test")

# indexs = np.arange(start=0, stop=len(ft_loss_train), step=15)
# plt.plot(indexs, ft_loss_train[indexs], "--r",label="Attention Train")
# plt.plot(indexs, ft_loss_test[indexs], "-r",label="Attention Test")

plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.savefig("./imgs/loss/loss_stacking.png")

# mlp_state_path = "./data/mlp.model"
# mlp_model.load_state_dict(state_dict=torch.load(mlp_state_path))

# mlp_emb_state_path = "./data/mlp_embedding.model"
# mlp_emb_model.load_state_dict(state_dict=torch.load(mlp_emb_state_path))


