import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

MAX_EPOCHS = 300
DEVICE= 'cpu' # 'gpu'
BATCH_SIZE = 8
# ==============================data==============================
def func1(x):
    return np.power(x, 3)

def func2(x):
    return np.power(x, 2)

# ============================= train data================
num_data = 32
data_x_numpy = np.linspace(-4, 4, num_data)
data_y1_numpy = func1(data_x_numpy) + np.random.normal(0, 1,len(data_x_numpy))
data_y_true_numpy = func1(data_x_numpy)
data_y2_numpy = func2(data_x_numpy) + np.random.normal(0, 0.5,len(data_x_numpy))
data_y2_true_numpy = func2(data_x_numpy)
# print("data y:", data_y)
# plt.plot(data_x, data_y, ".")
# plt.plot(data_x, data_y_true)
# plt.savefig("./data.png")
# plt.close()

data_x = np.reshape(data_x_numpy, [num_data, 1])
data_y = np.stack((data_y1_numpy, data_y2_numpy), axis = 1)

tensor_x = torch.Tensor(data_x) # transform to torch tensor
tensor_y = torch.Tensor(data_y)
OUTPUT_DIM = tensor_y.size()[1]
toy_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
dataloader = DataLoader(toy_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True) # create your dataloader

# ==============================test data =============================
num_data_test = 64
data_x_test_numpy  = np.linspace(-8, 8, num_data_test)
data_y_test_numpy = np.stack((func1(data_x_test_numpy), func2(data_x_test_numpy)), axis = 1)
data_x_test = np.reshape(data_x_test_numpy, [num_data_test, 1])
tensor_test_x = torch.Tensor(data_x_test) # t

# ==============================model==============================
class GaussianMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_variance_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_variance_dim = output_variance_dim

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_variance_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        mean, variance = torch.split(x, int(self.output_variance_dim/2), dim=-1)
        variance = F.softplus(variance) + 1e-6 #Positive constraint
        return mean, variance
    

# ==============================model==============================
pi = torch.tensor(np.pi)

# mean: samples * outputs
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

class GaussianMLP(pl.LightningModule):
    # outputdim = 2 * output(outputdim + it's variance dim)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = GaussianMLPModel(input_dim, hidden_dim, output_dim)
        self.loss_fn = gaussian_nll
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True
        # x = x.view(x.size(0), -1)
        mean, variance = self.forward(x)
        loss = self.loss_fn(mean, variance, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return optimizer

# ==============================1 model train==============================
# mlp = GaussianMLP(1,16,OUTPUT_DIM*2)
# trainer = pl.Trainer(devices=1, accelerator=DEVICE, max_epochs=MAX_EPOCHS, logger=False, enable_checkpointing=False)
# trainer.fit(mlp, dataloader)

# ==============================eval==============================
# mlp.eval()

# means_all, variances_all  = mlp(tensor_test_x)
# for i in range(0, OUTPUT_DIM):
#     means = means_all[:,i]
#     variances = variances_all[:, i]

#     means = means.flatten()
#     variances = variances.flatten()
#     sigmas = torch.sqrt(variances)
#     upper = means + sigmas
#     lower = means - sigmas

#     plt.plot(data_x_test_numpy, data_y_test_numpy[:,i], "r")
#     plt.plot(data_x_numpy, data_y[:,i], ".")
#     # plt.plot(data_x_numpy, data_y_true_numpy, "-r")
#     plt.plot(data_x_test_numpy, means.detach().numpy())
#     plt.fill_between(data_x_test_numpy, lower.detach().numpy(), upper.detach().numpy())
# plt.savefig(fr"./gaussian_mlp_mo.png")
# plt.close()

# ==============================N models train==============================
models = []
import random
import threading
def train(i):
    print(fr"thread {i} working ======")
    seed = random.randint(0, 9999)
    print(i, seed)
    pl.seed_everything(seed)
    mlp = GaussianMLP(1,16,OUTPUT_DIM*2)
    trainer = pl.Trainer(devices=1, accelerator=DEVICE, max_epochs=MAX_EPOCHS, logger=False, enable_checkpointing=False)
    dataloader = DataLoader(toy_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    trainer.fit(mlp, dataloader)
    models.append(mlp)
    print(fr"thread {i} done ======")

threads = []
for i in range(5):
    t = threading.Thread(target=train, args={i})
    threads.append(t)
    t.start()
for t in threads:
    t.join()

# ==============================N models eval==============================
means = []
variances = []
for model in models:
    model.eval()
    mu, vr  = model(tensor_test_x)
    means.append(mu)
    variances.append(vr)

# (models_dim, samples_dim, output_dim)
means_all = torch.stack(means)
variances_all = torch.stack(variances)

for i in range(0, OUTPUT_DIM):
    means = means_all[:,:, i]
    variances = variances_all[:,:, i]
    mean = means.mean(0)
    variance = (variances +  means.pow(2)).mean(0) - mean.pow(2)
    sigma = torch.sqrt(variance)
    print(means.size(), variances.size(), mean.size(), sigma.size())

    upper = mean + sigma
    lower = mean - sigma

    plt.plot(data_x_test_numpy, data_y_test_numpy[:,i], "r")
    plt.plot(data_x_numpy, data_y[:, i], ".")
    # plt.plot(data_x_numpy, data_y_true_numpy, "-r")
    plt.plot(data_x_test_numpy, mean.detach().numpy())
    plt.fill_between(data_x_test_numpy, lower.detach().numpy(), upper.detach().numpy())

plt.savefig("./gaussian_mlp_mo_N.png")
