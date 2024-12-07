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
def func(x):
    return np.power(x, 3)

num_data = 32
data_x_numpy = np.linspace(-4, 4, num_data)
data_y_numpy = func(data_x_numpy) + np.random.normal(0, 1,len(data_x_numpy))
data_y_true_numpy = func(data_x_numpy)

# print("data y:", data_y)
# plt.plot(data_x, data_y, ".")
# plt.plot(data_x, data_y_true)
# plt.savefig("./data.png")
# plt.close()

data_x = np.reshape(data_x_numpy, [num_data, 1])
data_y = np.reshape(data_y_numpy, [num_data, 1])

tensor_x = torch.Tensor(data_x) # transform to torch tensor
tensor_y = torch.Tensor(data_y)
toy_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
dataloader = DataLoader(toy_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True) # create your dataloader

# ==============================model==============================
class GaussianMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        mean, variance = torch.split(x, int(self.output_dim/2), dim=-1)
        variance = F.softplus(variance) + 1e-6 #Positive constraint
        return mean, variance
    

# ==============================model==============================
pi = torch.tensor(np.pi)

def gaussian_nll(mean, variance, y):
    return torch.mean((torch.log(variance) * 0.5) + ((0.5 * (y - mean).square()) / variance)) + torch.log(2 * pi)

class GaussianMLP(pl.LightningModule):
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

mlp = GaussianMLP(1,16,2)
trainer = pl.Trainer(devices=1, accelerator=DEVICE, max_epochs=MAX_EPOCHS)
trainer.fit(mlp, dataloader)

# ==============================eval==============================
mlp.eval()
means, variances  = mlp(tensor_x)
means = means.flatten()
variances = variances.flatten()
sigmas = torch.sqrt(variances)
upper = means + sigmas
lower = means - sigmas

plt.plot(data_x_numpy, data_y_numpy, ".")
plt.plot(data_x_numpy, data_y_true_numpy, "-r")
plt.plot(data_x_numpy, means.detach().numpy())
plt.fill_between(data_x_numpy, lower.detach().numpy(), upper.detach().numpy())
plt.savefig("./gaussian_mlp_1.png")
plt.close()

# ==============================N models train==============================
models = []
import random
import threading
def train(i):
    print(fr"thread {i} working ======")
    seed = random.randint(0, 9999)
    print(i, seed)
    pl.seed_everything(seed)
    mlp = GaussianMLP(1,16,2)
    trainer = pl.Trainer(devices=1, accelerator=DEVICE, max_epochs=MAX_EPOCHS)
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
means = torch.Tensor([])
variances = torch.Tensor([])
for model in models:
    model.eval()

    mu, vr  = model(tensor_x)
    means = torch.cat((means, mu), dim=1)
    variances = torch.cat((variances, vr), dim=1)


mean = means.mean(1)
variance = (variances +  means.pow(2)).mean(1) - mean.pow(2)
sigma = torch.sqrt(variance)
print(means.size(), variances.size(), mean.size(), sigma.size())

upper = mean + sigma
lower = mean - sigma

plt.plot(data_x_numpy, data_y_numpy, ".")
plt.plot(data_x_numpy, data_y_true_numpy, "-r")
plt.plot(data_x_numpy, mean.detach().numpy())
plt.fill_between(data_x_numpy, lower.detach().numpy(), upper.detach().numpy())
plt.savefig("./gaussian_mlp_N.png")
