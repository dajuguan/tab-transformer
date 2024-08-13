import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
import matplotlib.pyplot as plt
import random
import time
import tqdm
from mfnn.data_loader import loadStackingData


import sys
sys.path.append("../")

import torch.nn as nn
class MySimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.

    **Example**::

        >>> from avalanche.models import SimpleMLP
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleMLP(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1,
        drop_rate=0.5,
    ):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()

        layers = nn.Sequential(
            *(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rate),
            )
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size),
                        nn.LeakyReLU(inplace=True),
                        nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# setup_seed(30)

X, Y, dataloader, _, xt, yt = loadStackingData()
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 

# pressure_distribution = np.loadtxt(r"/home/po/phd/avalanche/pressure_distribution.csv", delimiter=",", dtype=np.float32)
# data = np.loadtxt(r"/home/po/phd/avalanche/data.csv", delimiter=",",skiprows=1, dtype=np.float32)
# X = torch.from_numpy(pressure_distribution)
# Y = torch.from_numpy(data[:,13:16]) / 5
# dataset  = torch.utils.data.dataset.TensorDataset(X, Y)   
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(X), pin_memory=True)

model = MySimpleMLP(Y.shape[-1], X.shape[-1], hidden_size=128, hidden_layers=5, drop_rate=0.00)

# model.to(torch.double)
model_path = "./data/mlp_stacking.model"

loss_path = "./data/mlp_stacking.loss"
def train(model, dataloader, critrion, optimizer, steps, device="cpu"):
        "Train the model by given dataloader."
        t_start = time.time()
        train_loss, test_loss = [], []
        _xt = xt.to(device)
        model.train()
        epochs_iter = tqdm.tqdm(range(steps), desc="Epoch")
        tq = tqdm.tqdm(dataloader, desc="train", ncols=None, leave=False, unit="batch")
        # User defined preprocess
        for epoch in epochs_iter:
            runningLoss = 0.0
            for x, y in tq:
                x = x.to(device)
                y = y.to(device)

                # compute prediction error
                y_pred = model(x)
                loss = critrion(y_pred, y)
                runningLoss += float(loss.item())
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tq.set_postfix(loss=f"{loss.item():.4e}")

                # record results
            epochs_iter.set_postfix(loss=f"{runningLoss:.4e}")
            
            scheduler.step()

            yt_pred = model(_xt).to("cpu")
            loss_test = critrion(yt_pred, yt).item()
            train_loss.append(runningLoss)
            test_loss.append(loss_test)

        torch.save(model.state_dict(), model_path)
        print("training complete", f'wall time = {time.time()- t_start:.2f}s')
        torch.save({"train": train_loss, "test": test_loss}, loss_path)




if __name__ == "__main__":
    #  print("x_low", x_low.shape, y_high.shape, y_low.shape)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    STEPS = 2000
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5*STEPS, 0.8*STEPS])
    critrion = torch.nn.L1Loss()
    critrion = torch.nn.MSELoss()
    critrion = torch.nn.SmoothL1Loss()

    train(model, dataloader, critrion, optimizer, STEPS, device)
    ## eval
    model.load_state_dict(state_dict=torch.load(model_path))
    N=10
    model.eval()

    x_test = X[:N].to(device)
    y_low_test = Y[:N].to(device)
    y_pred = model(x_test).to("cpu").detach().numpy()

    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print("y_pred:\n", y_pred)
    print("y_true:\n", Y[:N].detach().numpy())
