from mfnn.mfnn import FCNN
from mfnn.xydata import XYDataSet
import torch
import tqdm
import matplotlib.pyplot as plt
import time


def func_low(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(8 * torch.pi * x)


def func_high(x: torch.Tensor) -> torch.Tensor:
    return (x - 2**.5) * func_low(x) ** 2

def figure1(x_low, y_low, x_high,  y_high, x, y):
    "Plot the multi-fidelity data along with true data."
    y_high = func_high(x_high)

    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.plot(x_low, y_low, 'o', color='None', markeredgecolor='b', label='low')
    ax.plot(x_high, y_high, 'rx', label='high')
    ax.plot(x, y, 'k:', label='true')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(x[0], x[-1])
    ax.grid()
    fig.tight_layout(pad=0)
    plt.savefig("./imgs/multi.png")
    return fig

x_low = torch.linspace(0, 1, 80).reshape(-1, 10)
y_low = torch.linspace(0.2, 0.8, 16).reshape(-1, 2)
y_high = torch.cat((x_low, y_low), dim=1) + func_high(torch.cat((x_low, y_low), dim=1))
# y_high = y_high[:, :-5]
loader_high = torch.utils.data.DataLoader(XYDataSet(x_low, y_low, y_high), batch_size=len(x_low))

## plot
# figure1(x_low, y_low, x_high,  y_high, x, y)

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 
print("training on: ", device)
model = FCNN(x_low.size()[-1], y_high.size()[-1], [16, 16], torch.nn.Tanh, low_fidelity_features=y_low.size()[-1])

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
STEPS = 5000
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.6*STEPS, 0.8*STEPS])
critrion = torch.nn.MSELoss()

def train(model, dataloader, critrion, optimizer, steps, device="cpu"):
        "Train the model by given dataloader."
        t_start = time.time()
        model.train()
        epochs_iter = tqdm.tqdm(range(steps), desc="Epoch")
        tq = tqdm.tqdm(dataloader, desc="train", ncols=None, leave=False, unit="batch")
        # User defined preprocess
        for epoch in epochs_iter:
            runningLoss = 0.0
            for x, y_low, y in tq:
                x = x.to(device)
                y_low = y_low.to(device)
                y = y.to(device)

                # compute prediction error
                y_pred = model(x, y_low)
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
        print("training complete", f'wall time = {time.time()- t_start:.2f}s')

train(model, loader_high, critrion, optimizer, STEPS, device)


## eval
N=5
model.eval()
x_test = x_low[:N].to(device)
y_low_test = y_low[:N].to(device)
y_pred = model(x_test, y_low_test).to("cpu").detach().numpy()

import numpy as np
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
print("y_pred:\n", y_pred[:, :7])
print("y_true:\n", y_high[:N].detach().numpy()[:, :7])
