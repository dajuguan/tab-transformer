from mfnn.mfnn import FCNN
from mfnn.xydata import XYDataSet
import torch
import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np

x_low = np.loadtxt("./data/x_features.txt")
y_low = np.loadtxt("./data/loss_2d.txt")
y_high = np.loadtxt("./data/loss_3d.txt")

x_low = torch.from_numpy(x_low).to(torch.double)
y_low = torch.from_numpy(y_low).to(torch.double)
y_high = torch.from_numpy(y_high).to(torch.double)

loader_high = torch.utils.data.DataLoader(XYDataSet(x_low, y_low, y_high), batch_size=len(x_low))

## plot
# figure1(x_low, y_low, x_high,  y_high, x, y)

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 
print("training on: ", device)
model = FCNN(x_low.size()[-1], y_high.size()[-1], [16, 16], torch.nn.Tanh, low_fidelity_features=y_low.size()[-1])

model.to(device).to(torch.double)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
STEPS = 3000
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
N=2
model.eval()
x_test = x_low[:N].to(device)
y_low_test = y_low[:N].to(device)
y_pred = model(x_test, y_low_test).to("cpu").detach().numpy()

import numpy as np
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
print("y_pred:\n", y_pred[:, :7])
print("y_true:\n", y_high[:N].detach().numpy()[:, :7])
