from mfnn.mfnn import FCNN
from mfnn.xydata import XYDataSet
import torch
import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
from mfnn.data_loader import loadData

x_low,y_low, y_high, loader_high, r_2d, r_3d = loadData() 
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 
model = FCNN(x_low.size()[-1], y_high.size()[-1], [32, 64, 128], 0, torch.nn.LeakyReLU, low_fidelity_features=y_low.size()[-1])
model.to(torch.double)
model_path = "./data/ft.model"
# print("x_low", x_low.shape, y_high.shape, y_low.shape)

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
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    print("training on: ", device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    STEPS = 5000
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5*STEPS, 0.8*STEPS])
    critrion = torch.nn.L1Loss()
    critrion = torch.nn.MSELoss()
    train(model, loader_high, critrion, optimizer, STEPS, device)
    ## eval
    N=1
    model.eval()
    x_test = x_low[:N].to(device)
    y_low_test = y_low[:N].to(device)
    y_pred = model(x_test, y_low_test).to("cpu").detach().numpy()

    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print("y_pred:\n", y_pred[:, :7],y_pred[:, -7:])
    print("y_true:\n", y_high[:N].detach().numpy()[:, :7], y_high[:N].detach().numpy()[:, -7:])

    # for i in range(1, 50):
    #     x_test = x_low[i-1:i].to(device)
    #     y_low_test = y_low[i-1:i].to(device)

    #     y_pred = model(x_test, y_low_test).to("cpu").detach().numpy()[0]
    #     y_true = y_high[i].detach().numpy()

    #     plt.plot(y_low[i]/100, r_2d, "-b", label="2d")
    #     plt.plot(y_pred/100, r_3d, '-r',label="pred")
    #     plt.plot(y_true/100, r_3d, '--k',label="true")
    #     plt.legend()
    #     plt.savefig(fr"./imgs/ft/{i}.png")
    #     plt.close()
