from mfnn.mfnn import FCNN
from mfnn.xydata import XYZDataSet
import torch
import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
from mfnn.data_loader import loadData

x_low,y_low, y_high, loader_high, r_2d, r_3d, xt_low, yt_low, yt_high = loadData() 
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 
print(x_low.shape, y_high.shape)

## plot
# figure1(x_low, y_low, x_high,  y_high, x, y)

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 
model = FCNN(
     x_low.size()[-1], 
     y_high.size()[-1], 
    [128, 256, 512, 256], 
     0, 
     torch.nn.LeakyReLU, 
    # torch.nn.ReLU, 
     low_fidelity_features=y_low.size()[-1],
     enable_attention=False,
     enable_embedding=True
     )

model.to(torch.double)
model_path = "./data/mlp_embedding.model"
loss_path = "./data/mlp_embedding.loss"

def train(model, dataloader, critrion, optimizer, steps, device="cpu"):
    "Train the model by given dataloader."
    t_start = time.time()
    train_loss, test_loss = [], []
    _xt_low, _yt_low = xt_low.to(device), yt_low.to(device)
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
        # run test
        yt_pred = model(_xt_low, _yt_low).to("cpu")
        loss_test = critrion(yt_pred, yt_high).item()
        train_loss.append(runningLoss)
        test_loss.append(loss_test)

        scheduler.step()
    print("training complete", f'wall time = {time.time()- t_start:.2f}s')
    torch.save(model.state_dict(), model_path)
    torch.save({"train": train_loss, "test": test_loss}, loss_path)




if __name__ == "__main__":
    #  print("x_low", x_low.shape, y_high.shape, y_low.shape)
    from autoencoder import loadModel
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
    STEPS = 5000
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5*STEPS, 0.8*STEPS])
    critrion = torch.nn.L1Loss()
    critrion = torch.nn.MSELoss()
    critrion = torch.nn.SmoothL1Loss()

    train(model, loader_high, critrion, optimizer, STEPS, device)
    ## eval
    N=1
    model.load_state_dict(state_dict=torch.load(model_path))
    model.eval()
    x_test = x_low[:N].to(device)
    y_low_test = y_low[:N].to(device)
    y_pred = model(x_test, y_low_test).to("cpu").detach().numpy()

    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print("y_pred:\n", y_pred)
    # print("y_true:\n", y_high_encoded.detach().numpy())

    # ae_model = loadModel()
    # for i in range(1, 50):
    #     x_test = x_low[i-1:i].to(device)
    #     y_low_test = y_low[i-1:i].to(device)

    #     y_pred = model(x_test, y_low_test).to("cpu").detach().numpy()[0]
    #     # y_pred = model(x_test, y_low_test).to("cpu")
    #     # y_pred = ae_model.decoder(y_pred).detach().numpy()[0]
    #     y_true = y_high[i-1].detach().numpy()

    #     plt.plot(y_low[i-1]/100, r_2d, "-b", label="2d")
    #     plt.plot(y_pred/100, r_3d, '-r',label="pred")
    #     plt.plot(y_true/100, r_3d, '--k',label="true")
    #     plt.legend()
    #     plt.savefig(fr"./imgs/mlp/{i}.png")
    #     plt.close()
