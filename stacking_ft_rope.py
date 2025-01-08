import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from tab_transformer_pytorch import FTTransformer
from mfnn.data_loader import loadStackingData
import numpy as np
import time
import tqdm


X, Y, dataloader, dim_rope_seq, xt, yt = loadStackingData()
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu") 
model = FTTransformer(
    categories = (),      # tuple containing the number of unique values within each category
    num_continuous = X.shape[-1],                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = Y.shape[-1],              # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    dim_rope_seq = dim_rope_seq,        # the first d_rope features that needs positional encoding
    attn_dropout = 0.00,                # post-attention dropout
    ff_dropout = 0.00                   # feed forward dropout
)

model_path = "./data/ft_rope_stacking.model"
loss_path = "./data/ft_rope_stacking.loss"

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
                y_pred = model(torch.tensor([]), x)
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

            yt_pred = model(torch.tensor([]), _xt).to("cpu")
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

    # train(model, dataloader, critrion, optimizer, STEPS, device)
    ## eval
    N=1
    model.eval()
    x_test = X[:N].to(device)
    y_low_test = Y[:N].to(device)
    y_pred = model(torch.tensor([]), x_test).to("cpu").detach().numpy()

    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print("y_pred:\n", y_pred)
    print("y_true:\n", Y[:N].detach().numpy())