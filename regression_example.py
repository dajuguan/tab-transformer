import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from tab_transformer_pytorch import FTTransformer

cont_mean_std = torch.randn(10, 2)

model = TabTransformer(
    categories=(),
    num_continuous = 10,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 1,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1,                   # feed forward dropout
    mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
)

model = FTTransformer(
    categories = (),      # tuple containing the number of unique values within each category
    num_continuous = 3,                # number of continuous values
    dim = 32,                           # dimension, paper set at 32
    dim_out = 3,                        # binary prediction, but could be anything
    depth = 6,                          # depth, paper recommended 6
    heads = 8,                          # heads, paper recommends 8
    d_rope = 4,                         # the first d_rope features that needs positional encoding
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1                    # feed forward dropout
)

import numpy as np

x_categ = torch.from_numpy(np.array([]))
N=100
x_data = torch.rand(100, 3)
y_data = torch.Tensor([(x_data[:,0]*2).tolist(), (x_data[:,1]*3).tolist(),(x_data[:,2]*4).tolist()]).transpose(0,1)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
STEPS = 1500
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.4 * STEPS, 0.7 * STEPS], gamma=0.1)


for epoch in range(STEPS):
    y_pred = model(x_categ, x_data)
    loss = criterion(y_pred, y_data)
    if epoch % 50 == 0:
        print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

model.eval()
N=4
pred, attn = model(x_categ,x_data[0:N,:], True) # (1, 1)
print("true:\n", y_data[0:N,:])
print("pred:\n", pred)
print("attn:", attn.size())