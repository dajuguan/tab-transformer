import torch
import torch.utils
import torch.utils.data
import numpy as np

class AE(torch.nn.Module):
    def __init__(self, inDim, encodedDim):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(inDim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, encodedDim)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encodedDim, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, inDim)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


loss_3d = np.loadtxt(r"/home/po/phd/tab-transformer-pytorch/data/loss_3d.txt", dtype=np.float64) 
loss_3d = torch.from_numpy(loss_3d)
dataset = torch.utils.data.dataset.TensorDataset(loss_3d)
loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = len(loss_3d), shuffle = False)

model = AE(loss_3d.shape[-1], 16)
model.to(torch.double)
model_path = r"/home/po/phd/tab-transformer-pytorch/data/ae.model"

def loadModel():
    model.load_state_dict(state_dict=torch.load(model_path))
    return model

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-8)
    epochs = 1800
    def train():
        for epoch in range(epochs):
            runningLoss = 0.0
            for (xx) in loader:
                xx = xx[0]
                # Output of Autoencoder
                reconstructed = model(xx)
                
                # Calculating the loss function
                loss = loss_function(reconstructed, xx)
                
                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                runningLoss += loss.item()
            print(f"epoch: {epoch}, loss: {runningLoss}")
        torch.save(model.state_dict(), model_path)

    train()
    model.load_state_dict(state_dict=torch.load(model_path))
    loss_3d_encoded = model.encoder(loss_3d)
    np.savetxt(r"/home/po/phd/tab-transformer-pytorch/data/loss_3d_encoded.txt", loss_3d_encoded.detach().numpy())

    # r_3d = np.loadtxt(r"/home/po/phd/tab-transformer-pytorch/data/r_3d.txt")
    # for i in range(len(loss_3d)):
    #     pred_3d = model(loss_3d[i:i+1]).detach().numpy()[0]
    #     plt.plot(pred_3d, r_3d,  label="pred")
    #     plt.plot(loss_3d[i], r_3d,  label="true")
    #     plt.legend()
    #     plt.show()