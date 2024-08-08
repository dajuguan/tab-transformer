import torch
import os
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from mfnn.data_loader import loadData

x_low,y_low, y_high, loader_high, r_2d, r_3d = loadData() 
X_train = torch.cat((x_low,y_low), dim=1).numpy()
y_train = y_high.numpy()
print(X_train.shape, y_train.shape)

clf = TabNetRegressor()

max_epochs = 5000 if not os.getenv("CI", False) else 2

# from pytorch_tabnet.augmentations import RegressionSMOTE
# aug = RegressionSMOTE(p=0.2)

if __name__ == "__main__":
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train)],
        eval_name=['train'],
        eval_metric=['rmsle', 'mae', 'rmse', 'mse'],
        max_epochs=max_epochs,
        patience=100,
        batch_size=len(X_train), 
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
        # loss_fn=torch.nn.MSELoss(),
        loss_fn=torch.nn.L1Loss()
        # augmentations=aug, #aug
    ) 

    saving_path_name = "./data/tabnet.model"
    saved_filepath = clf.save_model(saving_path_name)

    preds = clf.predict(X_train[:1])

    y_true = y_train[:1]

    test_score = mean_squared_error(y_pred=preds, y_true=y_true)

    print(f"BEST VALID SCORE: {clf.best_cost}")
    print(f"FINAL TEST SCORE: {test_score}")

