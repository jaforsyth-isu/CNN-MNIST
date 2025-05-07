#Adapted from code by oniani
#https://github.com/oniani/mnist

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import MNISTModel
from train import train
from test import test

if __name__ == "__main__":
    test_only = False
    use_cuda = True
    model_file = "MNIST_Model"
    use_saved_model = False

    model = MNISTModel()

    if test_only:
        model.load_state_dict(torch.load(model_file))
        print(f"Accuracy: {test(model=model, use_cuda=use_cuda)}")
    else:
        if use_saved_model:
            model.load_state_dict(torch.load(model_file))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = 0.0001
        )
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.1,
            patience=10
        )

        train(model=model,
              epochs=10,
              batch_size=60,
              optimizer=optimizer,
              criterion=criterion,
              scheduler=scheduler,
              use_cuda=use_cuda)

        # Save the model
        torch.save(model.state_dict(),
                   os.path.join(model_file))
