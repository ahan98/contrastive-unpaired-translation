import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from torch.utils.data import DataLoader
from training.train import train

def test():
    batch_shape = (2, 3, 256, 256)
    minibatch_size = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    X_batch = torch.zeros(batch_shape)
    X_dataloader = DataLoader(X_batch)

    Y_batch = torch.zeros(batch_shape)
    Y_dataloader = DataLoader(Y_batch, shuffle=True)

    train(X_dataloader, Y_dataloader, n_epochs=1, device=device)

    print("Train test passed")
    return True

if __name__ == "__main__":
    test()
