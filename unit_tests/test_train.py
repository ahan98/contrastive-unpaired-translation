import torch
from torch.utils.data import DataLoader
from training.checkpoint_utils import load_models_and_losses
from training.train import train


def test():
    batch_shape = (2, 3, 256, 256)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    X_batch = torch.zeros(batch_shape)
    X_dataloader = DataLoader(X_batch)

    Y_batch = torch.zeros(batch_shape)
    Y_dataloader = DataLoader(Y_batch, shuffle=True)

    models_dict, loss_per_minibatch = load_models_and_losses(device=device)
    train(models_dict, loss_per_minibatch, X_dataloader, Y_dataloader,
          device=device, n_epochs=1, checkpoint_epoch=0)
    print("Losses per minibatch", loss_per_minibatch)
    print("Train test passed")
    return True


if __name__ == "__main__":
    test()
