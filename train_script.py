from training.train import train
from training.train_utils import *
import torch
import torchvision.transforms as T
import os
from glob import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device", device)

# define path to image folders
path_to_train = "/dataset/horse2zebra/"
X_class_name = "trainA"
Y_class_name = "trainB"

print("Loading training data...")
batches_by_class = get_batches_from_path(path_to_train)
X_train_dataloader = make_dataloader(batches_by_class, X_class_name)
Y_train_dataloader = make_dataloader(batches_by_class, Y_class_name, replacement=True)

# Hyperparameters
lr_D, lr_G, lr_P = 8e-8, 5e-7, 2e-3

# init models
print("Initializing models...")
path_to_checkpoints = "/checkpoints/"
models_dict, loss_per_minibatch = load_models_and_losses(path_to_checkpoints,
                                                         lr_D, lr_G, lr_P,
                                                         device)

n_epochs = 340
print_every = len(X_train_dataloader)
checkpoint_epoch = 5

D, G, P, loss_per_minibatch = \
    train(models_dict, loss_per_minibatch, X_train_dataloader,
          Y_train_dataloader, device=device, n_epochs=n_epochs,
          print_every=print_every, checkpoint_epoch=checkpoint_epoch)
