import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from torch.utils.data import DataLoader
from models.GAN.Discriminator import Discriminator
from models.GAN.Generator import Generator
from models.PatchNCE.PatchNCE import PatchNCE
from .trainers.GANTrainer import GANTrainer
from .trainers.PatchNCETrainer import PatchNCETrainer
from tqdm import tqdm

def train(X_dataloader, Y_dataloader, device="cpu", n_epochs=400,
          learning_rates=(2e-3, 2e-3, 2e-3), betas=(0.9, 0.999), print_every=100):
    """
    Train all networks (Discriminator, Generator, PatchNCE).

    For implementation details of the training functions, see documentation for
    .train_fns.GAN_fns and .train_fns.patch_fns, as well as sections 3 and C.1
    of Park et al.

    Inputs:
    - [DataLoader] X_dataloader: iterable dataset to be sequentially sampled
    - [DataLoader] Y_dataloader: iterable dataset to be randomly sampled
    - [int] n_epochs: number of iterations for X_dataloader
    - [float] lr: learning rate for Adam optimizer
    - [int] print_every: print every X iterations
    - [String] device: name of device to load data (e.g., "cuda:0")

    Returns all trained networks and their loss histories.
    """

    # init networks
    D = Discriminator().to(device)
    G = Generator().to(device)
    P = PatchNCE(G.encoder).to(device)

    # init solvers
    lr_D, lr_G, lr_P = learning_rates
    solver_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=betas)
    solver_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=betas)
    solver_P = torch.optim.Adam(P.parameters(), lr=lr_P, betas=betas)

    # init iterator to draw (random) samples from Y_dataloader
    Y_iter = iter(Y_dataloader)

    # dictionary of losses per minibatch
    loss_histories = {"discriminator": [], "generator": [], "patchNCE": []}

    # variables to print progress
    n_iter = 0
    batch_size = len(X_dataloader)

    for epoch in tqdm(range(n_epochs)):
        print("Epoch {}/{}".format(epoch, n_epochs))

        for n_batch, real_X in enumerate(X_dataloader):
            real_X = real_X.to(device)

            # train discriminator
            loss_D = GANTrainer.train_discriminator(G, D, solver_D, real_X, device)

            # train generator
            loss_G, fake_X = GANTrainer.train_generator(G, D, solver_G, real_X.shape, device)

            # train PatchNCE
            loss_P = PatchNCETrainer.train_patchnce(P, solver_P, real_X, fake_X, device)

            # get random sample from Y, treating it as the "real" data
            try:
                real_Y = next(Y_iter).to(device)
            except StopIteration:
                # reshuffle Dataloader if all samples have been used once
                Y_iter = iter(Y_dataloader)
                real_Y = next(Y_iter).to(device)

            real_Y = real_Y.to(device)

            # train PatchNCE again, this time comparing real and fake images
            # from Y dataset
            noise = torch.randn(real_Y.shape, device=device)
            fake_Y = G(noise)
            loss_P += PatchNCETrainer.train_patchnce(P, solver_P, real_Y, fake_Y, device)

            # store loss for this minibatch
            loss_histories["discriminator"].append(loss_D)
            loss_histories["generator"].append(loss_G)
            loss_histories["patchNCE"].append(loss_P)

            n_iter += 1
            if n_iter % print_every == 0:
                print("iteration: {}/{}, loss_D: {:e}, loss_G: {:e}, loss_P: {:e}"
                      .format(n_iter, batch_size, loss_D, loss_G, loss_P))

    return D, G, P, loss_histories

