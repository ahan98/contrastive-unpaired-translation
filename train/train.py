import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from torch import optim, randn
from models.GAN.Discriminator import Discriminator
from models.GAN.Generator import Generator
from models.PatchNCE.PatchNCE import PatchNCE
from .train_fns.GAN_fns import train_D, train_G
from .train_fns.patch_nce_fns import train_P
from data_utils.process_data import shuffled_data_loader

# TODO: write logger

def train(X_dataset, Y_dataset, n_epochs=400, n_steps_D=1, lr=2e-3, print_every=100):
    """
    Train all networks (Discriminator, Generator, PatchNCE).

    For implementation details of the training functions, see documentation for
    .train_fns.GAN_fns and .train_fns.patch_fns, as well as sections 3 and C.1
    of Park et al.

    Inputs:
    - [BatchDataset] X_dataset: Iterable batch of image tensors
    - [BatchDataset] Y_dataset: Iterable batch of image tensors
    - [int] n_epochs: number of iterations for X_dataset
    - [int] n_steps_D: number of steps to train discriminator per minibatch
    - [float] lr: learning rate for Adam optimizer
    - [int] print_every: print every X iterations

    Returns None
    """

    # init networks
    D = Discriminator()
    G = Generator()
    P = PatchNCE(G.encoder)

    # init solvers
    solver_D = optim.Adam(D.parameters(), lr=lr)
    solver_G = optim.Adam(G.parameters(), lr=lr)
    solver_P = optim.Adam(P.parameters(), lr=lr)

    shuffled_Y = shuffled_data_loader(Y_dataset)
    Y_iter = iter(shuffled_Y)

    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch, n_epochs))

        for n_batch, real_X in enumerate(X_dataset):

            # train discriminator
            for _ in range(n_steps_D):
                loss_D, pred_real_D, pred_fake_D = train_D(D, solver_D, real_X)

            # train generator
            loss_G, fake_X = train_G(G, D, solver_G, real_X.shape)

            # train PatchNCE
            loss_P = train_P(P, solver_P, real_X, fake_X)

            # get random sample from Y, treating it as the "real" data
            try:
                real_Y = next(Y_iter)
            except StopIteration:
                # reshuffle Dataloader if all samples have been used once
                Y_iter = iter(shuffled_Y)
                real_Y = next(Y_iter)

            # train PatchNCE again, this time comparing real and fake images
            # from Y dataset
            noise = randn(real_Y.shape)
            fake_Y = G(noise)
            loss_P += train_P(P, solver_P, real_Y, fake_Y)

        if n_batch % print_every == 0:
            print("loss_D: {:e}, loss_G: {:e}, loss_P: {:e}"
                  .format(loss_D, loss_G, loss_P))

