import os, sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from glob import glob
from torch.utils.data import DataLoader
from .trainers.GANTrainer import GANTrainer
from .trainers.PatchNCETrainer import PatchNCETrainer

def train(models_dict, loss_per_minibatch, X_dataloader, Y_dataloader,
          device="cpu", n_epochs=400, print_every=100, checkpoint_epoch=1):
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

    # init from models_dict
    D, solver_D, scheduler_D = models_dict["discriminator"]
    G, solver_G, scheduler_G = models_dict["generator"]
    P, solver_P, scheduler_P = models_dict["patchNCE"]

    # init iterator to draw (random) samples from Y_dataloader
    Y_iter = iter(Y_dataloader)

    # variables to print progress
    batch_size = len(X_dataloader)

    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))

        for n_batch, real_X in enumerate(X_dataloader):
            real_X = real_X.to(device)

            # train discriminator
            loss_D = GANTrainer.train_discriminator(G, D, solver_D, real_X, device)
            loss_D = loss_D.item()

            # train generator
            loss_G, fake_X = GANTrainer.train_generator(G, D, solver_G, real_X.shape, device)
            loss_G = loss_G.item()

            # train PatchNCE
            loss_P_X = PatchNCETrainer.train_patchnce(P, solver_P, real_X, fake_X, device)
            loss_P = loss_P_X.item()

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
            loss_P_Y = PatchNCETrainer.train_patchnce(P, solver_P, real_Y, fake_Y, device)
            loss_P += loss_P_Y.item()

            loss_per_minibatch["discriminator"].append(loss_D)
            loss_per_minibatch["generator"].append(loss_G)
            loss_per_minibatch["patchNCE"].append(loss_P)

            # print the first minibatch, then every `print_every` minibatches
            if print_every and (not n_batch) or ((n_batch + 1) % print_every == 0):
                print("Iteration {}/{}, loss_D: {:e}, loss_G: {:e}, loss_P: {:e}"
                      .format(n_batch + 1, batch_size, loss_D, loss_G, loss_P))

            # # manage GPU memory
            # del loss_D, loss_G, loss_P
            # del real_X, real_Y, fake_X, fake_Y, noise

        scheduler_D.step()
        scheduler_G.step()
        scheduler_P.step()

        # save model params every `checkpoint_epoch` epochs
        if checkpoint_epoch and ((epoch + 1) % checkpoint_epoch == 0):
            # print("Deleting previous model checkpoints...")
            # purge_files(currentdir + "/checkpoint*.pt")

            print("Saving model checkpoints after {} epochs...".format(epoch + 1))
            models_dict = {
                "discriminator": (D, solver_D, scheduler_D),
                "generator": (G, solver_G, scheduler_G),
                "patchNCE": (P, solver_P, scheduler_P)
            }
            save_models(models_dict, epoch + 1)

            print("Saving losses...")
            save_losses(loss_per_minibatch, epoch + 1)

    return D, G, P, loss_per_minibatch


def purge_files(filelist):
    for file_ in glob(filelist):
        os.remove(file_)


def save_models(models_dict, epoch):
    for model_name, model_solver_scheduler in models_dict.items():
        model, solver, scheduler = model_solver_scheduler
        state = {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "solver_state_dict": solver.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch
        }
        filename = "checkpoint_{}_{}.pt".format(model_name, epoch)
        torch.save(state, filename)


def save_losses(loss_per_minibatch, epoch):
    torch.save(loss_per_minibatch, "checkpoint_losses_{}.pt".format(epoch))
