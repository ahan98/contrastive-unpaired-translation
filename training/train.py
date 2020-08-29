import torch
from torch.utils.data import DataLoader
from .trainers.PatchNCETrainer import PatchNCETrainer
from .trainers.GANTrainer import GANTrainer
from .checkpoint_utils import save_models, save_losses


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
    discriminator, solver_discriminator = models_dict["discriminator"]
    generator, solver_generator = models_dict["generator"]
    patchNCE, solver_patchNCE = models_dict["patchNCE"]

    # init iterator to draw (random) samples from Y_dataloader
    Y_iter = iter(Y_dataloader)

    # variables to print progress
    batch_size = len(X_dataloader)

    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))

        for n_batch, real_X in enumerate(X_dataloader):

            ### LOAD SAMPLES ###

            real_X = real_X.to(device)

            # get random sample from Y
            try:
                real_Y = next(Y_iter).to(device)
            except StopIteration:
                # reshuffle Dataloader if all samples have been used once
                Y_iter = iter(Y_dataloader)
                real_Y = next(Y_iter).to(device)

            ### DISCRIMINATOR ###

            fake_Y1 = generator(real_Y)
            fake_Y1_detached = fake_Y1.detach()

            # make sure discriminator requires grad before training
            set_requires_grad(discriminator, True)
            solver_discriminator.zero_grad()

            # train discriminator
            loss_discriminator = \
                GANTrainer.train_discriminator(discriminator, real_Y,
                                               fake_Y1_detached, device)

            loss_discriminator.backward()
            solver_discriminator.step()

            ### GENERATOR ###

            # shutoff backprop for discriminator while training generator
            set_requires_grad(discriminator, False)
            solver_generator.zero_grad()

            # train generator
            loss_generator = GANTrainer.train_generator(discriminator, fake_Y1,
                                                        device)

            ### PATCHNCE ###

            # train PatchNCE
            loss_patchNCE_X = PatchNCETrainer.train_patchnce(patchNCE, real_X,
                                                             fake_Y1, device)

            # get random sample from Y
            try:
                real_Y = next(Y_iter).to(device)
            except StopIteration:
                # reshuffle Dataloader if all samples have been used once
                Y_iter = iter(Y_dataloader)
                real_Y = next(Y_iter).to(device)

            fake_Y2 = generator(real_Y)
            loss_patchNCE_Y = PatchNCETrainer.train_patchnce(patchNCE, fake_Y1,
                                                             fake_Y2, device)

            loss_patchNCE_average = 0.5 * (loss_patchNCE_X + loss_patchNCE_Y)
            loss_generator_total = loss_generator + loss_patchNCE_average

            loss_generator_total.backward()
            solver_generator.step()
            solver_patchNCE.step()

            loss_discriminator = loss_discriminator.item()
            loss_generator_total = loss_generator_total.item()
            loss_patchNCE_average = loss_patchNCE_average.item()

            loss_per_minibatch["discriminator"].append(loss_discriminator)
            loss_per_minibatch["generator"].append(loss_generator_total)
            loss_per_minibatch["patchNCE"].append(loss_patchNCE_average)

            # print the first minibatch, then every `print_every` minibatches
            if print_every and (n_batch == 0) or ((n_batch + 1) % print_every == 0):
                print(("Iteration {}/{}, loss_discriminator: {:e}, "
                       "loss_generator: {:e}, loss_patchNCE: {:e}")
                      .format(n_batch + 1, batch_size, loss_discriminator,
                              loss_generator_total, loss_patchNCE_average))

        # update model checkpoints after every epoch
        models_dict = {
            "discriminator": (discriminator, solver_discriminator),
            "generator": (generator, solver_generator),
            "patchNCE": (patchNCE, solver_patchNCE)
        }

        # save checkpoints every `checkpoint_epoch` epochs
        if checkpoint_epoch and ((epoch + 1) % checkpoint_epoch == 0):
            save_models(models_dict, epoch + 1)
            save_losses(loss_per_minibatch, epoch + 1)

    # since models_dict is updated every epoch, we are returning the model
    # states after all training is done
    return models_dict, loss_per_minibatch


def set_requires_grad(network, requires_grad):
    for param in network.parameters():
        param.requires_grad = requires_grad
