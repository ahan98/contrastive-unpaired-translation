"""
A collection of helper methods for loading and saving model states.
"""

from glob import glob
import torch
from torch import optim as optim
from torchvision.datasets import ImageFolder
from models.GAN.Discriminator import Discriminator
from models.GAN.Generator import Generator
from models.PatchNCE.PatchNCE import PatchNCE


def load_models_and_losses(lr_discriminator=2e-3, lr_generator=2e-3, lr_patchNCE=2e-3,
                           checkpoint_files_dict=None, device="cpu"):
    # init models
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    patchNCE = PatchNCE(generator.encoder).to(device)

    # init Adam optimizers
    print("lr_discriminator = {:e}, lr_generator = {:e}, lr_patchNCE = {:e}"
          .format(lr_discriminator, lr_generator, lr_patchNCE))
    solver_discriminator = \
        optim.Adam(discriminator.parameters(), lr=lr_discriminator)
    solver_generator = \
        optim.Adam(generator.parameters(), lr=lr_generator)
    solver_patchNCE = \
        optim.Adam(patchNCE.parameters(), lr=lr_patchNCE)

    # # init schedulers
    # scheduler_D = optim.lr_scheduler.StepLR(solver_D, step_size=20, gamma=0.1)
    # scheduler_G = optim.lr_scheduler.StepLR(solver_G, step_size=20, gamma=0.1)
    # scheduler_P = optim.lr_scheduler.StepLR(solver_P, step_size=20, gamma=0.1)

    loss_per_minibatch = {"discriminator": [], "generator": [], "patchNCE": []}

    if checkpoint_files_dict:

        print("Loading checkpoint for loss...")
        loss_state = checkpoint_files_dict["loss"]
        loss_per_minibatch = torch.load(loss_state)

        print("Loading checkpoint for discriminator...")
        discriminator_states = checkpoint_files_dict["discriminator"]
        _load_states(discriminator_states, discriminator, solver_discriminator)

        print("Loading checkpoint for generator...")
        generator_states = checkpoint_files_dict["generator"]
        _load_states(generator_states, generator, solver_generator)

        print("Loading checkpoint for patchNCE...")
        patchNCE_states = checkpoint_files_dict["patchNCE"]
        _load_states(patchNCE_states, patchNCE, solver_patchNCE)

    models_dict = {
        "discriminator": (discriminator, solver_discriminator),
        "generator": (generator, solver_generator),
        "patchNCE": (patchNCE, solver_patchNCE)
    }

    return models_dict, loss_per_minibatch


def _load_states(states, model, solver):
    model_state = states["model_state_dict"]
    model.load_state_dict(model_state)
    solver_state = states["solver_state_dict"]
    solver.load_state_dict(solver_state)


def save_models(models_dict, epoch):
    print("Saving model checkpoints after", epoch, "epochs...")
    for model_name, model_solver in models_dict.items():
        model, solver = model_solver
        state = {
            "model_state_dict": model.state_dict(),
            "solver_state_dict": solver.state_dict(),
            "epoch": epoch
        }
        filename = "checkpoint_{}_{}.pt".format(model_name, epoch)
        torch.save(state, filename)


def save_losses(loss_per_minibatch, epoch):
    print("Saving losses after", epoch, "epochs...")
    torch.save(loss_per_minibatch, "checkpoint_losses_{}.pt".format(epoch))
